import os
import json
import math
import requests
import numpy as np
import pandas as pd

from concurrent.futures import ThreadPoolExecutor, as_completed


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = "data"
OUTPUT_DIR = "outputs"

SURVIVAL_DATA = os.path.join(
    DATA_DIR,
    "survival_ml_cleaned.csv"
)

CANDIDATE_DATA = os.path.join(
    DATA_DIR,
    "pune_final_candidate_pool.csv"
)

V5_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "candidate_v5_predictions.csv"
)

EVIDENCE_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "candidate_evidence_scores.csv"
)

SOIL_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "candidate_soil_compatibility.csv"
)

POLLUTION_CAPABILITY = os.path.join(
    DATA_DIR,
    "pollution_capability.csv"
)

FINAL_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "v6_final_ranking.csv"
)

CLIMATE_CACHE = os.path.join(
    OUTPUT_DIR,
    "v6_climate_cache.csv"
)

POLLUTION_CURRENT = os.path.join(
    OUTPUT_DIR,
    "pollution_current.json"
)

POLLUTION_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "pollution_scores.csv"
)


# ============================================================
# LOCATION
# ============================================================

PUNE_LAT = 18.5204
PUNE_LON = 73.8567


# ============================================================
# FINAL WEIGHTS
# ============================================================

WEIGHT_V5 = 0.40
WEIGHT_EVIDENCE = 0.15
WEIGHT_SOIL = 0.15
WEIGHT_CLIMATE = 0.10
WEIGHT_POLLUTION = 0.10
WEIGHT_USER = 0.10


# ============================================================
# CLIMATE VARIABLES
# ============================================================

CLIMATE_VARIABLES = [
    "T2M",
    "T2M_MAX",
    "T2M_MIN",
    "PRECTOTCORR",
    "RH2M",
    "ALLSKY_SFC_SW_DWN",
    "WS10M",
    "EVPTRNS",
]


# ============================================================
# HELPERS
# ============================================================

def normalize_species(value):

    if pd.isna(value):
        return ""

    parts = str(value).strip().lower().split()

    if len(parts) >= 2:
        return f"{parts[0]} {parts[1]}"

    return str(value).strip().lower()


def find_column(df, candidates):

    for col in candidates:

        if col in df.columns:
            return col

    return None


# ============================================================
# CLIMATE API
# ============================================================

def get_climate(lat, lon):

    url = (
        "https://power.larc.nasa.gov/api/"
        "temporal/climatology/point"
    )

    params = {
        "parameters": ",".join(CLIMATE_VARIABLES),
        "community": "AG",
        "longitude": float(lon),
        "latitude": float(lat),
        "format": "JSON",
    }

    try:

        response = requests.get(
            url,
            params=params,
            timeout=30
        )

        response.raise_for_status()

        data = response.json()

        values = data["properties"]["parameter"]

        result = {
            "lat_dec": float(lat),
            "lon_dec": float(lon),
        }

        for variable in CLIMATE_VARIABLES:

            vals = []

            for value in values.get(
                variable,
                {}
            ).values():

                try:

                    value = float(value)

                    if math.isfinite(value):
                        vals.append(value)

                except Exception:
                    pass

            if vals:

                result[
                    f"{variable}_annual"
                ] = float(
                    np.mean(vals)
                )

            else:

                result[
                    f"{variable}_annual"
                ] = np.nan

        return result

    except Exception as e:

        print(
            f"Climate request failed "
            f"{lat}, {lon}: {e}"
        )

        return None


# ============================================================
# POLLUTION API
# ============================================================

def get_air_quality(lat, lon):

    url = (
        "https://air-quality-api.open-meteo.com/"
        "v1/air-quality"
    )

    params = {
        "latitude": float(lat),
        "longitude": float(lon),
        "current": (
            "pm2_5,"
            "pm10,"
            "nitrogen_dioxide,"
            "sulphur_dioxide,"
            "ozone"
        ),
        "timezone": "auto",
    }

    response = requests.get(
        url,
        params=params,
        timeout=20
    )

    response.raise_for_status()

    current = response.json().get(
        "current",
        {}
    )

    return {
        "time": current.get("time"),
        "pm2_5": current.get("pm2_5"),
        "pm10": current.get("pm10"),
        "no2": current.get(
            "nitrogen_dioxide"
        ),
        "so2": current.get(
            "sulphur_dioxide"
        ),
        "o3": current.get(
            "ozone"
        ),
    }


def pollution_pressure(value, low, high):

    if value is None:
        return None

    value = float(value)

    if value <= low:
        return 0.0

    if value >= high:
        return 1.0

    return (
        (value - low)
        / (high - low)
    )


def calculate_pollution_severity(air):

    """
    Prototype pollution-pressure index.

    This is NOT official AQI.

    Thresholds follow the general pollutant
    concentration bands exposed by Open-Meteo's
    air-quality documentation.
    """

    values = {

        "pm2_5": pollution_pressure(
            air["pm2_5"],
            5,
            75
        ),

        "pm10": pollution_pressure(
            air["pm10"],
            15,
            150
        ),

        "no2": pollution_pressure(
            air["no2"],
            10,
            100
        ),

        "so2": pollution_pressure(
            air["so2"],
            20,
            125
        ),

        "o3": pollution_pressure(
            air["o3"],
            60,
            180
        ),
    }

    weights = {
        "pm2_5": 0.35,
        "pm10": 0.25,
        "no2": 0.20,
        "so2": 0.10,
        "o3": 0.10,
    }

    valid = {
        k: v
        for k, v in values.items()
        if v is not None
    }

    weighted = sum(
        valid[k] * weights[k]
        for k in valid
    )

    total_weight = sum(
        weights[k]
        for k in valid
    )

    return weighted / total_weight


# ============================================================
# POLLUTION EVIDENCE
# ============================================================

def load_pollution():

    df = pd.read_csv(
        POLLUTION_CAPABILITY
    )

    df["species_key"] = (
        df["species_normalized"]
        .apply(normalize_species)
    )

    df["pollution_response"] = pd.to_numeric(
        df["pollution_response"],
        errors="coerce"
    )

    df["pollution_response"] = (
        df["pollution_response"]
        .fillna(0.5)
        .clip(0, 1)
    )

    return df


def build_pollution_scores():

    print("\n" + "=" * 70)
    print("POLLUTION")
    print("=" * 70)

    air = get_air_quality(
        PUNE_LAT,
        PUNE_LON
    )

    severity = calculate_pollution_severity(
        air
    )

    evidence = load_pollution()

    # Pollution benefit:
    #
    # environmental pressure
    # ×
    # evidence-backed species response
    #
    # This does NOT claim kg/tree removal.

    evidence["pollution_score"] = (
        severity
        * evidence["pollution_response"]
        * 100
    )

    evidence["pollution_score"] = (
        evidence["pollution_score"]
        .clip(0, 100)
        .round(2)
    )

    with open(
        POLLUTION_CURRENT,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            {
                "latitude": PUNE_LAT,
                "longitude": PUNE_LON,
                "air_quality": air,
                "pollution_severity": severity,
            },
            f,
            indent=2
        )

    evidence.to_csv(
        POLLUTION_OUTPUT,
        index=False
    )

    print(
        "\nCurrent pollution:"
    )

    for key, value in air.items():

        print(
            f"{key}: {value}"
        )

    print(
        f"\nPollution severity: "
        f"{severity:.4f}"
    )

    print(
        "\nPollution ranking:"
    )

    print(
        evidence[
            [
                "species_normalized",
                "pollution_response",
                "evidence_level",
                "pollution_score",
            ]
        ]
        .sort_values(
            "pollution_score",
            ascending=False
        )
        .to_string(index=False)
    )

    return evidence, severity


# ============================================================
# ROBUST CLIMATE SCORE
# ============================================================

def calculate_climate_score(
    species_rows,
    pune_climate
):

    scores = []

    for variable in CLIMATE_VARIABLES:

        column = f"{variable}_annual"

        values = pd.to_numeric(
            species_rows[column],
            errors="coerce"
        ).dropna()

        pune_value = pune_climate.get(
            column
        )

        if len(values) < 3:
            continue

        if pd.isna(pune_value):
            continue

        q20 = values.quantile(0.20)
        q80 = values.quantile(0.80)

        median = values.median()

        spread = q80 - q20

        if spread < 1e-6:

            spread = max(
                abs(median) * 0.05,
                0.1
            )

        distance = (
            abs(
                float(pune_value)
                - float(median)
            )
            / spread
        )

        # Smooth score.
        #
        # Inside historical 20-80% range:
        # approximately high suitability.
        #
        # Outside range:
        # gradually decreases rather than
        # collapsing immediately to zero.

        score = (
            100
            * math.exp(
                -0.5
                * distance
                * distance
            )
        )

        scores.append(score)

    if not scores:
        return 50.0

    return float(
        np.mean(scores)
    )


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("FINAL V6 RECOMMENDATION PROTOTYPE")
print("=" * 70)

print("\nLoading datasets...")

survival = pd.read_csv(
    SURVIVAL_DATA
)

candidates = pd.read_csv(
    CANDIDATE_DATA
)

v5 = pd.read_csv(
    V5_OUTPUT
)

evidence = pd.read_csv(
    EVIDENCE_OUTPUT
)

soil = pd.read_csv(
    SOIL_OUTPUT
)


print(
    f"Survival: {survival.shape}"
)

print(
    f"Candidates: {candidates.shape}"
)

print(
    f"V5: {v5.shape}"
)

print(
    f"Evidence: {evidence.shape}"
)

print(
    f"Soil: {soil.shape}"
)


# ============================================================
# SPECIES KEYS
# ============================================================

survival_col = find_column(
    survival,
    [
        "species_full",
        "species_normalized",
    ]
)

candidate_col = find_column(
    candidates,
    [
        "species_normalized",
        "species_full",
        "species",
    ]
)

v5_col = find_column(
    v5,
    [
        "species_normalized",
        "species_full",
        "species",
    ]
)

evidence_col = find_column(
    evidence,
    [
        "species_normalized",
        "species_full",
        "species",
    ]
)

soil_col = find_column(
    soil,
    [
        "species_normalized",
        "species_full",
        "species",
    ]
)


survival["species_key"] = (
    survival[survival_col]
    .apply(normalize_species)
)

candidates["species_key"] = (
    candidates[candidate_col]
    .apply(normalize_species)
)

v5["species_key"] = (
    v5[v5_col]
    .apply(normalize_species)
)

evidence["species_key"] = (
    evidence[evidence_col]
    .apply(normalize_species)
)

soil["species_key"] = (
    soil[soil_col]
    .apply(normalize_species)
)


# ============================================================
# V5
# ============================================================

v5_prediction_col = find_column(
    v5,
    [
        "v5_predicted_survival",
        "v5_prediction",
        "predicted_survival",
        "prediction",
        "survival_prediction",
        "v5_score",
    ]
)

if v5_prediction_col is None:

    numeric = [
        col
        for col in v5.columns
        if (
            col != "species_key"
            and pd.api.types.is_numeric_dtype(
                v5[col]
            )
        )
    ]

    if len(numeric) != 1:

        raise ValueError(
            "Cannot identify V5 prediction column."
        )

    v5_prediction_col = numeric[0]


v5_base = v5[
    [
        "species_key",
        v5_prediction_col,
    ]
].rename(
    columns={
        v5_prediction_col:
            "v5_score"
    }
)


# ============================================================
# EVIDENCE
# ============================================================

evidence_score_col = find_column(
    evidence,
    [
        "evidence_score",
        "final_score",
    ]
)

if evidence_score_col is None:

    raise ValueError(
        "Evidence score not found."
    )


evidence_base = evidence[
    [
        "species_key",
        evidence_score_col,
    ]
].rename(
    columns={
        evidence_score_col:
            "evidence_score"
    }
)


# ============================================================
# SOIL
# ============================================================

soil_score_col = find_column(
    soil,
    [
        "soil_compatibility",
        "soil_score",
        "compatibility_score",
        "soil_fit",
    ]
)

if soil_score_col is None:

    raise ValueError(
        "Soil score not found."
    )


soil_base = soil[
    [
        "species_key",
        soil_score_col,
    ]
].rename(
    columns={
        soil_score_col:
            "soil_score"
    }
)


# ============================================================
# MERGE COMPONENTS
# ============================================================

result = candidates.copy()

result = result.merge(
    v5_base,
    on="species_key",
    how="left"
)

result = result.merge(
    evidence_base,
    on="species_key",
    how="left"
)

result = result.merge(
    soil_base,
    on="species_key",
    how="left"
)


# ============================================================
# POLLUTION
# ============================================================

pollution, pollution_severity = (
    build_pollution_scores()
)

result = result.merge(
    pollution[
        [
            "species_key",
            "pollution_response",
            "pollution_score",
            "evidence_level",
            "evidence_metric",
            "evidence_value",
            "evidence_unit",
            "source_note",
        ]
    ],
    on="species_key",
    how="left"
)


# ============================================================
# CLIMATE DATA
# ============================================================

print("\n" + "=" * 70)
print("CLIMATE")
print("=" * 70)


climate_survival = survival[
    survival["duration_months"] > 0
].copy()

climate_survival["lat_dec"] = pd.to_numeric(
    climate_survival["lat_dec"],
    errors="coerce"
)

climate_survival["lon_dec"] = pd.to_numeric(
    climate_survival["lon_dec"],
    errors="coerce"
)

climate_survival = climate_survival.dropna(
    subset=[
        "lat_dec",
        "lon_dec",
        "species_key",
    ]
)


locations = (
    climate_survival[
        [
            "lat_dec",
            "lon_dec",
        ]
    ]
    .drop_duplicates()
)


# ============================================================
# CLIMATE CACHE
# ============================================================

if os.path.exists(
    CLIMATE_CACHE
):

    climate_cache = pd.read_csv(
        CLIMATE_CACHE
    )

else:

    climate_cache = pd.DataFrame()


cached = set()

if len(climate_cache) > 0:

    for _, row in climate_cache.iterrows():

        cached.add(
            (
                round(
                    float(row["lat_dec"]),
                    5
                ),
                round(
                    float(row["lon_dec"]),
                    5
                ),
            )
        )


missing = []

for _, row in locations.iterrows():

    key = (
        round(
            float(row["lat_dec"]),
            5
        ),
        round(
            float(row["lon_dec"]),
            5
        ),
    )

    if key not in cached:

        missing.append(
            (
                float(row["lat_dec"]),
                float(row["lon_dec"]),
            )
        )


print(
    f"Historical locations: "
    f"{len(locations)}"
)

print(
    f"New climate requests: "
    f"{len(missing)}"
)


# ============================================================
# DOWNLOAD MISSING CLIMATE
# ============================================================

if missing:

    new_results = []

    with ThreadPoolExecutor(
        max_workers=8
    ) as executor:

        futures = {
            executor.submit(
                get_climate,
                lat,
                lon,
            )
            for lat, lon in missing
        }

        for future in as_completed(
            futures
        ):

            data = future.result()

            if data is not None:

                new_results.append(
                    data
                )


    if new_results:

        climate_cache = pd.concat(
            [
                climate_cache,
                pd.DataFrame(
                    new_results
                ),
            ],
            ignore_index=True
        )


if len(climate_cache) > 0:

    climate_cache.to_csv(
        CLIMATE_CACHE,
        index=False
    )


# ============================================================
# PUNE CLIMATE
# ============================================================

pune_climate = get_climate(
    PUNE_LAT,
    PUNE_LON
)

if pune_climate is None:

    raise RuntimeError(
        "Could not obtain Pune climate."
    )


# ============================================================
# HISTORICAL SPECIES CLIMATE
# ============================================================

hist = climate_survival.merge(
    climate_cache,
    on=[
        "lat_dec",
        "lon_dec",
    ],
    how="left"
)


climate_scores = []


for species_key in result[
    "species_key"
]:

    species_rows = hist[
        hist["species_key"]
        == species_key
    ]

    score = calculate_climate_score(
        species_rows,
        pune_climate
    )

    climate_scores.append(
        {
            "species_key":
                species_key,

            "climate_score":
                round(score, 2),

            "climate_observations":
                len(species_rows),
        }
    )


climate_df = pd.DataFrame(
    climate_scores
)


result = result.merge(
    climate_df,
    on="species_key",
    how="left"
)


# ============================================================
# USER FIT
# ============================================================

# Prototype default.
#
# These will later come directly from
# website/app inputs.

result["user_fit_score"] = 50.0


# ============================================================
# CLEAN COMPONENTS
# ============================================================

component_columns = [
    "v5_score",
    "evidence_score",
    "soil_score",
    "climate_score",
    "pollution_score",
    "user_fit_score",
]

for col in component_columns:

    result[col] = pd.to_numeric(
        result[col],
        errors="coerce"
    )

    result[col] = (
        result[col]
        .fillna(50.0)
        .clip(0, 100)
    )


# ============================================================
# FINAL SCORE
# ============================================================

result["v6_score"] = (

    WEIGHT_V5
    * result["v5_score"]

    +

    WEIGHT_EVIDENCE
    * result["evidence_score"]

    +

    WEIGHT_SOIL
    * result["soil_score"]

    +

    WEIGHT_CLIMATE
    * result["climate_score"]

    +

    WEIGHT_POLLUTION
    * result["pollution_score"]

    +

    WEIGHT_USER
    * result["user_fit_score"]
)


# ============================================================
# RANK
# ============================================================

result = result.sort_values(
    "v6_score",
    ascending=False
).reset_index(
    drop=True
)

result["rank"] = (
    np.arange(
        len(result)
    ) + 1
)


# ============================================================
# SAVE
# ============================================================

output_columns = [
    "rank",
    candidate_col,

    "v6_score",

    "v5_score",
    "evidence_score",
    "soil_score",
    "climate_score",
    "pollution_score",
    "user_fit_score",

    "pollution_response",
    "evidence_level",
    "evidence_metric",
    "evidence_value",
    "evidence_unit",
    "source_note",

    "climate_observations",
]


for col in [
    "common_name",
    "local_name",
    "pune_tree_count",
    "canonical_botanical_name",
    "dominant_condition",
    "survival_observations",
    "observed_survival_mean",
]:

    if col in result.columns:

        output_columns.append(
            col
        )


output_columns = list(
    dict.fromkeys(
        output_columns
    )
)


result[
    output_columns
].to_csv(
    FINAL_OUTPUT,
    index=False
)


# ============================================================
# DISPLAY
# ============================================================

print("\n" + "=" * 70)
print("FINAL V6 RANKING")
print("=" * 70)


display_columns = [
    "rank",
    candidate_col,
    "v6_score",
    "v5_score",
    "evidence_score",
    "soil_score",
    "climate_score",
    "pollution_score",
]


if "common_name" in result.columns:

    display_columns.insert(
        2,
        "common_name"
    )


print(
    result[
        display_columns
    ].to_string(
        index=False
    )
)


print("\n" + "=" * 70)
print("TOP 5 RECOMMENDATIONS")
print("=" * 70)


for _, row in result.head(5).iterrows():

    print(
        f"\n#{int(row['rank'])} "
        f"{row[candidate_col]}"
    )

    if "common_name" in result.columns:

        print(
            f"Common name: "
            f"{row['common_name']}"
        )

    if "local_name" in result.columns:

        print(
            f"Local name: "
            f"{row['local_name']}"
        )

    print(
        f"Final score: "
        f"{row['v6_score']:.2f}"
    )

    print(
        f"Survival: "
        f"{row['v5_score']:.2f}"
    )

    print(
        f"Evidence: "
        f"{row['evidence_score']:.2f}"
    )

    print(
        f"Soil: "
        f"{row['soil_score']:.2f}"
    )

    print(
        f"Climate: "
        f"{row['climate_score']:.2f}"
    )

    print(
        f"Pollution benefit: "
        f"{row['pollution_score']:.2f}"
    )


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("V6 COMPLETE")
print("=" * 70)

print(
    f"Final ranking: {FINAL_OUTPUT}"
)

print(
    f"Pollution data: {POLLUTION_OUTPUT}"
)

print(
    f"Pollution severity: "
    f"{pollution_severity:.4f}"
)

print(
    "\nWeights:"
)

print(
    f"Survival={WEIGHT_V5:.0%}, "
    f"Evidence={WEIGHT_EVIDENCE:.0%}, "
    f"Soil={WEIGHT_SOIL:.0%}, "
    f"Climate={WEIGHT_CLIMATE:.0%}, "
    f"Pollution={WEIGHT_POLLUTION:.0%}, "
    f"User={WEIGHT_USER:.0%}"
)

print(
    "\nIMPORTANT:"
)

print(
    "V6 is a recommendation prototype, "
    "not a calibrated survival probability."
)

print(
    "Pollution score represents evidence-backed "
    "pollution response/benefit, not measured "
    "kg of pollutant removed per tree."
)