from pathlib import Path
import json
import requests
import pandas as pd


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CAPABILITY_FILE = PROJECT_ROOT / "data" / "pollution_capability.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------
# Open-Meteo Air Quality API
# ---------------------------------------------------------

AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"


# ---------------------------------------------------------
# Pollution data
# ---------------------------------------------------------

def get_air_quality(latitude: float, longitude: float):
    """
    Get current air-quality measurements for a location.

    Returns PM2.5, PM10, NO2, SO2 and O3.
    """

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "pm2_5,pm10,nitrogen_dioxide,sulphur_dioxide,ozone",
        "timezone": "auto",
    }

    response = requests.get(
        AIR_QUALITY_URL,
        params=params,
        timeout=20
    )

    response.raise_for_status()

    data = response.json()

    current = data.get("current", {})

    result = {
        "latitude": latitude,
        "longitude": longitude,
        "time": current.get("time"),
        "pm2_5": current.get("pm2_5"),
        "pm10": current.get("pm10"),
        "no2": current.get("nitrogen_dioxide"),
        "so2": current.get("sulphur_dioxide"),
        "o3": current.get("ozone"),
    }

    return result


# ---------------------------------------------------------
# Convert pollutant concentration into severity
# ---------------------------------------------------------

def normalize(value, low, high):
    """
    Convert a pollutant concentration into 0-1 severity.
    """

    if value is None:
        return None

    value = float(value)

    if value <= low:
        return 0.0

    if value >= high:
        return 1.0

    return (value - low) / (high - low)


def calculate_pollution_severity(air):
    """
    Calculate overall pollution pressure.

    The thresholds are prototype thresholds rather than
    regulatory AQI calculations.

    PM2.5 is given the highest weight because fine
    particulate pollution is particularly important
    for urban air-quality applications.
    """

    components = {}

    components["pm2_5"] = normalize(
        air.get("pm2_5"),
        5,
        75
    )

    components["pm10"] = normalize(
        air.get("pm10"),
        15,
        150
    )

    components["no2"] = normalize(
        air.get("no2"),
        10,
        100
    )

    components["so2"] = normalize(
        air.get("so2"),
        5,
        75
    )

    components["o3"] = normalize(
        air.get("o3"),
        50,
        180
    )

    weights = {
        "pm2_5": 0.35,
        "pm10": 0.25,
        "no2": 0.20,
        "so2": 0.10,
        "o3": 0.10,
    }

    valid = {
        key: value
        for key, value in components.items()
        if value is not None
    }

    if not valid:
        raise ValueError(
            "No pollutant measurements were returned."
        )

    weighted_sum = sum(
        valid[key] * weights[key]
        for key in valid
    )

    weight_sum = sum(
        weights[key]
        for key in valid
    )

    severity = weighted_sum / weight_sum

    return {
        "pollution_severity": round(severity, 4),
        "components": {
            key: None if value is None else round(value, 4)
            for key, value in components.items()
        }
    }


# ---------------------------------------------------------
# Species pollution capability
# ---------------------------------------------------------

def load_capability_table():
    """
    Load species pollution capability table.
    """

    if not CAPABILITY_FILE.exists():
        raise FileNotFoundError(
            f"Missing file: {CAPABILITY_FILE}"
        )

    df = pd.read_csv(CAPABILITY_FILE)

    required = {
        "species_normalized",
        "pollution_capability",
        "evidence_level",
        "pollution_metric",
        "source_note",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns in pollution_capability.csv: {missing}"
        )

    df["species_normalized"] = (
        df["species_normalized"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df["pollution_capability"] = pd.to_numeric(
        df["pollution_capability"],
        errors="coerce"
    )

    return df


# ---------------------------------------------------------
# Pollution benefit
# ---------------------------------------------------------

def calculate_pollution_scores(
    species_df,
    pollution_severity
):
    """
    Calculate pollution benefit:

        pollution severity × species capability

    This means pollution becomes more important when
    pollution at the selected location is high.
    """

    df = species_df.copy()

    df["pollution_capability"] = pd.to_numeric(
        df["pollution_capability"],
        errors="coerce"
    )

    df["pollution_capability"] = (
        df["pollution_capability"]
        .fillna(0)
        .clip(0, 100)
    )

    # 0-1 capability
    df["capability_normalized"] = (
        df["pollution_capability"] / 100.0
    )

    # Severity × capability
    df["pollution_benefit_raw"] = (
        pollution_severity
        * df["capability_normalized"]
    )

    # Convert pollution benefit to a 0-100 score.
    #
    # IMPORTANT:
    # Do NOT normalize across candidate species.
    # The score must retain the effect of actual pollution
    # severity at the selected location.
    #
    # Example:
    #   severity = 0.10, capability = 90
    #       -> benefit = 0.09
    #
    #   severity = 0.80, capability = 90
    #       -> benefit = 0.72
    #
    # Therefore high-pollution locations produce much stronger
    # pollution-related ranking pressure.

    df["pollution_score"] = (
        df["pollution_benefit_raw"] * 100
    ).clip(0, 100)

    df["pollution_score"] = df[
        "pollution_score"
    ].round(2)

    df["pollution_benefit_raw"] = df[
        "pollution_benefit_raw"
    ].round(4)

    return df


# ---------------------------------------------------------
# Complete pollution pipeline
# ---------------------------------------------------------

def run_pollution_pipeline(
    latitude,
    longitude
):
    """
    Complete pollution pipeline:

    location
       ↓
    air-quality API
       ↓
    pollution severity
       ↓
    species capability
       ↓
    pollution benefit
       ↓
    species pollution scores
    """

    print("\nFetching air quality...")

    air = get_air_quality(
        latitude,
        longitude
    )

    severity = calculate_pollution_severity(
        air
    )

    capability = load_capability_table()

    scores = calculate_pollution_scores(
        capability,
        severity["pollution_severity"]
    )

    # Save current pollution information
    pollution_output = {
        "location": {
            "latitude": latitude,
            "longitude": longitude,
        },
        "air_quality": air,
        "pollution_severity": severity[
            "pollution_severity"
        ],
        "severity_components": severity[
            "components"
        ],
    }

    json_path = (
        OUTPUT_DIR /
        "pollution_current.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            pollution_output,
            f,
            indent=2
        )

    # Save species scores
    scores_path = (
        OUTPUT_DIR /
        "pollution_scores.csv"
    )

    scores.to_csv(
        scores_path,
        index=False
    )

    return (
        air,
        severity,
        scores
    )


# ---------------------------------------------------------
# Command-line test
# ---------------------------------------------------------

if __name__ == "__main__":

    # Pune approximate city-center coordinate.
    # This is ONLY a test coordinate.
    LATITUDE = 18.5204
    LONGITUDE = 73.8567

    air, severity, scores = run_pollution_pipeline(
        LATITUDE,
        LONGITUDE
    )

    print("\n" + "=" * 60)
    print("POLLUTION MODULE TEST")
    print("=" * 60)

    print("\nAir quality:")
    for key, value in air.items():
        print(f"{key}: {value}")

    print(
        "\nPollution severity:",
        severity["pollution_severity"]
    )

    print("\nSpecies pollution ranking:")

    display_columns = [
        "species_normalized",
        "pollution_capability",
        "evidence_level",
        "pollution_score",
    ]

    print(
        scores[
            display_columns
        ]
        .sort_values(
            "pollution_score",
            ascending=False
        )
        .to_string(index=False)
    )

    print(
        f"\nSaved: {OUTPUT_DIR / 'pollution_current.json'}"
    )

    print(
        f"Saved: {OUTPUT_DIR / 'pollution_scores.csv'}"
    )