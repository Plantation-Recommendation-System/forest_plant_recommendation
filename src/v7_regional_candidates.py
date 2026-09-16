"""
V7.5 REGIONAL CANDIDATE ENGINE

Purpose
-------
Build a regional plant candidate pool around any user location.

GBIF is used ONLY as regional occurrence evidence.

Distance bands
--------------
0-10 km   : local evidence
10-25 km  : regional evidence
25-50 km  : extended evidence

Important
---------
This is NOT a suitability model.

GBIF presence != planting suitability.

The output is later combined with:
    - survival model
    - soil
    - climate
    - pollution
    - user/site requirements
    - local municipal inventories

V7.5 design
------------
- No occurrence/count endpoint
- No 23k-record download
- Maximum 5 pages per radius
- Maximum 1,500 records per radius
- Uses 10 / 25 / 50 km bands
- Caches regional records
- Matches survival evidence
- Matches local candidate evidence
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import pandas as pd
import requests


# ============================================================
# LOCATION
# ============================================================

# For now Pune.
# Later these will come from the user.

LATITUDE = 18.5204
LONGITUDE = 73.8567


# ============================================================
# REGIONAL SEARCH CONFIG
# ============================================================

RADII_KM = [
    10,
    25,
    50,
]

PAGE_SIZE = 300

MAX_PAGES_PER_RADIUS = 5

REQUEST_DELAY = 0.2

REQUEST_TIMEOUT = 30


# ============================================================
# FILES
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"

OUTPUT_DIR = PROJECT_ROOT / "outputs"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SURVIVAL_FILE = (
    DATA_DIR
    / "survival_ml_cleaned.csv"
)

LOCAL_CANDIDATE_FILE = (
    DATA_DIR
    / "pune_final_candidate_pool.csv"
)

RAW_CACHE_FILE = (
    OUTPUT_DIR
    / "v7_gbif_bounded_records.csv"
)

DISCOVERY_FILE = (
    OUTPUT_DIR
    / "v7_gbif_discovered_species.csv"
)

FINAL_FILE = (
    OUTPUT_DIR
    / "v7_regional_candidates.csv"
)


# ============================================================
# GBIF
# ============================================================

GBIF_SEARCH_URL = (
    "https://api.gbif.org/v1/occurrence/search"
)


SESSION = requests.Session()

SESSION.headers.update(
    {
        "User-Agent":
            "PlantSurvivalAI/7.5 "
            "(regional candidate prototype)"
    }
)


# ============================================================
# HTTP
# ============================================================

def request_json(
    url,
    params,
    retries=3,
):

    for attempt in range(
        1,
        retries + 1,
    ):

        try:

            response = SESSION.get(
                url,
                params=params,
                timeout=REQUEST_TIMEOUT,
            )

            if response.status_code == 429:

                wait = (
                    3 * attempt
                )

                print(
                    f"Rate limited. "
                    f"Waiting {wait}s..."
                )

                time.sleep(wait)

                continue

            response.raise_for_status()

            return response.json()

        except requests.RequestException as exc:

            print(
                f"Request failed "
                f"({attempt}/{retries}): "
                f"{exc}"
            )

            if attempt < retries:

                time.sleep(
                    2 * attempt
                )

            else:

                raise

    raise RuntimeError(
        "GBIF request failed."
    )


# ============================================================
# SPECIES NAME
# ============================================================

def normalize_species_name(
    value,
):

    if pd.isna(value):

        return None

    text = (
        str(value)
        .strip()
        .lower()
    )

    if not text:

        return None

    tokens = (
        text
        .replace(",", " ")
        .split()
    )

    if len(tokens) < 2:

        return None

    return (
        tokens[0]
        + " "
        + tokens[1]
    )


# ============================================================
# DISTANCE
# ============================================================

def haversine_km(
    lat1,
    lon1,
    lat2,
    lon2,
):

    if any(
        pd.isna(x)
        for x in [
            lat1,
            lon1,
            lat2,
            lon2,
        ]
    ):

        return None

    R = 6371.0088

    p1 = math.radians(
        float(lat1)
    )

    p2 = math.radians(
        float(lat2)
    )

    dlat = math.radians(
        float(lat2) - float(lat1)
    )

    dlon = math.radians(
        float(lon2) - float(lon1)
    )

    a = (
        math.sin(dlat / 2) ** 2
        +
        math.cos(p1)
        * math.cos(p2)
        * math.sin(dlon / 2) ** 2
    )

    return (
        2
        * R
        * math.asin(
            math.sqrt(a)
        )
    )


# ============================================================
# GBIF SEARCH
# ============================================================

def fetch_radius(
    radius_km,
):

    print()
    print(
        "-" * 60
    )

    print(
        f"GBIF radius: {radius_km} km"
    )

    print(
        f"Maximum pages: "
        f"{MAX_PAGES_PER_RADIUS}"
    )

    records = []

    for page in range(
        MAX_PAGES_PER_RADIUS
    ):

        offset = (
            page
            * PAGE_SIZE
        )

        params = {

            "geoDistance":
                f"{LATITUDE},"
                f"{LONGITUDE},"
                f"{radius_km}km",

            "kingdomKey": 6,

            "taxonRank":
                "SPECIES",

            "hasCoordinate":
                "true",

            "occurrenceStatus":
                "PRESENT",

            "hasGeospatialIssue":
                "false",

            "limit":
                PAGE_SIZE,

            "offset":
                offset,
        }

        try:

            data = request_json(
                GBIF_SEARCH_URL,
                params,
            )

        except Exception as exc:

            print(
                f"Stopping {radius_km} km "
                f"search because GBIF "
                f"request failed: {exc}"
            )

            break

        results = data.get(
            "results",
            [],
        )

        total = data.get(
            "count",
            0,
        )

        records.extend(
            results
        )

        print(
            f"page {page + 1}/"
            f"{MAX_PAGES_PER_RADIUS}"
            f" | received "
            f"{len(results)}"
            f" | collected "
            f"{len(records)}"
            f" | GBIF total "
            f"{total:,}"
        )

        if len(results) < PAGE_SIZE:

            print(
                "Final page reached."
            )

            break

        time.sleep(
            REQUEST_DELAY
        )

    return records


# ============================================================
# EXTRACT RECORDS
# ============================================================

def convert_records(
    records,
):

    rows = []

    for record in records:

        species = (
            record.get("species")
            or record.get(
                "scientificName"
            )
        )

        normalized = (
            normalize_species_name(
                species
            )
        )

        if normalized is None:

            continue

        lat = record.get(
            "decimalLatitude"
        )

        lon = record.get(
            "decimalLongitude"
        )

        distance = haversine_km(
            LATITUDE,
            LONGITUDE,
            lat,
            lon,
        )

        rows.append(
            {
                "species_normalized":
                    normalized,

                "scientific_name_gbif":
                    species,

                "species_key":
                    record.get(
                        "speciesKey"
                    ),

                "accepted_taxon_key":
                    record.get(
                        "acceptedTaxonKey"
                    ),

                "latitude":
                    lat,

                "longitude":
                    lon,

                "distance_km":
                    distance,

                "dataset_key":
                    record.get(
                        "datasetKey"
                    ),

                "dataset_name":
                    record.get(
                        "datasetName"
                    ),

                "publisher":
                    record.get(
                        "publishingOrgKey"
                    ),

                "gbif_id":
                    record.get(
                        "key"
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# SURVIVAL
# ============================================================

def load_survival():

    if not SURVIVAL_FILE.exists():

        return pd.DataFrame(
            columns=[
                "species_normalized",
                "survival_n",
                "survival_mean",
                "survival_median",
            ]
        )

    df = pd.read_csv(
        SURVIVAL_FILE,
        low_memory=False,
    )

    df[
        "species_normalized"
    ] = (
        df["species_full"]
        .apply(
            normalize_species_name
        )
    )

    return (
        df
        .dropna(
            subset=[
                "species_normalized",
                "survival_per",
            ]
        )
        .groupby(
            "species_normalized"
        )
        .agg(
            survival_n=(
                "survival_per",
                "count",
            ),

            survival_mean=(
                "survival_per",
                "mean",
            ),

            survival_median=(
                "survival_per",
                "median",
            ),
        )
        .reset_index()
    )


# ============================================================
# LOCAL CANDIDATES
# ============================================================

def load_local_species():

    if not LOCAL_CANDIDATE_FILE.exists():

        return set()

    df = pd.read_csv(
        LOCAL_CANDIDATE_FILE,
        low_memory=False,
    )

    for column in [
        "species_normalized",
        "species",
        "species_full",
        "botanical_name",
    ]:

        if column in df.columns:

            return set(
                df[column]
                .dropna()
                .apply(
                    normalize_species_name
                )
                .dropna()
            )

    return set()


# ============================================================
# AGGREGATE
# ============================================================

def aggregate_species(
    df,
):

    if df.empty:

        return pd.DataFrame()

    output = []

    for species, group in (
        df.groupby(
            "species_normalized"
        )
    ):

        distances = pd.to_numeric(
            group[
                "distance_km"
            ],
            errors="coerce",
        ).dropna()

        # ----------------------------------------------------
        # Distance bands
        # ----------------------------------------------------

        within_10 = (
            distances <= 10
        ).sum()

        within_25 = (
            distances <= 25
        ).sum()

        within_50 = (
            distances <= 50
        ).sum()

        # ----------------------------------------------------
        # Spatial diversity
        # ----------------------------------------------------

        unique_coordinates = (
            group[
                [
                    "latitude",
                    "longitude",
                ]
            ]
            .dropna()
            .drop_duplicates()
            .shape[0]
        )

        # ----------------------------------------------------
        # Dataset diversity
        # ----------------------------------------------------

        unique_datasets = (
            group[
                "dataset_key"
            ]
            .dropna()
            .nunique()
        )

        # ----------------------------------------------------
        # Publisher diversity
        # ----------------------------------------------------

        unique_publishers = (
            group[
                "publisher"
            ]
            .dropna()
            .nunique()
        )

        # ----------------------------------------------------
        # Name
        # ----------------------------------------------------

        names = (
            group[
                "scientific_name_gbif"
            ]
            .dropna()
            .unique()
        )

        scientific_name = (
            names[0]
            if len(names)
            else species
        )

        # ----------------------------------------------------
        # Taxon key
        # ----------------------------------------------------

        accepted = (
            group[
                "accepted_taxon_key"
            ]
            .dropna()
            .unique()
        )

        species_keys = (
            group[
                "species_key"
            ]
            .dropna()
            .unique()
        )

        if len(accepted):

            taxon_key = accepted[0]

        elif len(species_keys):

            taxon_key = species_keys[0]

        else:

            taxon_key = None

        # ----------------------------------------------------
        # Evidence
        # ----------------------------------------------------

        count = len(group)

        if count >= 20:

            evidence = "strong"

        elif count >= 5:

            evidence = "moderate"

        elif count >= 2:

            evidence = "weak"

        else:

            evidence = "very_weak"

        output.append(
            {
                "species_normalized":
                    species,

                "scientific_name_gbif":
                    scientific_name,

                "accepted_taxon_key":
                    taxon_key,

                "gbif_records_sampled":
                    count,

                "unique_coordinates":
                    unique_coordinates,

                "unique_datasets":
                    unique_datasets,

                "unique_publishers":
                    unique_publishers,

                "within_10km":
                    int(within_10),

                "within_25km":
                    int(within_25),

                "within_50km":
                    int(within_50),

                "nearest_distance_km":
                    (
                        float(
                            distances.min()
                        )
                        if len(distances)
                        else None
                    ),

                "median_distance_km":
                    (
                        float(
                            distances.median()
                        )
                        if len(distances)
                        else None
                    ),

                "regional_evidence":
                    evidence,
            }
        )

    return (
        pd.DataFrame(
            output
        )
        .sort_values(
            [
                "within_10km",
                "within_25km",
                "within_50km",
                "unique_coordinates",
            ],
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# CANDIDATE TIER
# ============================================================

def classify_tier(
    row,
):

    survival_n = (
        row["survival_n"]
    )

    local_presence = (
        row["local_presence"]
    )

    regional_support = (
        row["within_25km"] >= 1
        or row["within_10km"] >= 1
    )

    extended_support = (
        row["within_50km"] >= 1
    )

    survival_support = (
        pd.notna(survival_n)
        and survival_n >= 5
    )

    if (
        survival_support
        and regional_support
    ):

        return "A_ML_SUPPORTED"

    if regional_support:

        return "B_REGIONAL_ONLY"

    if (
        extended_support
        and local_presence
    ):

        return "C_LOCAL_EXTENDED"

    return "REJECT"


# ============================================================
# MAIN
# ============================================================

def main():

    start = time.time()

    print()
    print("=" * 70)
    print("V7.5 REGIONAL CANDIDATE ENGINE")
    print("=" * 70)

    print()
    print(
        f"Location: "
        f"{LATITUDE}, {LONGITUDE}"
    )

    print()
    print(
        "Distance bands:"
    )

    print(
        "  0-10 km  = local"
    )

    print(
        "  10-25 km = regional"
    )

    print(
        "  25-50 km = extended"
    )

    print()
    print(
        "Maximum GBIF retrieval:"
    )

    print(
        f"  {len(RADII_KM)} radii"
    )

    print(
        f"  {MAX_PAGES_PER_RADIUS} pages/radius"
    )

    print(
        f"  {PAGE_SIZE} records/page"
    )

    # ========================================================
    # FETCH
    # ========================================================

    all_records = []

    for radius in RADII_KM:

        radius_records = (
            fetch_radius(
                radius
            )
        )

        for record in radius_records:

            record[
                "_query_radius_km"
            ] = radius

        all_records.extend(
            radius_records
        )

    print()
    print(
        f"Total raw records collected: "
        f"{len(all_records):,}"
    )

    # ========================================================
    # CONVERT
    # ========================================================

    df = convert_records(
        all_records
    )

    if df.empty:

        raise RuntimeError(
            "No usable GBIF records."
        )

    # ========================================================
    # CACHE
    # ========================================================

    df.to_csv(
        RAW_CACHE_FILE,
        index=False,
    )

    print()
    print(
        "Saved GBIF cache:"
    )

    print(
        f"  {RAW_CACHE_FILE}"
    )

    # ========================================================
    # AGGREGATE
    # ========================================================

    species = (
        aggregate_species(
            df
        )
    )

    print()
    print(
        f"Regional species discovered: "
        f"{len(species)}"
    )

    # ========================================================
    # SURVIVAL
    # ========================================================

    survival = (
        load_survival()
    )

    print(
        f"Species with survival evidence: "
        f"{len(survival)}"
    )

    # ========================================================
    # LOCAL
    # ========================================================

    local_species = (
        load_local_species()
    )

    print(
        f"Local candidate species: "
        f"{len(local_species)}"
    )

    # ========================================================
    # MERGE
    # ========================================================

    result = species.merge(
        survival,
        on="species_normalized",
        how="left",
    )

    result[
        "local_presence"
    ] = (
        result[
            "species_normalized"
        ]
        .isin(local_species)
    )

    # ========================================================
    # TIERS
    # ========================================================

    result[
        "candidate_tier"
    ] = result.apply(
        classify_tier,
        axis=1,
    )

    # ========================================================
    # REMOVE REJECT
    # ========================================================

    result = result[
        result[
            "candidate_tier"
        ] != "REJECT"
    ].copy()

    # ========================================================
    # SORT
    # ========================================================

    tier_order = {
        "A_ML_SUPPORTED": 1,
        "B_REGIONAL_ONLY": 2,
        "C_LOCAL_EXTENDED": 3,
    }

    result[
        "_tier"
    ] = result[
        "candidate_tier"
    ].map(
        tier_order
    )

    result = (
        result
        .sort_values(
            [
                "_tier",
                "within_10km",
                "within_25km",
                "survival_n",
                "survival_mean",
            ],
            ascending=[
                True,
                False,
                False,
                False,
                False,
            ],
            na_position="last",
        )
        .drop(
            columns=[
                "_tier"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # ========================================================
    # LOCATION METADATA
    # ========================================================

    result[
        "query_latitude"
    ] = LATITUDE

    result[
        "query_longitude"
    ] = LONGITUDE

    # ========================================================
    # SAVE
    # ========================================================

    species.to_csv(
        DISCOVERY_FILE,
        index=False,
    )

    result.to_csv(
        FINAL_FILE,
        index=False,
    )

    # ========================================================
    # REPORT
    # ========================================================

    elapsed = (
        time.time()
        - start
    )

    print()
    print("=" * 70)
    print("V7.5 COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Raw GBIF records: "
        f"{len(df):,}"
    )

    print(
        f"Regional species: "
        f"{len(species)}"
    )

    print(
        f"Final candidates: "
        f"{len(result)}"
    )

    print()

    print(
        "Candidate tiers:"
    )

    print(
        result[
            "candidate_tier"
        ]
        .value_counts()
    )

    print()

    if not result.empty:

        print(
            "Top candidates:"
        )

        columns = [
            "species_normalized",
            "within_10km",
            "within_25km",
            "within_50km",
            "unique_coordinates",
            "survival_n",
            "survival_mean",
            "candidate_tier",
        ]

        print(
            result[
                columns
            ]
            .head(30)
            .to_string(
                index=False
            )
        )

    print()
    print(
        "Files:"
    )

    print(
        f"  {RAW_CACHE_FILE}"
    )

    print(
        f"  {DISCOVERY_FILE}"
    )

    print(
        f"  {FINAL_FILE}"
    )

    print()
    print(
        f"Runtime: "
        f"{elapsed:.1f} seconds"
    )

    print()
    print(
        "GBIF presence is regional evidence, "
        "not a survival prediction."
    )


if __name__ == "__main__":
    main()