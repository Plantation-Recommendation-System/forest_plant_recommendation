from pathlib import Path
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from scipy.stats import spearmanr


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = Path(
    "models/survival_v5_CORRECT.cbm"
)

TEST_FILE = Path(
    "data/survival_test.csv"
)

OUTPUT_FILE = Path(
    "outputs/v5_ranking_evaluation.csv"
)

PREDICTIONS_FILE = Path(
    "outputs/v5_test_predictions_for_ranking.csv"
)


# ============================================================
# V5 FEATURES
# ============================================================

FEATURES = [
    "country",
    "province",
    "disturbance",
    "forest_condition",
    "comp_removal",
    "shading",
    "soil_prep",
    "water_reg",
    "fertilisation",
    "protection",
    "planting_density",
    "planting_sp_no",
    "age_0",
    "height_0",
    "species_full",
    "genus",
    "species",
    "family",
    "lat_dec",
    "lon_dec",
    "duration_months",
    "w_meanWD",
    "w_sdWD",
    "w_nInd",
    "treatment",
    "lat_squared",
    "lon_squared",
    "lat_lon_interaction",
]


# ============================================================
# CATEGORICAL FEATURES
# ============================================================

CATEGORICAL_FEATURES = [
    "country",
    "province",
    "disturbance",
    "forest_condition",
    "comp_removal",
    "shading",
    "soil_prep",
    "water_reg",
    "fertilisation",
    "protection",
    "species_full",
    "genus",
    "species",
    "family",
    "treatment",
]


# ============================================================
# NUMERIC FEATURES
# ============================================================

NUMERIC_FEATURES = [
    "planting_density",
    "planting_sp_no",
    "age_0",
    "height_0",
    "lat_dec",
    "lon_dec",
    "duration_months",
    "w_meanWD",
    "w_sdWD",
    "w_nInd",
    "lat_squared",
    "lon_squared",
    "lat_lon_interaction",
]


# ============================================================
# EVALUATION WINDOW
# ============================================================

MIN_DURATION = 10
MAX_DURATION = 24

MIN_SPECIES_PER_LOCATION = 3


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    if not MODEL_PATH.exists():

        print()
        print("ERROR: V5 model not found:")
        print(f"  {MODEL_PATH}")
        print()

        sys.exit(1)

    print()
    print("Loading V5 CatBoost model...")

    model = CatBoostRegressor()

    model.load_model(
        str(MODEL_PATH)
    )

    print("Model loaded successfully.")

    return model


# ============================================================
# LOAD TEST DATA
# ============================================================

def load_test_data():

    if not TEST_FILE.exists():

        print()
        print("ERROR: Test CSV not found:")
        print(f"  {TEST_FILE}")
        print()

        sys.exit(1)

    df = pd.read_csv(
        TEST_FILE
    )

    print()
    print(
        f"Test rows loaded: {len(df)}"
    )

    return df


# ============================================================
# CREATE V5 ENGINEERED FEATURES
# ============================================================

def create_engineered_features(df):

    df = df.copy()

    print()
    print(
        "Creating V5 geographic features..."
    )

    # --------------------------------------------------------
    # Convert coordinates to numeric
    # --------------------------------------------------------

    df["lat_dec"] = pd.to_numeric(
        df["lat_dec"],
        errors="coerce"
    )

    df["lon_dec"] = pd.to_numeric(
        df["lon_dec"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # EXACT V5 ENGINEERING
    # --------------------------------------------------------

    df["lat_squared"] = (
        df["lat_dec"] ** 2
    )

    df["lon_squared"] = (
        df["lon_dec"] ** 2
    )

    df["lat_lon_interaction"] = (
        df["lat_dec"]
        *
        df["lon_dec"]
    )

    print(
        "  ✓ lat_squared"
    )

    print(
        "  ✓ lon_squared"
    )

    print(
        "  ✓ lat_lon_interaction"
    )

    return df


# ============================================================
# PREPARE V5 FEATURES
# ============================================================

def prepare_features(df):

    df = create_engineered_features(
        df
    )

    # --------------------------------------------------------
    # Check required columns AFTER engineering
    # --------------------------------------------------------

    missing = [
        column
        for column in FEATURES
        if column not in df.columns
    ]

    if missing:

        print()
        print(
            "ERROR: Missing V5 features:"
        )

        for column in missing:
            print(
                f"  - {column}"
            )

        sys.exit(1)

    # --------------------------------------------------------
    # Numeric
    # --------------------------------------------------------

    for column in NUMERIC_FEATURES:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        if df[column].isna().any():

            median = df[column].median()

            if pd.isna(median):
                median = 0.0

            df[column] = (
                df[column]
                .fillna(median)
            )

    # --------------------------------------------------------
    # Categorical
    # --------------------------------------------------------

    for column in CATEGORICAL_FEATURES:

        df[column] = (
            df[column]
            .fillna("Unknown")
            .astype(str)
        )

    return df


# ============================================================
# CREATE LOCATION ID
# ============================================================

def create_location_id(df):

    df = df.copy()

    df["location_id"] = (
        df["lat_dec"]
        .round(4)
        .astype(str)
        +
        "_"
        +
        df["lon_dec"]
        .round(4)
        .astype(str)
    )

    return df


# ============================================================
# PREDICT
# ============================================================

def generate_predictions(
    model,
    df
):

    print()
    print(
        "Generating V5 predictions..."
    )

    X = df[
        FEATURES
    ]

    predictions = model.predict(
        X
    )

    df[
        "v5_predicted_survival"
    ] = np.clip(
        predictions,
        0,
        100
    )

    print(
        "Predictions generated:"
        f" {len(predictions)}"
    )

    return df


# ============================================================
# AGGREGATE LOCATION + SPECIES
# ============================================================

def aggregate_species(df):

    print()
    print(
        "Aggregating location × species..."
    )

    grouped = (
        df
        .groupby(
            [
                "location_id",
                "species_full",
            ],
            as_index=False
        )
        .agg(
            actual_survival=(
                "survival_per",
                "mean"
            ),

            predicted_survival=(
                "v5_predicted_survival",
                "mean"
            ),

            observations=(
                "survival_per",
                "count"
            ),

            duration_mean=(
                "duration_months",
                "mean"
            ),
        )
    )

    return grouped


# ============================================================
# EVALUATE ONE LOCATION
# ============================================================

def evaluate_location(group):

    if len(group) < MIN_SPECIES_PER_LOCATION:
        return None

    group = group.copy()

    # --------------------------------------------------------
    # Actual ranking
    # --------------------------------------------------------

    group["actual_rank"] = (
        group[
            "actual_survival"
        ]
        .rank(
            method="average",
            ascending=False
        )
    )

    # --------------------------------------------------------
    # Predicted ranking
    # --------------------------------------------------------

    group["predicted_rank"] = (
        group[
            "predicted_survival"
        ]
        .rank(
            method="average",
            ascending=False
        )
    )

    # --------------------------------------------------------
    # Spearman correlation
    # --------------------------------------------------------

    correlation = spearmanr(
        group[
            "predicted_survival"
        ],
        group[
            "actual_survival"
        ]
    ).statistic

    # --------------------------------------------------------
    # Actual best
    # --------------------------------------------------------

    actual_best = (
        group
        .sort_values(
            "actual_survival",
            ascending=False
        )
        .iloc[0]
    )

    actual_best_species = (
        actual_best[
            "species_full"
        ]
    )

    # --------------------------------------------------------
    # Predicted best
    # --------------------------------------------------------

    predicted_best = (
        group
        .sort_values(
            "predicted_survival",
            ascending=False
        )
        .iloc[0]
    )

    predicted_best_species = (
        predicted_best[
            "species_full"
        ]
    )

    # --------------------------------------------------------
    # Top 1
    # --------------------------------------------------------

    top1 = int(
        predicted_best_species
        ==
        actual_best_species
    )

    # --------------------------------------------------------
    # Top 3
    # --------------------------------------------------------

    actual_top3 = set(
        group
        .sort_values(
            "actual_survival",
            ascending=False
        )
        .head(3)[
            "species_full"
        ]
    )

    predicted_top3 = set(
        group
        .sort_values(
            "predicted_survival",
            ascending=False
        )
        .head(3)[
            "species_full"
        ]
    )

    top3 = int(
        len(
            actual_top3
            &
            predicted_top3
        ) > 0
    )

    # --------------------------------------------------------
    # Rank of actual best
    # --------------------------------------------------------

    actual_best_prediction_rank = (
        group[
            group[
                "species_full"
            ]
            ==
            actual_best_species
        ]
        [
            "predicted_rank"
        ]
        .iloc[0]
    )

    return {
        "location_id":
            group[
                "location_id"
            ].iloc[0],

        "n_species":
            len(group),

        "spearman":
            correlation,

        "top1":
            top1,

        "top3":
            top3,

        "actual_best_species":
            actual_best_species,

        "predicted_best_species":
            predicted_best_species,

        "rank_of_actual_best":
            actual_best_prediction_rank,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("V5 SPECIES-RANKING EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    model = load_model()

    df = load_test_data()

    # --------------------------------------------------------
    # Required original columns
    # --------------------------------------------------------

    required = [
        "survival_per",
        "species_full",
        "lat_dec",
        "lon_dec",
        "duration_months",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        print()
        print(
            "ERROR: Missing required columns:"
        )

        for column in missing:
            print(
                f"  - {column}"
            )

        sys.exit(1)

    # --------------------------------------------------------
    # Duration filter
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("APPLYING COMMON DURATION WINDOW")
    print("-" * 70)

    before = len(df)

    df["duration_months"] = pd.to_numeric(
        df["duration_months"],
        errors="coerce"
    )

    df = df[
        (
            df["duration_months"]
            >= MIN_DURATION
        )
        &
        (
            df["duration_months"]
            <= MAX_DURATION
        )
    ].copy()

    print()
    print(
        f"Rows before filtering: {before}"
    )

    print(
        f"Rows after filtering:  {len(df)}"
    )

    print(
        f"Window: "
        f"{MIN_DURATION}–"
        f"{MAX_DURATION} months"
    )

    # --------------------------------------------------------
    # Location IDs
    # --------------------------------------------------------

    df = create_location_id(
        df
    )

    print()
    print(
        "Locations after filtering:",
        df[
            "location_id"
        ].nunique()
    )

    # --------------------------------------------------------
    # Prepare features
    # --------------------------------------------------------

    df = prepare_features(
        df
    )

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    df = generate_predictions(
        model,
        df
    )

    # --------------------------------------------------------
    # Save row-level predictions
    # --------------------------------------------------------

    PREDICTIONS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        PREDICTIONS_FILE,
        index=False
    )

    print()
    print(
        "Saved row-level predictions:"
    )

    print(
        f"  {PREDICTIONS_FILE}"
    )

    # --------------------------------------------------------
    # Aggregate
    # --------------------------------------------------------

    grouped = aggregate_species(
        df
    )

    print()
    print(
        "Location-species combinations:",
        len(grouped)
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("EVALUATING LOCATIONS")
    print("-" * 70)

    results = []

    for location_id, group in (
        grouped.groupby(
            "location_id"
        )
    ):

        result = evaluate_location(
            group
        )

        if result is not None:

            results.append(
                result
            )

    results_df = pd.DataFrame(
        results
    )

    if results_df.empty:

        print()
        print(
            "ERROR: No locations contained "
            f"at least {MIN_SPECIES_PER_LOCATION} "
            "species."
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    results_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    mean_spearman = (
        results_df[
            "spearman"
        ]
        .mean()
    )

    median_spearman = (
        results_df[
            "spearman"
        ]
        .median()
    )

    top1_accuracy = (
        results_df[
            "top1"
        ]
        .mean()
    )

    top3_accuracy = (
        results_df[
            "top3"
        ]
        .mean()
    )

    mean_rank = (
        results_df[
            "rank_of_actual_best"
        ]
        .mean()
    )

    median_rank = (
        results_df[
            "rank_of_actual_best"
        ]
        .median()
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("V5 RANKING RESULTS")
    print("=" * 70)

    print()

    print(
        f"Locations evaluated: "
        f"{len(results_df)}"
    )

    print(
        f"Mean species per location: "
        f"{grouped.groupby('location_id').size().mean():.2f}"
    )

    print()

    print(
        f"Mean Spearman correlation: "
        f"{mean_spearman:.4f}"
    )

    print(
        f"Median Spearman correlation: "
        f"{median_spearman:.4f}"
    )

    print()

    print(
        f"Top-1 accuracy: "
        f"{top1_accuracy * 100:.2f}%"
    )

    print(
        f"Top-3 accuracy: "
        f"{top3_accuracy * 100:.2f}%"
    )

    print()

    print(
        f"Mean rank of actual best species: "
        f"{mean_rank:.2f}"
    )

    print(
        f"Median rank of actual best species: "
        f"{median_rank:.2f}"
    )

    # --------------------------------------------------------
    # Location results
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("LOCATION RESULTS")
    print("-" * 70)

    print()

    display_columns = [
        "location_id",
        "n_species",
        "spearman",
        "top1",
        "top3",
        "actual_best_species",
        "predicted_best_species",
        "rank_of_actual_best",
    ]

    print(
        results_df[
            display_columns
        ]
        .sort_values(
            "location_id"
        )
        .to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Interpretation
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("INTERPRETATION")
    print("-" * 70)

    print()

    if mean_spearman >= 0.50:

        print(
            "V5 shows strong positive ranking "
            "agreement with observed survival."
        )

    elif mean_spearman >= 0.20:

        print(
            "V5 shows moderate positive ranking "
            "agreement with observed survival."
        )

    elif mean_spearman > 0:

        print(
            "V5 shows weak positive ranking "
            "agreement with observed survival."
        )

    else:

        print(
            "V5 does not show useful positive "
            "ranking agreement in this evaluation."
        )

    print()

    print(
        "Higher Spearman and Top-k accuracy "
        "are better."
    )

    print(
        "Lower rank of the actual best species "
        "is better."
    )

    print()

    print(
        "This evaluates geographically held-out "
        "locations."
    )

    print(
        "It does NOT establish guaranteed field "
        "survival."
    )

    print()

    print(
        "Detailed results:"
    )

    print(
        f"  {OUTPUT_FILE}"
    )

    print(
        "Predictions:"
    )

    print(
        f"  {PREDICTIONS_FILE}"
    )

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()