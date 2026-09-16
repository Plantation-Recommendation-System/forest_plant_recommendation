import pandas as pd
import numpy as np
from pathlib import Path
from catboost import CatBoostRegressor


# ============================================================
# PROJECT PATHS
# ============================================================

BASE = Path(__file__).resolve().parents[1]

SURVIVAL_FILE = BASE / "data" / "survival_ml_cleaned.csv"
CANDIDATE_FILE = BASE / "data" / "pune_final_candidate_pool.csv"
MODEL_FILE = BASE / "models" / "survival_v5_CORRECT.cbm"

OUTPUT_FILE = BASE / "outputs" / "candidate_v5_predictions.csv"


# ============================================================
# V5 FEATURES
# EXACT ORDER OF LOCKED MODEL
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
# NUMERICAL FEATURES
# ============================================================

NUMERICAL_FEATURES = [
    feature
    for feature in FEATURES
    if feature not in CATEGORICAL_FEATURES
]


# ============================================================
# ENGINEERED FEATURES
# ============================================================

ENGINEERED_FEATURES = [
    "lat_squared",
    "lon_squared",
    "lat_lon_interaction",
]


# ============================================================
# SPECIES NORMALIZATION
# ============================================================

def normalize_species_name(name):

    if pd.isna(name):
        return ""

    name = str(name).strip().lower()

    tokens = name.split()

    if len(tokens) < 2:
        return ""

    return f"{tokens[0]} {tokens[1]}"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("V5 CANDIDATE SPECIES PREDICTION")
    print("=" * 70)

    # --------------------------------------------------------
    # File checks
    # --------------------------------------------------------

    print("\nChecking files...")

    for file in [
        SURVIVAL_FILE,
        CANDIDATE_FILE,
        MODEL_FILE,
    ]:

        if not file.exists():
            raise FileNotFoundError(
                f"Required file not found:\n{file}"
            )

        print(f"OK: {file}")

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print("\nLoading datasets...")

    survival = pd.read_csv(SURVIVAL_FILE)
    candidates = pd.read_csv(CANDIDATE_FILE)

    print(f"Survival dataset: {survival.shape}")
    print(f"Pune candidates: {candidates.shape}")

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\nLoading V5 model...")

    model = CatBoostRegressor()
    model.load_model(MODEL_FILE)

    print(f"Model trees: {model.tree_count_}")
    print(f"Model features: {len(model.feature_names_)}")

    # --------------------------------------------------------
    # Verify model
    # --------------------------------------------------------

    if model.tree_count_ != 1200:
        raise ValueError(
            f"Expected 1200 trees, "
            f"got {model.tree_count_}"
        )

    if len(model.feature_names_) != 28:
        raise ValueError(
            f"Expected 28 features, "
            f"got {len(model.feature_names_)}"
        )

    if model.feature_names_ != FEATURES:

        print("\nActual model features:")
        print(model.feature_names_)

        print("\nExpected features:")
        print(FEATURES)

        raise ValueError(
            "Model feature order mismatch."
        )

    print("V5 model verification: PASSED")

    # --------------------------------------------------------
    # Normalize species names
    # --------------------------------------------------------

    print("\nPreparing species matching...")

    survival["species_normalized"] = (
        survival["species_full"]
        .apply(normalize_species_name)
    )

    candidates["species_normalized"] = (
        candidates["species_normalized"]
        .apply(normalize_species_name)
    )

    # --------------------------------------------------------
    # Match candidates
    # --------------------------------------------------------

    matched_count = 0

    for _, candidate in candidates.iterrows():

        species_name = candidate["species_normalized"]

        matched = survival[
            survival["species_normalized"] == species_name
        ]

        if matched.empty:

            print(
                f"WARNING: No survival data for "
                f"{species_name}"
            )

        else:

            matched_count += 1

    print(
        f"\nMatched candidates: "
        f"{matched_count} / {len(candidates)}"
    )

    if matched_count == 0:

        raise ValueError(
            "No candidate species matched "
            "the survival dataset."
        )

    # --------------------------------------------------------
    # Pune reference location
    # --------------------------------------------------------

    pune_lat = 18.5204
    pune_lon = 73.8567

    print("\nPrediction location:")
    print(f"Latitude:  {pune_lat}")
    print(f"Longitude: {pune_lon}")

    # --------------------------------------------------------
    # Standardized prediction context
    # --------------------------------------------------------

    context = {

        # Categorical
        "country": "India",
        "province": "Maharashtra",

        "disturbance": "Unknown",
        "forest_condition": "Unknown",
        "comp_removal": "Unknown",
        "shading": "Unknown",
        "soil_prep": "Unknown",
        "water_reg": "Unknown",
        "fertilisation": "Unknown",
        "protection": "Unknown",

        "planting_sp_no": "Unknown",

        "treatment": "Unknown",

        # Numeric
        "planting_density": np.nan,

        "age_0": np.nan,

        "height_0": np.nan,

        "lat_dec": pune_lat,

        "lon_dec": pune_lon,

        "duration_months": 12,

        "w_meanWD": np.nan,

        "w_sdWD": np.nan,

        "w_nInd": np.nan,

        # Engineered numeric
        "lat_squared": pune_lat ** 2,

        "lon_squared": pune_lon ** 2,

        "lat_lon_interaction": (
            pune_lat * pune_lon
        ),
    }

    # --------------------------------------------------------
    # Build prediction rows
    # --------------------------------------------------------

    print("\nBuilding prediction rows...")

    prediction_rows = []

    for _, candidate in candidates.iterrows():

        species_name = candidate["species_normalized"]

        matched = survival[
            survival["species_normalized"] == species_name
        ]

        if matched.empty:
            continue

        species_row = matched.iloc[0]

        row = context.copy()

        row["species_full"] = species_row["species_full"]

        row["genus"] = species_row["genus"]

        row["species"] = species_row["species"]

        row["family"] = species_row["family"]

        prediction_rows.append(row)

    prediction_df = pd.DataFrame(
        prediction_rows
    )

    prediction_df["species_normalized"] = (
        prediction_df["species_full"]
        .apply(normalize_species_name)
    )

    print(
        f"Prediction rows created: "
        f"{len(prediction_df)}"
    )

    # --------------------------------------------------------
    # Numerical missing values
    # --------------------------------------------------------

    print("\nFilling numerical missing values...")

    for feature in NUMERICAL_FEATURES:

        if feature not in prediction_df.columns:

            raise ValueError(
                f"Missing numerical feature: "
                f"{feature}"
            )

        # Engineered features are already calculated.
        if feature in ENGINEERED_FEATURES:
            continue

        # Calculate training-data median.
        median_value = survival[feature].median()

        if pd.isna(median_value):

            raise ValueError(
                f"Cannot calculate median for "
                f"numeric feature: {feature}"
            )

        prediction_df[feature] = (
            pd.to_numeric(
                prediction_df[feature],
                errors="coerce"
            )
        )

        prediction_df[feature] = (
            prediction_df[feature]
            .fillna(median_value)
        )

    # --------------------------------------------------------
    # Categorical missing values
    # --------------------------------------------------------

    print("Filling categorical missing values...")

    for feature in CATEGORICAL_FEATURES:

        if feature not in prediction_df.columns:

            raise ValueError(
                f"Missing categorical feature: "
                f"{feature}"
            )

        prediction_df[feature] = (
            prediction_df[feature]
            .fillna("Unknown")
            .astype(str)
        )

    # --------------------------------------------------------
    # Final numeric type enforcement
    # --------------------------------------------------------

    for feature in NUMERICAL_FEATURES:

        prediction_df[feature] = pd.to_numeric(
            prediction_df[feature],
            errors="coerce"
        )

        if prediction_df[feature].isna().any():

            raise ValueError(
                f"NaN remains in numeric feature: "
                f"{feature}"
            )

    # --------------------------------------------------------
    # Final categorical type enforcement
    # --------------------------------------------------------

    for feature in CATEGORICAL_FEATURES:

        prediction_df[feature] = (
            prediction_df[feature]
            .astype(str)
        )

    # --------------------------------------------------------
    # Verify all model features
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in FEATURES
        if feature not in prediction_df.columns
    ]

    if missing_features:

        raise ValueError(
            f"Missing model features: "
            f"{missing_features}"
        )

    # --------------------------------------------------------
    # Build model input
    # --------------------------------------------------------

    X = prediction_df[FEATURES].copy()

    # --------------------------------------------------------
    # Display model input types
    # --------------------------------------------------------

    print("\nModel input validation:")

    for feature in FEATURES:

        print(
            f"{feature:<25} "
            f"{str(X[feature].dtype):<10} "
            f"sample={X[feature].iloc[0]}"
        )

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    print("\nRunning V5 predictions...")

    predictions = model.predict(X)

    predictions = np.asarray(
        predictions,
        dtype=float
    )

    predictions = np.clip(
        predictions,
        0,
        100
    )

    prediction_df[
        "v5_predicted_survival"
    ] = predictions

    # --------------------------------------------------------
    # Prepare output
    # --------------------------------------------------------

    candidate_columns = [
        "species_normalized",
        "pune_tree_count",
        "common_name",
        "local_name",
        "dominant_condition",
        "survival_observations",
        "observed_survival_mean",
        "evidence_tier",
    ]

    missing_candidate_columns = [
        column
        for column in candidate_columns
        if column not in candidates.columns
    ]

    if missing_candidate_columns:

        raise ValueError(
            "Candidate file missing columns: "
            f"{missing_candidate_columns}"
        )

    output = candidates[
        candidate_columns
    ].copy()

    # --------------------------------------------------------
    # Merge predictions
    # --------------------------------------------------------

    output = output.merge(
        prediction_df[
            [
                "species_normalized",
                "v5_predicted_survival",
            ]
        ],
        on="species_normalized",
        how="left",
    )

    # --------------------------------------------------------
    # Rank
    # --------------------------------------------------------

    output = output.sort_values(
        "v5_predicted_survival",
        ascending=False
    ).reset_index(drop=True)

    output.insert(
        0,
        "v5_rank",
        range(1, len(output) + 1)
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("V5 PREDICTION RESULTS")
    print("=" * 70)

    display_columns = [
        "v5_rank",
        "species_normalized",
        "v5_predicted_survival",
        "observed_survival_mean",
        "survival_observations",
    ]

    print(
        output[
            display_columns
        ].to_string(index=False)
    )

    print("\n")
    print("=" * 70)
    print("COMPLETE")
    print("=" * 70)

    print(f"\nSaved to:")
    print(OUTPUT_FILE)

    print(
        f"\nPredicted candidates: "
        f"{output['v5_predicted_survival'].notna().sum()}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()