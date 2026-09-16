import os
import numpy as np
import pandas as pd

from catboost import CatBoostRanker
from scipy.stats import spearmanr


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "models/survival_ranking_v1.cbm"
TEST_PATH = "data/survival_test.csv"

PREDICTIONS_PATH = "outputs/ranking_v1_test_predictions.csv"
RESULTS_PATH = "outputs/ranking_v1_evaluation.csv"

MIN_DURATION = 10
MAX_DURATION = 24


# ============================================================
# LOAD MODEL + TEST DATA
# ============================================================

print("=" * 70)
print("LOADING RANKING MODEL")
print("=" * 70)

model = CatBoostRanker()
model.load_model(MODEL_PATH)

test = pd.read_csv(TEST_PATH)

print(f"Test dataset: {test.shape}")


# ============================================================
# LOCATION ID
# ============================================================

test["location_id"] = (
    test["lat_dec"].round(4).astype(str)
    + "_"
    + test["lon_dec"].round(4).astype(str)
)


# ============================================================
# FILTER COMMON EVALUATION HORIZON
# ============================================================

print("\n" + "=" * 70)
print("FILTERING EVALUATION HORIZON")
print("=" * 70)

test = test[
    (test["duration_months"] >= MIN_DURATION)
    & (test["duration_months"] <= MAX_DURATION)
].copy()

print(
    f"Duration range: "
    f"{MIN_DURATION}-{MAX_DURATION} months"
)

print(f"Rows after filter: {len(test)}")
print(f"Locations after filter: {test['location_id'].nunique()}")


# ============================================================
# FEATURE ENGINEERING
# ============================================================

test["lat_squared"] = test["lat_dec"] ** 2
test["lon_squared"] = test["lon_dec"] ** 2
test["lat_lon_interaction"] = (
    test["lat_dec"] * test["lon_dec"]
)


# ============================================================
# FEATURES
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


NUMERIC_FEATURES = [
    c for c in FEATURES
    if c not in CATEGORICAL_FEATURES
]


# ============================================================
# MISSING VALUES
# ============================================================

for col in CATEGORICAL_FEATURES:
    test[col] = (
        test[col]
        .fillna("Unknown")
        .astype(str)
    )

for col in NUMERIC_FEATURES:
    test[col] = pd.to_numeric(
        test[col],
        errors="coerce"
    )

    test[col] = test[col].fillna(
        test[col].median()
    )


# ============================================================
# PREDICT
# ============================================================

print("\n" + "=" * 70)
print("GENERATING RANKING SCORES")
print("=" * 70)

X_test = test[FEATURES]

test["ranking_score"] = model.predict(X_test)


# ============================================================
# SAVE ROW-LEVEL PREDICTIONS
# ============================================================

os.makedirs("outputs", exist_ok=True)

test[
    [
        "location_id",
        "lat_dec",
        "lon_dec",
        "species_full",
        "survival_per",
        "duration_months",
        "ranking_score",
    ]
].to_csv(
    PREDICTIONS_PATH,
    index=False
)

print("Predictions saved:")
print(PREDICTIONS_PATH)


# ============================================================
# SPECIES AGGREGATION
# ============================================================

print("\n" + "=" * 70)
print("AGGREGATING SPECIES")
print("=" * 70)

location_species = (
    test
    .groupby(
        ["location_id", "species_full"],
        as_index=False
    )
    .agg(
        actual_survival=("survival_per", "mean"),
        predicted_score=("ranking_score", "mean"),
        observations=("survival_per", "size"),
    )
)


# ============================================================
# EVALUATE EACH LOCATION
# ============================================================

print("\n" + "=" * 70)
print("EVALUATING LOCATIONS")
print("=" * 70)

results = []

for location_id, group in location_species.groupby(
    "location_id"
):

    # Need at least 3 species for meaningful ranking evaluation.
    if len(group) < 3:
        continue

    group = group.copy()

    # Actual ranking:
    # highest observed survival = rank 1
    group["actual_rank"] = (
        group["actual_survival"]
        .rank(
            ascending=False,
            method="min"
        )
    )

    # Predicted ranking:
    # highest model score = rank 1
    group["predicted_rank"] = (
        group["predicted_score"]
        .rank(
            ascending=False,
            method="min"
        )
    )

    # --------------------------------------------------------
    # Spearman
    # --------------------------------------------------------

    spearman = spearmanr(
        group["actual_survival"],
        group["predicted_score"]
    ).statistic

    if np.isnan(spearman):
        spearman = 0.0

    # --------------------------------------------------------
    # Actual best
    # --------------------------------------------------------

    actual_best_idx = group[
        "actual_survival"
    ].idxmax()

    actual_best_species = group.loc[
        actual_best_idx,
        "species_full"
    ]

    # --------------------------------------------------------
    # Predicted best
    # --------------------------------------------------------

    predicted_best_idx = group[
        "predicted_score"
    ].idxmax()

    predicted_best_species = group.loc[
        predicted_best_idx,
        "species_full"
    ]

    # --------------------------------------------------------
    # Rank of actual best
    # --------------------------------------------------------

    actual_best_predicted_rank = group.loc[
        actual_best_idx,
        "predicted_rank"
    ]

    # --------------------------------------------------------
    # Top-1
    # --------------------------------------------------------

    top1 = int(
        actual_best_species
        == predicted_best_species
    )

    # --------------------------------------------------------
    # Top-3
    # --------------------------------------------------------

    predicted_top3 = set(
        group
        .sort_values(
            "predicted_score",
            ascending=False
        )
        .head(3)["species_full"]
    )

    top3 = int(
        actual_best_species
        in predicted_top3
    )

    results.append(
        {
            "location_id": location_id,
            "n_species": len(group),
            "spearman": spearman,
            "top1": top1,
            "top3": top3,
            "actual_best_species":
                actual_best_species,
            "predicted_best_species":
                predicted_best_species,
            "rank_of_actual_best":
                actual_best_predicted_rank,
        }
    )


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(results)


# ============================================================
# SAVE RESULTS
# ============================================================

results_df.to_csv(
    RESULTS_PATH,
    index=False
)


# ============================================================
# SUMMARY METRICS
# ============================================================

print("\n" + "=" * 70)
print("RANKING V1 RESULTS")
print("=" * 70)

if len(results_df) == 0:

    print("No locations had at least 3 species.")

else:

    mean_spearman = results_df[
        "spearman"
    ].mean()

    median_spearman = results_df[
        "spearman"
    ].median()

    top1_accuracy = results_df[
        "top1"
    ].mean()

    top3_accuracy = results_df[
        "top3"
    ].mean()

    mean_best_rank = results_df[
        "rank_of_actual_best"
    ].mean()

    median_best_rank = results_df[
        "rank_of_actual_best"
    ].median()

    mean_species = results_df[
        "n_species"
    ].mean()

    print(
        f"Locations evaluated: "
        f"{len(results_df)}"
    )

    print(
        f"Mean species per location: "
        f"{mean_species:.2f}"
    )

    print(
        f"\nMean Spearman correlation: "
        f"{mean_spearman:.4f}"
    )

    print(
        f"Median Spearman correlation: "
        f"{median_spearman:.4f}"
    )

    print(
        f"\nTop-1 accuracy: "
        f"{top1_accuracy:.2%}"
    )

    print(
        f"Top-3 accuracy: "
        f"{top3_accuracy:.2%}"
    )

    print(
        f"\nMean rank of actual best species: "
        f"{mean_best_rank:.2f}"
    )

    print(
        f"Median rank of actual best species: "
        f"{median_best_rank:.2f}"
    )


# ============================================================
# LOCATION RESULTS
# ============================================================

print("\n" + "=" * 70)
print("LOCATION RESULTS")
print("=" * 70)

if len(results_df) > 0:
    print(
        results_df.to_string(
            index=False
        )
    )


print("\n" + "=" * 70)
print("FILES SAVED")
print("=" * 70)

print(
    f"Evaluation: "
    f"{RESULTS_PATH}"
)

print(
    f"Predictions: "
    f"{PREDICTIONS_PATH}"
)