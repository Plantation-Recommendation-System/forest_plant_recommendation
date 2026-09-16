import os
import numpy as np
import pandas as pd

from catboost import CatBoostRanker


# ============================================================
# CONFIG
# ============================================================

DATA_PATH = "data/survival_ml_cleaned.csv"

TRAIN_PATH = "data/survival_train.csv"
VALIDATION_PATH = "data/survival_validation.csv"
TEST_PATH = "data/survival_test.csv"

MODEL_PATH = "models/survival_ranking_v1.cbm"
METRICS_PATH = "outputs/ranking_v1_training_metrics.csv"

RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("LOADING DATA")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

train_locations = pd.read_csv(TRAIN_PATH)
validation_locations = pd.read_csv(VALIDATION_PATH)
test_locations = pd.read_csv(TEST_PATH)

print(f"Full dataset: {df.shape}")
print(f"Train split:  {train_locations.shape}")
print(f"Validation:   {validation_locations.shape}")
print(f"Test split:   {test_locations.shape}")


# ============================================================
# CREATE LOCATION ID
# ============================================================

def add_location_id(data):
    data = data.copy()

    data["location_id"] = (
        data["lat_dec"].round(4).astype(str)
        + "_"
        + data["lon_dec"].round(4).astype(str)
    )

    return data


df = add_location_id(df)
train_locations = add_location_id(train_locations)
validation_locations = add_location_id(validation_locations)
test_locations = add_location_id(test_locations)


# ============================================================
# USE EXACT V5 GEOGRAPHIC SPLIT
# ============================================================

train_location_ids = set(train_locations["location_id"])
validation_location_ids = set(validation_locations["location_id"])
test_location_ids = set(test_locations["location_id"])

train = df[df["location_id"].isin(train_location_ids)].copy()
validation = df[df["location_id"].isin(validation_location_ids)].copy()
test = df[df["location_id"].isin(test_location_ids)].copy()

print("\n" + "=" * 70)
print("V5-COMPATIBLE GEOGRAPHIC SPLIT")
print("=" * 70)

print(f"Train rows:       {len(train)}")
print(f"Validation rows:  {len(validation)}")
print(f"Test rows:        {len(test)}")

print(f"Train locations:       {train['location_id'].nunique()}")
print(f"Validation locations:  {validation['location_id'].nunique()}")
print(f"Test locations:         {test['location_id'].nunique()}")


# ============================================================
# FEATURE ENGINEERING
# ============================================================

for data in [train, validation, test]:

    data["lat_squared"] = data["lat_dec"] ** 2
    data["lon_squared"] = data["lon_dec"] ** 2
    data["lat_lon_interaction"] = (
        data["lat_dec"] * data["lon_dec"]
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

TARGET = "survival_per"


# IMPORTANT:
# planting_sp_no is NUMERIC, not categorical.

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
# MISSING VALUE HANDLING
# ============================================================

# IMPORTANT:
# Calculate medians from TRAIN ONLY.
# This avoids test/validation information leakage.

train_medians = {}

for col in NUMERIC_FEATURES:
    train[col] = pd.to_numeric(train[col], errors="coerce")

    median_value = train[col].median()

    train_medians[col] = median_value

    train[col] = train[col].fillna(median_value)

    validation[col] = pd.to_numeric(
        validation[col],
        errors="coerce"
    ).fillna(median_value)

    test[col] = pd.to_numeric(
        test[col],
        errors="coerce"
    ).fillna(median_value)


for col in CATEGORICAL_FEATURES:

    train[col] = (
        train[col]
        .fillna("Unknown")
        .astype(str)
    )

    validation[col] = (
        validation[col]
        .fillna("Unknown")
        .astype(str)
    )

    test[col] = (
        test[col]
        .fillna("Unknown")
        .astype(str)
    )


# ============================================================
# REMOVE GROUPS WITH LESS THAN 2 OBSERVATIONS
# ============================================================

def keep_valid_ranking_groups(data):

    counts = data.groupby("location_id").size()

    valid_ids = counts[counts >= 2].index

    return data[
        data["location_id"].isin(valid_ids)
    ].copy()


train = keep_valid_ranking_groups(train)
validation = keep_valid_ranking_groups(validation)
test = keep_valid_ranking_groups(test)


# ============================================================
# SORT BY GROUP
# ============================================================

train = train.sort_values(
    ["location_id", TARGET],
    ascending=[True, False]
).reset_index(drop=True)

validation = validation.sort_values(
    ["location_id", TARGET],
    ascending=[True, False]
).reset_index(drop=True)

test = test.sort_values(
    ["location_id", TARGET],
    ascending=[True, False]
).reset_index(drop=True)


# ============================================================
# GROUP IDs
# ============================================================

# CRITICAL:
# CatBoost requires group_id to be a regular array.
# pandas StringArray causes:
#
# Invalid group_id type=<class 'pandas.arrays.StringArray'>
#
# Therefore explicitly convert to NumPy object arrays.

group_id_train = train["location_id"].astype(str).to_numpy()

group_id_validation = validation["location_id"].astype(str).to_numpy()

group_id_test = test["location_id"].astype(str).to_numpy()


# ============================================================
# VERIFY GROUP STRUCTURE
# ============================================================

print("\n" + "=" * 70)
print("RANKING GROUP STRUCTURE")
print("=" * 70)

print(
    f"Train ranking groups: "
    f"{train['location_id'].nunique()}"
)

print(
    f"Validation ranking groups: "
    f"{validation['location_id'].nunique()}"
)

print(
    f"Test ranking groups: "
    f"{test['location_id'].nunique()}"
)

print(
    f"Mean train observations/group: "
    f"{train.groupby('location_id').size().mean():.2f}"
)

print(
    f"Mean test observations/group: "
    f"{test.groupby('location_id').size().mean():.2f}"
)


# ============================================================
# PREPARE DATA
# ============================================================

X_train = train[FEATURES]
y_train = train[TARGET]

X_validation = validation[FEATURES]
y_validation = validation[TARGET]

X_test = test[FEATURES]
y_test = test[TARGET]


# ============================================================
# TRAIN
# ============================================================

from catboost import Pool

print("\n" + "=" * 70)
print("BUILDING CATBOOST RANKING POOLS")
print("=" * 70)


# ------------------------------------------------------------
# Create explicit CatBoost Pools
# ------------------------------------------------------------

train_pool = Pool(
    data=X_train,
    label=y_train,
    cat_features=CATEGORICAL_FEATURES,
    group_id=group_id_train,
)

validation_pool = Pool(
    data=X_validation,
    label=y_validation,
    cat_features=CATEGORICAL_FEATURES,
    group_id=group_id_validation,
)


print("Train Pool created successfully.")
print("Validation Pool created successfully.")


# ============================================================
# TRAIN RANKING MODEL
# ============================================================

print("\n" + "=" * 70)
print("TRAINING CATBOOST RANKER")
print("=" * 70)


model = CatBoostRanker(
    loss_function="YetiRankPairwise",
    eval_metric="NDCG:top=3",

    iterations=1200,
    depth=7,
    learning_rate=0.035,

    random_seed=RANDOM_STATE,

    verbose=100,

    allow_writing_files=False,
)


model.fit(
    train_pool,
    eval_set=validation_pool,
    use_best_model=True,
)


# ============================================================
# SAVE MODEL
# ============================================================

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

model.save_model(MODEL_PATH)


print("\n" + "=" * 70)
print("MODEL SAVED")
print("=" * 70)

print(MODEL_PATH)


# ============================================================
# SAVE TRAINING INFORMATION
# ============================================================

metrics = pd.DataFrame([
    {
        "model": "Ranking V1",
        "loss_function": "YetiRankPairwise",
        "iterations": model.tree_count_,
        "depth": 7,
        "learning_rate": 0.035,

        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),

        "train_locations":
            train["location_id"].nunique(),

        "validation_locations":
            validation["location_id"].nunique(),

        "test_locations":
            test["location_id"].nunique(),
    }
])


metrics.to_csv(
    METRICS_PATH,
    index=False
)


print("\nTraining information saved:")
print(METRICS_PATH)


# ============================================================
# FINAL CHECK
# ============================================================

print("\n" + "=" * 70)
print("RANKING MODEL TRAINING COMPLETE")
print("=" * 70)

print(f"Trees trained: {model.tree_count_}")
print(f"Train groups:  {train['location_id'].nunique()}")
print(f"Validation groups: {validation['location_id'].nunique()}")
print(f"Test groups:   {test['location_id'].nunique()}")

print("\nModel:")
print(MODEL_PATH)

print("\nNext step:")
print("Run the ranking evaluation script.")