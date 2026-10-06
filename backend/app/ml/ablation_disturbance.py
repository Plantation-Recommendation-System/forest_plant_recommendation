from pathlib import Path

import numpy as np
import pandas as pd

from catboost import CatBoostRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# =========================================================
# PATHS
# =========================================================

BACKEND_DIR = Path(__file__).resolve().parents[2]

TRAIN_PATH = (
    BACKEND_DIR
    / "audit_data"
    / "survival_train_with_climate.csv"
)

TEST_PATH = (
    BACKEND_DIR
    / "audit_data"
    / "survival_test_with_climate.csv"
)

BASELINE_MODEL_PATH = (
    BACKEND_DIR
    / "models"
    / "catboost_survival_final.cbm"
)

FINAL_MODEL_PATH = (
    BACKEND_DIR
    / "models"
    / "catboost_survival_final_28f.cbm"
)

TARGET = "survival_per"


# =========================================================
# FILE CHECK
# =========================================================

print("=" * 75)
print("DISTURBANCE ABLATION TEST")
print("=" * 75)

if not TRAIN_PATH.exists():
    raise FileNotFoundError(
        f"Training data not found: {TRAIN_PATH}"
    )

if not TEST_PATH.exists():
    raise FileNotFoundError(
        f"Test data not found: {TEST_PATH}"
    )

if not BASELINE_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Baseline model not found: {BASELINE_MODEL_PATH}"
    )


# =========================================================
# LOAD DATA
# =========================================================

train_df = pd.read_csv(
    TRAIN_PATH
)

test_df = pd.read_csv(
    TEST_PATH
)

print(
    "\nTraining rows:",
    len(train_df)
)

print(
    "Test rows:",
    len(test_df)
)


if TARGET not in train_df.columns:
    raise ValueError(
        f"Target column '{TARGET}' missing from training data."
    )

if TARGET not in test_df.columns:
    raise ValueError(
        f"Target column '{TARGET}' missing from test data."
    )


# =========================================================
# LOAD EXISTING 29-FEATURE MODEL
# =========================================================

baseline_model = CatBoostRegressor()

baseline_model.load_model(
    str(BASELINE_MODEL_PATH)
)


baseline_features = list(
    baseline_model.feature_names_
)

baseline_cat_indices = (
    baseline_model.get_cat_feature_indices()
)

baseline_cat_features = [
    baseline_features[index]
    for index in baseline_cat_indices
]


print(
    "\nBaseline feature count:",
    len(baseline_features)
)

print(
    "Disturbance in baseline:",
    "disturbance" in baseline_features
)


if "disturbance" not in baseline_features:
    raise ValueError(
        "The baseline model does not contain "
        "'disturbance'. Ablation cannot continue."
    )


# =========================================================
# CHECK REQUIRED COLUMNS
# =========================================================

required_columns = (
    set(baseline_features)
    | {TARGET}
)

missing_train = sorted(
    required_columns
    - set(train_df.columns)
)

missing_test = sorted(
    required_columns
    - set(test_df.columns)
)


if missing_train:
    raise ValueError(
        "Training data is missing columns: "
        f"{missing_train}"
    )

if missing_test:
    raise ValueError(
        "Test data is missing columns: "
        f"{missing_test}"
    )


# =========================================================
# PREPARE CATEGORICAL FEATURES
# =========================================================

for column in baseline_cat_features:

    train_df[column] = (
        train_df[column]
        .astype("string")
        .fillna("__MISSING__")
        .astype(str)
    )

    test_df[column] = (
        test_df[column]
        .astype("string")
        .fillna("__MISSING__")
        .astype(str)
    )


# =========================================================
# BASELINE MODEL TEST PERFORMANCE
# =========================================================

X_test_baseline = test_df[
    baseline_features
].copy()

y_test = (
    test_df[TARGET]
    .astype(float)
)


baseline_pred = baseline_model.predict(
    X_test_baseline
)


baseline_r2 = r2_score(
    y_test,
    baseline_pred
)

baseline_mae = mean_absolute_error(
    y_test,
    baseline_pred
)

baseline_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        baseline_pred
    )
)


# =========================================================
# CREATE 28-FEATURE VERSION
# REMOVE ONLY DISTURBANCE
# =========================================================

ablation_features = [
    feature
    for feature in baseline_features
    if feature != "disturbance"
]

ablation_cat_features = [
    feature
    for feature in baseline_cat_features
    if feature != "disturbance"
]


print(
    "\nAblation feature count:",
    len(ablation_features)
)

print(
    "Disturbance in ablation model:",
    "disturbance" in ablation_features
)


# =========================================================
# PREPARE TRAIN / TEST MATRICES
# =========================================================

X_train = train_df[
    ablation_features
].copy()

X_test = test_df[
    ablation_features
].copy()

y_train = (
    train_df[TARGET]
    .astype(float)
)


# =========================================================
# TRAIN 28-FEATURE CATBOOST MODEL
# SAME CONFIGURATION AS BASELINE
# =========================================================

ablation_model = CatBoostRegressor(
    loss_function="RMSE",
    eval_metric="RMSE",
    iterations=193,
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=3,
    random_seed=42,
    random_strength=1,
    bootstrap_type="MVS",
    grow_policy="SymmetricTree",
    verbose=False,
    allow_writing_files=False,
)


ablation_model.fit(
    X_train,
    y_train,
    cat_features=ablation_cat_features,
)


# =========================================================
# SAVE 28-FEATURE MODEL
# =========================================================

FINAL_MODEL_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

ablation_model.save_model(
    str(FINAL_MODEL_PATH)
)


print(
    "\nSaved 28-feature model:"
)

print(
    FINAL_MODEL_PATH
)


# =========================================================
# VERIFY SAVED MODEL
# =========================================================

verification_model = CatBoostRegressor()

verification_model.load_model(
    str(FINAL_MODEL_PATH)
)

saved_features = list(
    verification_model.feature_names_
)


print(
    "\nSaved model feature count:",
    len(saved_features)
)

print(
    "Disturbance in saved model:",
    "disturbance" in saved_features
)


if saved_features != ablation_features:
    raise ValueError(
        "Saved model feature list does not match "
        "the expected 28-feature list."
    )


# =========================================================
# EVALUATE 28-FEATURE MODEL
# =========================================================

ablation_pred = ablation_model.predict(
    X_test
)


ablation_r2 = r2_score(
    y_test,
    ablation_pred
)

ablation_mae = mean_absolute_error(
    y_test,
    ablation_pred
)

ablation_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        ablation_pred
    )
)


# =========================================================
# RESULTS
# =========================================================

print(
    "\n" + "=" * 75
)

print(
    "BASELINE - 29 FEATURES"
)

print(
    "=" * 75
)

print(
    f"R2   : {baseline_r2:.6f}"
)

print(
    f"MAE  : {baseline_mae:.6f}"
)

print(
    f"RMSE : {baseline_rmse:.6f}"
)


print(
    "\n" + "=" * 75
)

print(
    "WITHOUT DISTURBANCE - 28 FEATURES"
)

print(
    "=" * 75
)

print(
    f"R2   : {ablation_r2:.6f}"
)

print(
    f"MAE  : {ablation_mae:.6f}"
)

print(
    f"RMSE : {ablation_rmse:.6f}"
)


# =========================================================
# PERFORMANCE CHANGE
# =========================================================

r2_change = (
    ablation_r2
    - baseline_r2
)

mae_change = (
    ablation_mae
    - baseline_mae
)

rmse_change = (
    ablation_rmse
    - baseline_rmse
)


print(
    "\n" + "=" * 75
)

print(
    "CHANGE AFTER REMOVING DISTURBANCE"
)

print(
    "=" * 75
)

print(
    f"R2 change   : {r2_change:+.6f}"
)

print(
    f"MAE change  : {mae_change:+.6f}"
)

print(
    f"RMSE change : {rmse_change:+.6f}"
)


# =========================================================
# SIMPLE DECISION
# =========================================================

r2_loss = (
    baseline_r2
    - ablation_r2
)

mae_increase = (
    ablation_mae
    - baseline_mae
)

rmse_increase = (
    ablation_rmse
    - baseline_rmse
)


print(
    "\n" + "=" * 75
)

print(
    "FINAL DISTURBANCE DECISION"
)

print(
    "=" * 75
)


if (
    r2_loss <= 0.01
    and mae_increase <= 1.0
    and rmse_increase <= 1.0
):

    print(
        "REMOVE disturbance."
    )

    print(
        "The feature does not provide enough "
        "predictive improvement to justify "
        "requiring it at prediction time."
    )

    print(
        "The saved 28-feature model can be used "
        "for the next backend update."
    )

else:

    print(
        "KEEP disturbance for now."
    )

    print(
        "Removing it caused a noticeable "
        "performance reduction."
    )


print(
    "=" * 75
)