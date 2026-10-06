from pathlib import Path

import pandas as pd
from catboost import CatBoostRegressor


BACKEND_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    BACKEND_DIR
    / "models"
    / "catboost_survival_final.cbm"
)


print("=" * 75)
print("FINAL CATBOOST MODEL AUDIT")
print("=" * 75)


# =========================================================
# 1. FILE CHECK
# =========================================================

print("\nMODEL FILE")
print("-" * 75)

print("Path:", MODEL_PATH)
print("Exists:", MODEL_PATH.exists())


if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )


print(
    "Size MB:",
    round(
        MODEL_PATH.stat().st_size
        / 1024
        / 1024,
        3,
    )
)


# =========================================================
# 2. LOAD EXACT SAVED MODEL
# =========================================================

model = CatBoostRegressor()

model.load_model(
    str(MODEL_PATH)
)


print("\nMODEL TYPE")
print("-" * 75)

print(
    type(model)
)


# =========================================================
# 3. TREE COUNT
# =========================================================

print("\nTREE INFORMATION")
print("-" * 75)

print(
    "Tree count:",
    model.tree_count_
)


# =========================================================
# 4. EXACT FEATURE NAMES
# =========================================================

features = list(
    model.feature_names_
)


print("\nFEATURES")
print("-" * 75)

print(
    "Feature count:",
    len(features)
)


for i, feature in enumerate(
    features,
    start=1,
):

    print(
        f"{i:02d}. {feature}"
    )


# =========================================================
# 5. CATEGORICAL FEATURES
# =========================================================

cat_indices = list(
    model.get_cat_feature_indices()
)


cat_features = [

    features[index]

    for index
    in cat_indices
]


print("\nCATEGORICAL FEATURES")
print("-" * 75)

print(
    "Categorical count:",
    len(cat_features)
)

print(
    "Categorical indices:",
    cat_indices
)


for feature in cat_features:

    print(
        "-",
        feature
    )


# =========================================================
# 6. MODEL PARAMETERS
# =========================================================

print("\nMODEL PARAMETERS")
print("-" * 75)


params = model.get_all_params()


important_params = [

    "loss_function",
    "eval_metric",
    "iterations",
    "learning_rate",
    "depth",
    "l2_leaf_reg",
    "random_seed",
    "random_strength",
    "bootstrap_type",
    "grow_policy",
]


for key in important_params:

    if key in params:

        print(
            f"{key}:",
            params[key]
        )


# =========================================================
# 7. FEATURE IMPORTANCE
# =========================================================

print("\nFEATURE IMPORTANCE")
print("-" * 75)


importance = model.get_feature_importance()


importance_df = pd.DataFrame(
    {
        "feature": features,
        "importance": importance,
    }
)


importance_df = (
    importance_df
    .sort_values(
        "importance",
        ascending=False,
    )
    .reset_index(drop=True)
)


print(
    importance_df.to_string(
        index=False
    )
)


# =========================================================
# 8. SAFETY CHECK — FORBIDDEN FEATURES
# =========================================================

forbidden = {

    "row_id",
    "survival_per",
    "split_group_id",
    "split_set",
}


present_forbidden = (

    forbidden
    &
    set(features)
)


print("\nLEAKAGE CHECK")
print("-" * 75)


if present_forbidden:

    print(
        "❌ FORBIDDEN FEATURES FOUND:",
        sorted(
            present_forbidden
        )
    )

else:

    print(
        "✅ No target/split/row ID leakage "
        "features found in saved model."
    )


# =========================================================
# 9. EXPECTED CLIMATE FEATURES
# =========================================================

expected_climate = {

    "climate_mean_annual_temp_c",

    "climate_temp_seasonality_c",

    "climate_warmest_month_max_temp_c",

    "climate_annual_precip_mm",

    "climate_driest_month_precip_mm",

    "climate_precip_seasonality_cv",

    "climate_driest_quarter_monthly_precip_mm",
}


missing_climate = (

    expected_climate
    -
    set(features)
)


print("\nCLIMATE FEATURE CHECK")
print("-" * 75)


if missing_climate:

    print(
        "❌ Missing:",
        sorted(
            missing_climate
        )
    )

else:

    print(
        "✅ All 7 expected CHELSA "
        "features are present."
    )


print("\n" + "=" * 75)
print("✅ CATBOOST MODEL STRUCTURE AUDIT COMPLETE")
print("=" * 75)