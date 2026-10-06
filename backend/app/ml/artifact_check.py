from pathlib import Path

import pandas as pd
import torch
from catboost import CatBoostRegressor


# =========================================================
# PROJECT PATHS
# =========================================================

BACKEND_DIR = Path(__file__).resolve().parents[2]

MODEL_DIR = BACKEND_DIR / "models"
DATA_DIR = BACKEND_DIR / "data"


CATBOOST_PATH = MODEL_DIR / "catboost_survival_final.cbm"

SOIL_MODEL_PATH = MODEL_DIR / "efficientnet_b0_soil_final.pth"

CLIMATE_PATH = DATA_DIR / "species_climate_profiles.csv"

SOIL_PROFILE_PATH = DATA_DIR / "species_soil_profiles.csv"

SOIL_OBS_PATH = DATA_DIR / "species_soil_texture_observations.csv"

METADATA_PATH = DATA_DIR / "species_metadata.csv"


print("=" * 60)
print("PRODUCTION ARTIFACT CHECK")
print("=" * 60)


# =========================================================
# 1. CHECK FILES EXIST
# =========================================================

files = {
    "CatBoost model": CATBOOST_PATH,
    "Soil image model": SOIL_MODEL_PATH,
    "Climate profiles": CLIMATE_PATH,
    "Soil profiles": SOIL_PROFILE_PATH,
    "Soil observations": SOIL_OBS_PATH,
    "Species metadata": METADATA_PATH,
}

all_found = True

for name, path in files.items():

    exists = path.exists()

    print(
        "✅" if exists else "❌",
        name,
        "->",
        path
    )

    all_found = all_found and exists


if not all_found:

    raise FileNotFoundError(
        "One or more production artifacts are missing."
    )


# =========================================================
# 2. LOAD CSV DATA
# =========================================================

climate_profiles = pd.read_csv(
    CLIMATE_PATH,
    low_memory=False
)

soil_profiles = pd.read_csv(
    SOIL_PROFILE_PATH,
    low_memory=False
)

soil_observations = pd.read_csv(
    SOIL_OBS_PATH,
    low_memory=False
)

species_metadata = pd.read_csv(
    METADATA_PATH,
    low_memory=False
)


print("\n" + "=" * 60)
print("DATASET CHECK")
print("=" * 60)

print(
    "Climate profile rows:",
    len(climate_profiles)
)

print(
    "Soil profile rows:",
    len(soil_profiles)
)

print(
    "Soil observation rows:",
    len(soil_observations)
)

print(
    "Species metadata rows:",
    len(species_metadata)
)


# =========================================================
# 3. LOAD CATBOOST MODEL
# =========================================================

catboost_model = CatBoostRegressor()

catboost_model.load_model(
    str(CATBOOST_PATH)
)


print("\n" + "=" * 60)
print("CATBOOST CHECK")
print("=" * 60)

print(
    "Feature count:",
    len(catboost_model.feature_names_)
)

print(
    "Categorical feature indices:",
    catboost_model.get_cat_feature_indices()
)

print("\nFeatures:")

for i, feature in enumerate(
    catboost_model.feature_names_,
    start=1
):

    print(
        f"{i:02d}. {feature}"
    )


# =========================================================
# 4. LOAD SOIL MODEL CHECKPOINT
# =========================================================

try:

    soil_checkpoint = torch.load(
        SOIL_MODEL_PATH,
        map_location="cpu",
        weights_only=False
    )

except TypeError:

    soil_checkpoint = torch.load(
        SOIL_MODEL_PATH,
        map_location="cpu"
    )


print("\n" + "=" * 60)
print("SOIL MODEL CHECK")
print("=" * 60)

print(
    "Checkpoint type:",
    type(soil_checkpoint).__name__
)

if isinstance(
    soil_checkpoint,
    dict
):

    print(
        "Checkpoint keys:",
        list(
            soil_checkpoint.keys()
        )
    )


# =========================================================
# 5. COLUMN INSPECTION
# =========================================================

print("\n" + "=" * 60)
print("CLIMATE PROFILE COLUMNS")
print("=" * 60)

print(
    climate_profiles.columns.tolist()
)


print("\n" + "=" * 60)
print("SOIL PROFILE COLUMNS")
print("=" * 60)

print(
    soil_profiles.columns.tolist()
)


print("\n" + "=" * 60)
print("SOIL OBSERVATION COLUMNS")
print("=" * 60)

print(
    soil_observations.columns.tolist()
)


print("\n" + "=" * 60)
print("SPECIES METADATA COLUMNS")
print("=" * 60)

print(
    species_metadata.columns.tolist()
)


# =========================================================
# 6. BASIC COVERAGE CHECKS
# =========================================================

print("\n" + "=" * 60)
print("BASIC COVERAGE")
print("=" * 60)


if "project_species_resolved" in climate_profiles.columns:

    print(
        "Unique climate species:",
        climate_profiles[
            "project_species_resolved"
        ].nunique()
    )


if "project_species_resolved" in soil_profiles.columns:

    print(
        "Unique soil-profile species:",
        soil_profiles[
            "project_species_resolved"
        ].nunique()
    )


if "project_species_resolved" in soil_observations.columns:

    print(
        "Unique soil-observation species:",
        soil_observations[
            "project_species_resolved"
        ].nunique()
    )


if "project_species_resolved" in species_metadata.columns:

    print(
        "Unique metadata species:",
        species_metadata[
            "project_species_resolved"
        ].nunique()
    )


print("\n" + "=" * 60)
print("RESULT")
print("=" * 60)

print(
    "✅ ALL PRODUCTION ARTIFACTS LOADED SUCCESSFULLY"
)