FOREST PLANTATION RECOMMENDATION — DEPLOYMENT EXPORT
====================================================

This folder contains ONLY production artifacts exported
from the Google Colab research/training project.

DO NOT run training in the website backend.

FILES
-----

models/catboost_survival_final.cbm
    Frozen CatBoost survival-support model.

models/efficientnet_b0_soil_final.pth
    Frozen EfficientNet-B0 soil-image model.

data/species_climate_profiles.csv
    Permanent CHELSA native-occurrence climate profiles.

data/species_soil_profiles.csv
    Permanent SoilGrids species soil profiles.

data/species_metadata.csv
    Compact taxonomy + wood-density metadata needed
    for CatBoost candidate rows.

data/species_soil_texture_observations.csv
    Complete SoilGrids texture observations, if found.

app/services/recommendation_logic.py
    Phase 69 final recommendation logic.

manifests/catboost_schema.json
    Exact 29-feature CatBoost schema.

manifests/soil_model_info.json
    Soil-model checkpoint inspection.

deployment_manifest.json
    Artifact manifest and hashes.

NEXT
----
Merge these files into the EXISTING backend directory
of the React + FastAPI GitHub repository.