from app.services.climate_service import (
    climate_service,
)

from app.services.soil_compatibility_service import (
    soil_compatibility_service,
)

from app.services.environmental_screening_service import (
    environmental_screening_service,
)

from app.services.survival_service import (
    survival_service,
)


# =========================================================
# DEMO SITE
# =========================================================

LATITUDE = 18.5204
LONGITUDE = 73.8567

COUNTRY = "India"
PROVINCE = "Maharashtra"


SITE_CLIMATE = {

    "climate_mean_annual_temp_c": 24.65,

    "climate_temp_seasonality_c": 2.191,

    "climate_warmest_month_max_temp_c": 36.35,

    "climate_annual_precip_mm": 998.4,

    "climate_driest_month_precip_mm": 1.0,

    "climate_precip_seasonality_cv": 119.1,

    "climate_driest_quarter_monthly_precip_mm": 5.0,
}


# =========================================================
# CURRENT TEST SOIL IMAGE
# =========================================================

USER_SOIL = {

    "clay": 0.2001,

    "loam": 0.6998,

    "sand": 0.1001,
}


# =========================================================
# CLIMATE
# =========================================================

climate_candidates = (
    climate_service
    .get_shortlist(
        SITE_CLIMATE,
        top_n=50,
    )
)


# =========================================================
# SOIL
# =========================================================

soil_matched = (
    soil_compatibility_service
    .match_candidates(
        candidates=climate_candidates,
        user_probabilities=USER_SOIL,
    )
)


# =========================================================
# ENVIRONMENTAL SCREENING
# =========================================================

environmental_candidates = (
    environmental_screening_service
    .get_catboost_candidates(
        soil_matched
    )
)


# =========================================================
# CATBOOST
#
# Demo scenario only.
# =========================================================

predicted = (
    survival_service
    .predict_candidates(

        candidates=
            environmental_candidates,

        site_climate=
            SITE_CLIMATE,

        latitude=
            LATITUDE,

        longitude=
            LONGITUDE,

        country=
            COUNTRY,

        province=
            PROVINCE,

        forest_condition=
            "open_degraded",

        management_actions=[
            "comp_removal",
            "protection",
        ],

        target_horizon_months=
            24,

        planting_density=
            None,
    )
)


print("=" * 60)
print("CATBOOST SURVIVAL SERVICE TEST")
print("=" * 60)


print(
    "Environmental candidates:",
    len(
        environmental_candidates
    )
)

print(
    "Predictions:",
    len(
        predicted
    )
)


print(
    "\nPrediction range:",
    round(
        predicted[
            "predicted_survival_pct_raw"
        ].min(),
        2,
    ),
    "to",
    round(
        predicted[
            "predicted_survival_pct_raw"
        ].max(),
        2,
    ),
)


print(
    "\nSeen in training:",
    int(
        predicted[
            "seen_in_catboost_training"
        ].sum()
    )
)


print(
    "Unseen in training:",
    int(
        (
            ~predicted[
                "seen_in_catboost_training"
            ]
        ).sum()
    )
)


print("\nCandidates + survival support:")


print(
    predicted[
        [
            "environmental_rank",
            "project_species_resolved",
            "climate_tier",
            "soil_texture_match_pct",
            "predicted_survival_pct",
            "seen_in_catboost_training",
            "catboost_evidence_status",
        ]
    ]
    .sort_values(
        "environmental_rank"
    )
    .to_string(
        index=False
    )
)


print(
    "\nâœ… CATBOOST SURVIVAL SERVICE TEST COMPLETE"
)
