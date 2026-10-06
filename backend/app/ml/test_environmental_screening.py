from app.services.climate_service import (
    climate_service,
)

from app.services.soil_compatibility_service import (
    soil_compatibility_service,
)

from app.services.environmental_screening_service import (
    environmental_screening_service,
)


# =========================================================
# DEMO CLIMATE
# =========================================================

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
# CURRENT TEST SOIL IMAGE RESULT
# =========================================================

USER_SOIL = {

    "clay": 0.2001,

    "loam": 0.6998,

    "sand": 0.1001,
}


# =========================================================
# CLIMATE TOP 50
# =========================================================

climate_candidates = (
    climate_service
    .get_shortlist(
        SITE_CLIMATE,
        top_n=50,
    )
)


# =========================================================
# SOIL MATCH
# =========================================================

soil_matched = (
    soil_compatibility_service
    .match_candidates(
        candidates=climate_candidates,
        user_probabilities=USER_SOIL,
    )
)


# =========================================================
# ENVIRONMENTAL HANDOFF
# =========================================================

ranked = (
    environmental_screening_service
    .rank_environmentally(
        soil_matched
    )
)


passed = (
    environmental_screening_service
    .get_catboost_candidates(
        soil_matched
    )
)


print("=" * 60)
print("ENVIRONMENTAL SCREENING TEST")
print("=" * 60)


print(
    "Climate shortlist:",
    len(climate_candidates)
)

print(
    "Soil-matched candidates:",
    len(soil_matched)
)

print(
    "Passed to CatBoost:",
    len(passed)
)


print("\nPassed by climate tier:")

print(
    passed[
        "climate_tier"
    ]
    .value_counts()
)


print("\nPassed candidates:")

print(
    passed[
        [
            "environmental_rank",
            "climate_rank",
            "project_species_resolved",
            "climate_tier",
            "soil_texture_match_pct",
            "soil_texture_match_label",
            "catboost_handoff_reason",
        ]
    ]
    .to_string(
        index=False
    )
)


print("\n" + "=" * 60)
print("NON-PASSED / FALLBACK COUNT")
print("=" * 60)

print(
    len(
        ranked[
            ~ranked[
                "catboost_handoff"
            ]
        ]
    )
)


print(
    "\n✅ ENVIRONMENTAL SCREENING TEST COMPLETE"
)