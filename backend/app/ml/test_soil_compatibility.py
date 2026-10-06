from app.services.climate_service import (
    climate_service,
)

from app.services.soil_compatibility_service import (
    soil_compatibility_service,
)


# =========================================================
# SAME DEMO CLIMATE AS BEFORE
# =========================================================

SITE_CLIMATE = {

    "climate_mean_annual_temp_c":
        24.65,

    "climate_temp_seasonality_c":
        2.191,

    "climate_warmest_month_max_temp_c":
        36.35,

    "climate_annual_precip_mm":
        998.4,

    "climate_driest_month_precip_mm":
        1.0,

    "climate_precip_seasonality_cv":
        119.1,

    "climate_driest_quarter_monthly_precip_mm":
        5.0,
}


# =========================================================
# YOUR CURRENT TEST IMAGE RESULT
# =========================================================

USER_SOIL = {

    "clay":
        0.2001,

    "loam":
        0.6998,

    "sand":
        0.1001,
}


climate_candidates = (
    climate_service
    .get_shortlist(
        SITE_CLIMATE,
        top_n=50,
    )
)


matched = (
    soil_compatibility_service
    .match_candidates(
        candidates=climate_candidates,
        user_probabilities=USER_SOIL,
    )
)


print("=" * 60)
print("SOIL COMPATIBILITY TEST")
print("=" * 60)

print(
    "Climate candidates:",
    len(
        climate_candidates
    )
)


print(
    "Soil-matched rows:",
    len(
        matched
    )
)


print("\nMatch counts:")

print(
    matched[
        "soil_texture_match_label"
    ]
    .value_counts(
        dropna=False
    )
)


print("\nTop 15 by existing climate order:")

print(
    matched[
        [
            "climate_rank",
            "project_species_resolved",
            "climate_tier",
            "sand_fraction",
            "loam_fraction",
            "clay_fraction",
            "soil_texture_match_pct",
            "soil_texture_match_label",
        ]
    ]
    .head(15)
    .to_string(
        index=False
    )
)


print(
    "\nQuestionnaire example:"
)

print(
    soil_compatibility_service
    .questionnaire_consistency(
        predicted_class="Loam",
        wet_behavior="weak_ball",
    )
)


print(
    "\n✅ SOIL COMPATIBILITY TEST COMPLETE"
)