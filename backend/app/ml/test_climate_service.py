from app.services.climate_service import climate_service


# =========================================================
# SAME DEMO SITE CLIMATE USED IN COLAB
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


ranked = climate_service.rank_species(
    SITE_CLIMATE
)


print("=" * 60)
print("CLIMATE SERVICE TEST")
print("=" * 60)

print(
    "Species ranked:",
    len(ranked)
)


print("\nTier counts:")

print(
    ranked[
        "climate_tier"
    ]
    .value_counts()
)


print("\nTop 15:")

print(
    ranked[
        [
            "climate_rank",
            "project_species_resolved",
            "climate_variables_inside",
            "climate_tier",
            "climate_median_mismatch",
            "climate_max_mismatch",
            "climate_complete_chelsa_cells",
            "climate_evidence_label",
        ]
    ]
    .head(15)
    .to_string(
        index=False
    )
)


shortlist = (
    climate_service
    .get_shortlist(
        SITE_CLIMATE,
        top_n=50,
    )
)


print(
    "\nClimate shortlist rows:",
    len(shortlist)
)


print("\n" + "=" * 60)
print("EXPECTED COLAB CHECK")
print("=" * 60)


tier_counts = (
    ranked[
        "climate_tier"
    ]
    .value_counts()
    .to_dict()
)


print(
    "A strict:",
    tier_counts.get(
        "A_strict_match",
        0,
    )
)

print(
    "B near:",
    tier_counts.get(
        "B_near_match",
        0,
    )
)

print(
    "C partial:",
    tier_counts.get(
        "C_partial_match",
        0,
    )
)

print(
    "D weak:",
    tier_counts.get(
        "D_weak_match",
        0,
    )
)


print(
    "\n✅ CLIMATE SERVICE TEST COMPLETE"
)