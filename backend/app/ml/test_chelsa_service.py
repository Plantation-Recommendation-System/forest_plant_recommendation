from app.services.chelsa_service import get_site_climate


LATITUDE = 18.5204
LONGITUDE = 73.8567


print("=" * 60)
print("LIVE CHELSA EXTRACTION TEST")
print("=" * 60)


climate = get_site_climate(
    latitude=LATITUDE,
    longitude=LONGITUDE,
)


for key, value in climate.items():

    print(
        f"{key}: {value}"
    )


print("\nExpected approximately:")

print(
    "climate_mean_annual_temp_c: 24.65"
)

print(
    "climate_temp_seasonality_c: 2.191"
)

print(
    "climate_warmest_month_max_temp_c: 36.35"
)

print(
    "climate_annual_precip_mm: 998.4"
)

print(
    "climate_driest_month_precip_mm: 1.0"
)

print(
    "climate_precip_seasonality_cv: 119.1"
)

print(
    "climate_driest_quarter_monthly_precip_mm: 5.0"
)


print(
    "\n✅ LIVE CHELSA TEST COMPLETE"
)