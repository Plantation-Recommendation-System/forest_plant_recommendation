from typing import Dict

import numpy as np
import rasterio


CHELSA_BASE_URL = (
    "https://os.zhdk.cloud.switch.ch/"
    "chelsav2/GLOBAL/climatologies/"
    "1981-2010/bio"
)


# =========================================================
# CHELSA VARIABLE CONFIG
#
# scale and offset match the project preprocessing.
# =========================================================

CHELSA_VARIABLES = {

    "climate_mean_annual_temp_c": {
        "bio": 1,
        "scale": 0.1,
        "offset": -273.15,
        "post_scale": 1.0,
    },

    "climate_temp_seasonality_c": {
        "bio": 4,
        "scale": 0.1,
        "offset": 0.0,

        # Project convention used during profile/model creation
        "post_scale": 0.01,
    },

    "climate_warmest_month_max_temp_c": {
        "bio": 5,
        "scale": 0.1,
        "offset": -273.15,
        "post_scale": 1.0,
    },

    "climate_annual_precip_mm": {
        "bio": 12,
        "scale": 0.1,
        "offset": 0.0,
        "post_scale": 1.0,
    },

    "climate_driest_month_precip_mm": {
        "bio": 14,
        "scale": 0.1,
        "offset": 0.0,
        "post_scale": 1.0,
    },

    "climate_precip_seasonality_cv": {
        "bio": 15,
        "scale": 0.1,
        "offset": 0.0,
        "post_scale": 1.0,
    },

    "climate_driest_quarter_monthly_precip_mm": {
        "bio": 17,
        "scale": 0.1,
        "offset": 0.0,
        "post_scale": 1.0,
    },
}


def _chelsa_url(
    bio_number: int,
) -> str:

    return (
        f"{CHELSA_BASE_URL}/"
        f"CHELSA_bio{bio_number}_"
        f"1981-2010_V.2.1.tif"
    )


def _extract_raw_pixel(
    url: str,
    latitude: float,
    longitude: float,
) -> float:

    # /vsicurl/ tells GDAL/rasterio to use
    # HTTP range requests rather than downloading
    # the whole GeoTIFF.
    remote_path = (
        f"/vsicurl/{url}"
    )

    with rasterio.open(
        remote_path
    ) as dataset:

        # Validate coordinate
        if not (
            dataset.bounds.left
            <= longitude
            <= dataset.bounds.right
            and
            dataset.bounds.bottom
            <= latitude
            <= dataset.bounds.top
        ):

            raise ValueError(
                "Location lies outside the CHELSA raster extent."
            )

        row, col = dataset.index(
            longitude,
            latitude,
        )

        value = dataset.read(
            1,
            window=(
                (row, row + 1),
                (col, col + 1),
            ),
            masked=True,
        )[0, 0]

        if np.ma.is_masked(
            value
        ):

            raise ValueError(
                "CHELSA returned NoData for this location."
            )

        return float(value)


def get_site_climate(
    latitude: float,
    longitude: float,
) -> Dict[str, float]:

    latitude = float(
        latitude
    )

    longitude = float(
        longitude
    )

    if not (
        -90
        <= latitude
        <= 90
    ):

        raise ValueError(
            "Latitude must be between -90 and 90."
        )

    if not (
        -180
        <= longitude
        <= 180
    ):

        raise ValueError(
            "Longitude must be between -180 and 180."
        )

    result = {}

    for (
        output_field,
        config,
    ) in CHELSA_VARIABLES.items():

        bio_number = config[
            "bio"
        ]

        url = _chelsa_url(
            bio_number
        )

        raw_value = _extract_raw_pixel(
            url=url,
            latitude=latitude,
            longitude=longitude,
        )

        transformed = (
            raw_value
            * config[
                "scale"
            ]
            +
            config[
                "offset"
            ]
        )

        transformed = (
            transformed
            * config[
                "post_scale"
            ]
        )

        result[
            output_field
        ] = round(
            float(
                transformed
            ),
            4,
        )

    return result