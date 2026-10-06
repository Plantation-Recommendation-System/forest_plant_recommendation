from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# =========================================================
# PATHS
# =========================================================

BACKEND_DIR = Path(__file__).resolve().parents[2]

CLIMATE_PROFILE_PATH = (
    BACKEND_DIR
    / "data"
    / "species_climate_profiles.csv"
)


# =========================================================
# CLIMATE VARIABLES
#
# Keys = live site-climate fields
# Values = profile-column base names
# =========================================================

CLIMATE_VARIABLES = {

    "climate_mean_annual_temp_c":
        "climate_mean_annual_temp_c",

    "climate_temp_seasonality_c":
        "climate_temp_seasonality_c",

    "climate_warmest_month_max_temp_c":
        "climate_warmest_month_max_temp_c",

    "climate_annual_precip_mm":
        "climate_annual_precip_mm",

    "climate_driest_month_precip_mm":
        "climate_driest_month_precip_mm",

    "climate_precip_seasonality_cv":
        "climate_precip_seasonality_cv",

    "climate_driest_quarter_monthly_precip_mm":
        "climate_driest_quarter_monthly_precip_mm",
}


# =========================================================
# SAFE NORMALIZED MISMATCH
#
# Same asymmetric logic used in Colab:
#
# site <= P50:
#     (P50 - site) / (P50 - P10)
#
# site > P50:
#     (site - P50) / (P90 - P50)
#
# Interpretation:
# 0   = at species median
# <=1 = inside P10–P90 envelope
# >1  = outside P10–P90 envelope
# =========================================================

def normalized_mismatch(
    site_value: float,
    p10: float,
    p50: float,
    p90: float,
) -> float:

    if any(
        pd.isna(x)
        for x in [
            site_value,
            p10,
            p50,
            p90,
        ]
    ):
        return np.nan

    if site_value <= p50:

        denominator = p50 - p10

        if denominator <= 0:

            return (
                0.0
                if site_value == p50
                else np.inf
            )

        return (
            p50 - site_value
        ) / denominator

    denominator = p90 - p50

    if denominator <= 0:

        return (
            0.0
            if site_value == p50
            else np.inf
        )

    return (
        site_value - p50
    ) / denominator


# =========================================================
# CLIMATE TIER
# =========================================================

def climate_tier(
    variables_inside_envelope: int,
) -> str:

    if variables_inside_envelope == 7:
        return "A_strict_match"

    if variables_inside_envelope == 6:
        return "B_near_match"

    if variables_inside_envelope == 5:
        return "C_partial_match"

    return "D_weak_match"


# =========================================================
# SERVICE
# =========================================================

class ClimateService:

    def __init__(
        self,
        profile_path: Path = CLIMATE_PROFILE_PATH,
    ):

        self.profile_path = profile_path

        self._profiles = None


    # -----------------------------------------------------
    # LOAD PROFILES ONCE
    # -----------------------------------------------------

    def load_profiles(
        self,
    ) -> pd.DataFrame:

        if self._profiles is None:

            if not self.profile_path.exists():

                raise FileNotFoundError(
                    "Climate profile file not found: "
                    f"{self.profile_path}"
                )

            self._profiles = pd.read_csv(
                self.profile_path,
                low_memory=False,
            )

        return self._profiles.copy()


    # -----------------------------------------------------
    # VALIDATE SITE CLIMATE
    # -----------------------------------------------------

    def validate_site_climate(
        self,
        site_climate: dict[str, float],
    ) -> None:

        missing = [

            field

            for field
            in CLIMATE_VARIABLES

            if field not in site_climate
        ]

        if missing:

            raise ValueError(
                "Missing site climate variables: "
                f"{missing}"
            )


        for field in CLIMATE_VARIABLES:

            value = site_climate[field]

            if value is None:

                raise ValueError(
                    f"Climate value is missing: {field}"
                )

            try:

                numeric_value = float(value)

            except (
                TypeError,
                ValueError,
            ) as error:

                raise ValueError(
                    f"Climate value must be numeric: "
                    f"{field}"
                ) from error


            if not np.isfinite(
                numeric_value
            ):

                raise ValueError(
                    f"Climate value is not finite: "
                    f"{field}"
                )


    # -----------------------------------------------------
    # RANK ALL SPECIES
    # -----------------------------------------------------

    def rank_species(
        self,
        site_climate: dict[str, float],
    ) -> pd.DataFrame:

        self.validate_site_climate(
            site_climate
        )

        profiles = self.load_profiles()


        mismatch_columns = []
        inside_columns = []


        # ================================================
        # EACH OF THE 7 CLIMATE VARIABLES
        # ================================================

        for (
            site_field,
            profile_base,
        ) in CLIMATE_VARIABLES.items():

            site_value = float(
                site_climate[
                    site_field
                ]
            )


            p10_col = (
                f"{profile_base}_p10"
            )

            p50_col = (
                f"{profile_base}_p50"
            )

            p90_col = (
                f"{profile_base}_p90"
            )


            required = {
                p10_col,
                p50_col,
                p90_col,
            }

            missing = (
                required
                -
                set(
                    profiles.columns
                )
            )


            if missing:

                raise ValueError(
                    "Climate profile file is missing "
                    f"columns: {sorted(missing)}"
                )


            mismatch_col = (
                f"mismatch_{site_field}"
            )

            inside_col = (
                f"inside_{site_field}"
            )


            profiles[
                mismatch_col
            ] = profiles.apply(

                lambda row:
                    normalized_mismatch(
                        site_value=site_value,
                        p10=row[p10_col],
                        p50=row[p50_col],
                        p90=row[p90_col],
                    ),

                axis=1,
            )


            profiles[
                inside_col
            ] = (

                (
                    site_value
                    >=
                    profiles[p10_col]
                )

                &

                (
                    site_value
                    <=
                    profiles[p90_col]
                )

            )


            mismatch_columns.append(
                mismatch_col
            )

            inside_columns.append(
                inside_col
            )


        # ================================================
        # SUMMARY METRICS
        # ================================================

        profiles[
            "climate_variables_inside"
        ] = (
            profiles[
                inside_columns
            ]
            .sum(
                axis=1
            )
            .astype(int)
        )


        profiles[
            "climate_median_mismatch"
        ] = (
            profiles[
                mismatch_columns
            ]
            .median(
                axis=1
            )
        )


        profiles[
            "climate_max_mismatch"
        ] = (
            profiles[
                mismatch_columns
            ]
            .max(
                axis=1
            )
        )


        profiles[
            "climate_tier"
        ] = (
            profiles[
                "climate_variables_inside"
            ]
            .apply(
                climate_tier
            )
        )


        # ================================================
        # EVIDENCE LABEL
        # ================================================

        def evidence_label(
            cell_count: Any,
        ) -> str:

            if pd.isna(
                cell_count
            ):
                return "unknown"

            count = int(
                cell_count
            )

            if count >= 100:
                return "strong"

            if count >= 20:
                return "moderate"

            if count >= 5:
                return "limited"

            return "very_limited"


        profiles[
            "climate_evidence_label"
        ] = (
            profiles[
                "climate_complete_chelsa_cells"
            ]
            .apply(
                evidence_label
            )
        )


        # ================================================
        # FINAL CLIMATE RANKING
        #
        # 1. More variables inside P10–P90
        # 2. Lower median mismatch
        # 3. Lower maximum mismatch
        # 4. More evidence cells
        # ================================================

        profiles = (
            profiles
            .sort_values(
                by=[
                    "climate_variables_inside",
                    "climate_median_mismatch",
                    "climate_max_mismatch",
                    "climate_complete_chelsa_cells",
                ],
                ascending=[
                    False,
                    True,
                    True,
                    False,
                ],
            )
            .reset_index(
                drop=True
            )
        )


        profiles[
            "climate_rank"
        ] = (
            np.arange(
                1,
                len(profiles) + 1,
            )
        )


        return profiles


    # -----------------------------------------------------
    # OPERATIONAL CLIMATE SHORTLIST
    #
    # top_n is a workflow choice,
    # NOT a biological threshold.
    # -----------------------------------------------------

    def get_shortlist(
        self,
        site_climate: dict[str, float],
        top_n: int = 50,
    ) -> pd.DataFrame:

        ranked = self.rank_species(
            site_climate
        )

        top_n = min(
            int(top_n),
            len(ranked),
        )

        shortlist = (
            ranked
            .head(
                top_n
            )
            .copy()
        )

        shortlist[
            "climate_shortlist_rule"
        ] = (
            f"operational_top_{top_n}"
        )

        return shortlist


climate_service = ClimateService()