from __future__ import annotations

import json
from typing import Any, Iterable

from app.services.location_service import location_service
from app.services.chelsa_service import get_site_climate
from app.services.climate_service import climate_service
from app.services.soil_compatibility_service import (
    soil_compatibility_service,
)
from app.services.environmental_screening_service import (
    environmental_screening_service,
)
from app.services.recommendation_logic import (
    get_top_recommendations,
)


# =========================================================
# FIELDS RETURNED TO THE FRONTEND
# =========================================================

RECOMMENDATION_API_COLUMNS = [
    "final_recommendation_rank",
    "project_species_resolved",
    "climate_rank",
    "climate_tier",
    "climate_variables_inside",
    "soil_texture_match_pct",
    "soil_texture_match_label",
    "predicted_survival_pct",
    "prediction_range_status",
    "seen_in_catboost_training",
    "catboost_coverage_status",
    "catboost_evidence_status",
    "climate_interpretation",
    "soil_interpretation",
    "recommendation_basis",
    "overall_evidence_label",
    "recommendation_display_group",
]


def dataframe_to_api_records(df):
    """
    Convert a pandas DataFrame into normal Python values
    that FastAPI/Pydantic can safely return as JSON.

    Pandas/numpy NaN values become JSON null.
    """

    if df is None or len(df) == 0:
        return []

    return json.loads(
        df.to_json(
            orient="records"
        )
    )


class RecommendationService:
    """
    Main live plantation recommendation pipeline.

    Ranking policy:

    1. Climate suitability is primary.
    2. Soil compatibility is secondary.
    3. Environmental rank is preserved.
    4. CatBoost survival is supporting evidence only.
    """

    def get_recommendations(
        self,
        *,
        location: str | None = None,
        latitude: float | None = None,
        longitude: float | None = None,
        soil_image_bytes: bytes,
        wet_behavior: str | None = None,
        forest_condition: str = "open_degraded",
        management_actions: Iterable[str] | None = None,
        target_horizon_months: float = 24.0,
        planting_density: float | None = None,
        top_n: int = 10,
    ) -> dict[str, Any]:

        # =====================================================
        # 0. BASIC VALIDATION
        # =====================================================

        if location is not None:
            location = str(
                location
            ).strip()

            if not location:
                location = None

        if (
            latitude is None
            and longitude is not None
        ):
            raise ValueError(
                "Latitude is required when longitude is provided."
            )

        if (
            longitude is None
            and latitude is not None
        ):
            raise ValueError(
                "Longitude is required when latitude is provided."
            )

        if (
            latitude is None
            and longitude is None
            and not location
        ):
            raise ValueError(
                "Provide either detected latitude/longitude "
                "or a manual location."
            )

        if not soil_image_bytes:
            raise ValueError(
                "A soil image is required."
            )

        if wet_behavior is not None:

            wet_behavior = str(
                wet_behavior
            ).strip()

            allowed_wet_behaviors = {
                "loose_gritty",
                "weak_ball",
                "sticky",
            }

            if (
                wet_behavior
                not in allowed_wet_behaviors
            ):
                raise ValueError(
                    "wet_behavior must be one of: "
                    "loose_gritty, weak_ball, sticky."
                )

        forest_condition = str(
            forest_condition
        ).strip()

        if not forest_condition:
            raise ValueError(
                "forest_condition cannot be empty."
            )

        target_horizon_months = float(
            target_horizon_months
        )

        if target_horizon_months <= 0:
            raise ValueError(
                "target_horizon_months must be greater than 0."
            )

        if planting_density is not None:

            planting_density = float(
                planting_density
            )

            if planting_density <= 0:
                raise ValueError(
                    "planting_density must be greater than 0."
                )

        top_n = int(
            top_n
        )

        if top_n <= 0:
            raise ValueError(
                "top_n must be greater than 0."
            )

        if management_actions is None:

            management_actions = []

        else:

            management_actions = [
                str(action).strip()

                for action
                in management_actions

                if str(action).strip()
            ]

        # =====================================================
        # 1. LOCATION
        #
        # Example:
        # Pune, Maharashtra, India
        #
        # becomes:
        #
        # latitude
        # longitude
        # country
        # province
        # =====================================================

        if (
            latitude is not None
            and longitude is not None
        ):

            latitude = float(
                latitude
            )

            longitude = float(
                longitude
            )

            site = (
                location_service
                .reverse_geocode(
                    latitude,
                    longitude,
                )
            )

        else:

            site = (
                location_service
                .geocode(
                    location
                )
            )

            latitude = float(
                site["latitude"]
            )

            longitude = float(
                site["longitude"]
            )

        country = str(
            site["country"]
        )

        province = str(
            site["province"]
        )

        # =====================================================
        # 2. LIVE CHELSA CLIMATE
        #
        # Extract all seven climate variables automatically
        # using latitude and longitude.
        # =====================================================

        site_climate = get_site_climate(
            latitude=latitude,
            longitude=longitude,
        )

        # =====================================================
        # 3. CLIMATE SHORTLIST
        #
        # Climate is the PRIMARY environmental filter.
        # =====================================================

        climate_candidates = (
            climate_service
            .get_shortlist(
                site_climate,
                top_n=50,
            )
        )

        climate_count = int(
            len(
                climate_candidates
            )
        )

        if climate_count == 0:

            return self._empty_response(
                site=site,
                site_climate=site_climate,
                soil_prediction={},
                forest_condition=forest_condition,
                management_actions=management_actions,
                target_horizon_months=
                    target_horizon_months,
                planting_density=
                    planting_density,
                climate_count=0,
                soil_count=0,
                environmental_count=0,
                catboost_count=0,
            )

        # =====================================================
        # 4. SOIL IMAGE CLASSIFICATION
        #
        # Lazy import is intentional.
        # PyTorch is loaded only after CHELSA extraction.
        # =====================================================

        from app.services.soil_service import (
            soil_image_service,
        )

        soil_prediction = (
            soil_image_service
            .predict_bytes(
                soil_image_bytes
            )
        )

        soil_probabilities = (
            soil_prediction.get(
                "probabilities"
            )
        )

        if not isinstance(
            soil_probabilities,
            dict,
        ):

            raise ValueError(
                "Soil model did not return valid probabilities."
            )

        # =====================================================
        # SOIL QUESTIONNAIRE CROSS-CHECK
        #
        # The soil image gives the model's broad texture class.
        # The user's wet-soil observation provides an
        # independent field clue.
        #
        # This does NOT create a weighted score.
        # =====================================================

        predicted_soil_class = (
            soil_prediction.get(
                "predicted_class",
                ""
            )
        )

        soil_questionnaire_consistency = (
            soil_compatibility_service
            .questionnaire_consistency(
                predicted_class=
                    predicted_soil_class,

                wet_behavior=
                    wet_behavior,
            )
            if wet_behavior
            else "questionnaire_unknown"
        )

        soil_prediction[
            "wet_behavior"
        ] = wet_behavior

        soil_prediction[
            "questionnaire_consistency"
        ] = (
            soil_questionnaire_consistency
        )

        # =====================================================
        # 5. SOIL COMPATIBILITY
        #
        # Soil is SECONDARY evidence.
        # =====================================================

        soil_matched = (
            soil_compatibility_service
            .match_candidates(
                candidates=
                    climate_candidates,

                user_probabilities=
                    soil_probabilities,
            )
        )

        soil_matched[
            "soil_questionnaire_consistency"
        ] = (
            soil_questionnaire_consistency
        )

        soil_count = int(
            len(
                soil_matched
            )
        )

        # =====================================================
        # 6. ENVIRONMENTAL SCREENING
        #
        # Determines which climate + soil candidates are
        # appropriate to send to CatBoost.
        # =====================================================

        environmental_candidates = (
            environmental_screening_service
            .get_catboost_candidates(
                soil_matched
            )
        )

        environmental_count = int(
            len(
                environmental_candidates
            )
        )

        if environmental_count == 0:

            return self._empty_response(
                site=site,
                site_climate=site_climate,
                soil_prediction=soil_prediction,
                forest_condition=forest_condition,
                management_actions=management_actions,
                target_horizon_months=
                    target_horizon_months,
                planting_density=
                    planting_density,
                climate_count=
                    climate_count,
                soil_count=
                    soil_count,
                environmental_count=0,
                catboost_count=0,
            )

        # =====================================================
        # 7. CATBOOST SURVIVAL SUPPORT
        #
        # CatBoost DOES NOT determine final ranking.
        # It adds supporting survival evidence.
        # =====================================================

        from app.services.survival_service import (
            survival_service,
        )

        predicted = (
            survival_service
            .predict_candidates(
                candidates=
                    environmental_candidates,

                site_climate=
                    site_climate,

                latitude=
                    latitude,

                longitude=
                    longitude,

                country=
                    country,

                province=
                    province,

                forest_condition=
                    forest_condition,

                management_actions=
                    management_actions,

                target_horizon_months=
                    target_horizon_months,

                planting_density=
                    planting_density,
            )
        )

        catboost_count = int(
            len(
                predicted
            )
        )

        # =====================================================
        # 8. FINAL RECOMMENDATIONS
        #
        # Existing environmental_rank is preserved.
        #
        # CatBoost does NOT re-rank species.
        # =====================================================

        top_recommendations = (
            get_top_recommendations(
                candidates=
                    predicted,

                top_n=
                    top_n,

                scenario_status=
                    "live_assessment",
            )
        )

        # =====================================================
        # 9. KEEP FRONTEND-FRIENDLY COLUMNS
        # =====================================================

        available_columns = [

            column

            for column
            in RECOMMENDATION_API_COLUMNS

            if column
            in top_recommendations.columns
        ]

        output_df = (
            top_recommendations[
                available_columns
            ]
            .copy()
        )

        recommendations = (
            dataframe_to_api_records(
                output_df
            )
        )

        # =====================================================
        # 10. FINAL API RESPONSE
        # =====================================================

        return {

            "success":
                True,

            "count":
                len(
                    recommendations
                ),

            "recommendations":
                recommendations,

            "mode":
                (
                    "live_environmental_"
                    "recommendation_pipeline"
                ),

            "site":
                {
                    "query":
                        site.get(
                            "query"
                        ),

                    "display_name":
                        site.get(
                            "display_name"
                        ),

                    "latitude":
                        latitude,

                    "longitude":
                        longitude,

                    "country":
                        country,

                    "province":
                        province,

                    "city":
                        site.get(
                            "city"
                        ),
                },

            "site_climate":
                {
                    key:
                        float(
                            value
                        )

                    for key, value
                    in site_climate.items()
                },

            "soil_prediction":
                soil_prediction,

            "scenario":
                {
                    "forest_condition":
                        forest_condition,

                    "management_actions":
                        management_actions,

                    "target_horizon_months":
                        target_horizon_months,

                    "planting_density":
                        planting_density,

                    "ranking_policy":
                        (
                            "Climate primary; "
                            "soil secondary; "
                            "CatBoost survival is "
                            "supporting evidence only."
                        ),
                },

            "pipeline_counts":
                {
                    "climate_candidates":
                        climate_count,

                    "soil_matched_candidates":
                        soil_count,

                    "environmental_candidates":
                        environmental_count,

                    "catboost_predictions":
                        catboost_count,

                    "final_recommendations":
                        int(
                            len(
                                recommendations
                            )
                        ),
                },
        }


    # =========================================================
    # EMPTY RESULT RESPONSE
    # =========================================================

    def _empty_response(
        self,
        *,
        site: dict[str, Any],
        site_climate: dict[str, float],
        soil_prediction: dict[str, Any],
        forest_condition: str,
        management_actions: list[str],
        target_horizon_months: float,
        planting_density: float | None,
        climate_count: int,
        soil_count: int,
        environmental_count: int,
        catboost_count: int,
    ) -> dict[str, Any]:

        return {

            "success":
                True,

            "count":
                0,

            "recommendations":
                [],

            "mode":
                (
                    "live_environmental_"
                    "recommendation_pipeline"
                ),

            "site":
                site,

            "site_climate":
                {
                    key:
                        float(
                            value
                        )

                    for key, value
                    in site_climate.items()
                },

            "soil_prediction":
                soil_prediction,

            "scenario":
                {
                    "forest_condition":
                        forest_condition,

                    "management_actions":
                        management_actions,

                    "target_horizon_months":
                        float(
                            target_horizon_months
                        ),

                    "planting_density":
                        planting_density,

                    "ranking_policy":
                        (
                            "Climate primary; "
                            "soil secondary; "
                            "CatBoost survival is "
                            "supporting evidence only."
                        ),
                },

            "pipeline_counts":
                {
                    "climate_candidates":
                        int(
                            climate_count
                        ),

                    "soil_matched_candidates":
                        int(
                            soil_count
                        ),

                    "environmental_candidates":
                        int(
                            environmental_count
                        ),

                    "catboost_predictions":
                        int(
                            catboost_count
                        ),

                    "final_recommendations":
                        0,
                },
        }


recommendation_service = RecommendationService()



