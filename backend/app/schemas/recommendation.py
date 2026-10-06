from typing import Any

from pydantic import BaseModel, Field


class RecommendationRequest(BaseModel):
    """
    Non-file inputs used by the recommendation pipeline.

    The soil image itself is uploaded separately as multipart
    form-data by the API route.
    """

    location: str = Field(
        min_length=1
    )

    forest_condition: str = (
        "open_degraded"
    )

    management_actions: list[str] = Field(
        default_factory=lambda: [
            "comp_removal",
            "protection",
        ]
    )

    target_horizon_months: float = Field(
        default=24.0,
        gt=0,
    )

    planting_density: float | None = Field(
        default=None,
        gt=0,
    )


class RecommendationResponse(BaseModel):
    success: bool = True

    count: int

    recommendations: list[
        dict[str, Any]
    ]

    mode: str = (
        "live_environmental_recommendation_pipeline"
    )

    site: dict[str, Any]

    site_climate: dict[
        str,
        float,
    ]

    soil_prediction: dict[
        str,
        Any,
    ]

    scenario: dict[
        str,
        Any,
    ]

    pipeline_counts: dict[
        str,
        int,
    ]
