from typing import Any

from pydantic import BaseModel, Field


class RecommendationRequest(BaseModel):
    location: str = Field(min_length=1)
    water: str | None = None
    sunlight: str | None = None
    maintenance: str | None = None
    site_type: str | None = None
    purpose: str | None = None


class RecommendationResponse(BaseModel):
    success: bool = True
    count: int
    recommendations: list[dict[str, Any]]
    mode: str = "historical_survival_dataset_only"
