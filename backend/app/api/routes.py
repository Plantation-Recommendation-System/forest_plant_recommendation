from fastapi import APIRouter, HTTPException

from app.schemas.recommendation import (
    RecommendationRequest,
    RecommendationResponse,
)
from app.services.recommendation_service import recommendation_service

router = APIRouter()


@router.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "recommendation-api",
        "survival_dataset_available": recommendation_service.survival_file.exists(),
    }


@router.post(
    "/recommendations",
    response_model=RecommendationResponse,
)
def create_recommendations(request: RecommendationRequest):
    try:
        return recommendation_service.get_recommendations(request)
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
