from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    UploadFile,
)

from app.schemas.recommendation import (
    RecommendationResponse,
)

from app.services.recommendation_service import (
    recommendation_service,
)


router = APIRouter()


# =========================================================
# HEALTH
# =========================================================

@router.get("/health")
def health_check():

    return {
        "status": "ok",
        "service": "recommendation-api",
        "mode": (
            "live_environmental_"
            "recommendation_pipeline"
        ),
    }


# =========================================================
# LIVE RECOMMENDATION ENDPOINT
# =========================================================

@router.post(
    "/recommendations",
    response_model=RecommendationResponse,
)
async def create_recommendations(

    location: str | None = Form(
        None
    ),

    latitude: float | None = Form(
        None
    ),

    longitude: float | None = Form(
        None
    ),

    soil_image: UploadFile = File(...),

    wet_behavior: str = Form(...),
):

    try:

        # -----------------------------------------------
        # Validate uploaded image type
        # -----------------------------------------------

        if (
            soil_image.content_type
            and not soil_image.content_type.startswith(
                "image/"
            )
        ):

            raise ValueError(
                "soil_image must be an image file."
            )

        # -----------------------------------------------
        # Read uploaded image
        # -----------------------------------------------

        image_bytes = await soil_image.read()

        if not image_bytes:

            raise ValueError(
                "Uploaded soil image is empty."
            )

        # -----------------------------------------------
        # Run complete recommendation pipeline
        # -----------------------------------------------

        return (
            recommendation_service
            .get_recommendations(

                location=
                    location,

                latitude=
                    latitude,

                longitude=
                    longitude,

                soil_image_bytes=
                    image_bytes,

                wet_behavior=
                    wet_behavior,

                forest_condition=
                    "open_degraded",

                management_actions=
                    [],

                target_horizon_months=
                    24.0,

                planting_density=
                    None,

                top_n=
                    10,
            )
        )

    except FileNotFoundError as error:

        raise HTTPException(
            status_code=503,
            detail=str(error),
        ) from error

    except RuntimeError as error:

        raise HTTPException(
            status_code=503,
            detail=str(error),
        ) from error

    except ValueError as error:

        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error

    finally:

        await soil_image.close()







