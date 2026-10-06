from pathlib import Path

from app.services.soil_service import (
    soil_image_service,
)


BACKEND_DIR = Path(__file__).resolve().parents[2]

IMAGE_PATH = (
    BACKEND_DIR
    / "test_assets"
    / "soil_test.png"
)


print("=" * 60)
print("SOIL IMAGE SERVICE TEST")
print("=" * 60)


result = soil_image_service.predict_file(
    IMAGE_PATH
)


print(
    "Predicted class:",
    result[
        "predicted_class"
    ]
)


print(
    "Confidence:",
    round(
        result[
            "confidence"
        ]
        * 100,
        2,
    ),
    "%",
)


print("\nProbabilities:")


for soil_class, probability in (
    result[
        "probabilities"
    ]
    .items()
):

    print(
        f"{soil_class}: "
        f"{probability * 100:.2f}%"
    )


print(
    "\n✅ SOIL IMAGE SERVICE TEST COMPLETE"
)