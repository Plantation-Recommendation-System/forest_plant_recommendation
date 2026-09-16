import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


# ============================================================
# PROJECT PATHS
# ============================================================

BASE = Path(__file__).resolve().parents[1]

SOIL_MODEL_FILE = (
    BASE
    / "models"
    / "soil"
    / "soil_efficientnet_b0_final.pt"
)

METRICS_FILE = (
    BASE
    / "models"
    / "soil"
    / "soil_model_metrics.json"
)


# ============================================================
# SOIL CLASSES
# ============================================================

SOIL_CLASSES = [
    "Alluvial",
    "Arid",
    "Black",
    "Laterite",
    "Mountain",
    "Red",
    "Yellow",
]


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# IMAGE TRANSFORM
# ============================================================

IMAGE_SIZE = 224

transform = transforms.Compose(
    [
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406,
            ],
            std=[
                0.229,
                0.224,
                0.225,
            ],
        ),
    ]
)


# ============================================================
# MODEL
# ============================================================

def build_model():

    model = models.efficientnet_b0(
        weights=None
    )

    num_features = (
        model.classifier[1].in_features
    )

    model.classifier[1] = nn.Linear(
        num_features,
        len(SOIL_CLASSES),
    )

    return model


# ============================================================
# LOAD MODEL
# ============================================================

def load_soil_model():

    if not SOIL_MODEL_FILE.exists():

        raise FileNotFoundError(
            "\nSoil model not found.\n\n"
            f"Expected:\n{SOIL_MODEL_FILE}\n\n"
            "Place soil_efficientnet_b0_final.pt "
            "inside models/soil/"
        )

    model = build_model()

    checkpoint = torch.load(
        SOIL_MODEL_FILE,
        map_location=DEVICE,
        weights_only=False,
    )

    # --------------------------------------------------------
    # Handle different possible checkpoint formats
    # --------------------------------------------------------

    if isinstance(checkpoint, dict):

        if "state_dict" in checkpoint:

            state_dict = checkpoint["state_dict"]

        elif "model_state_dict" in checkpoint:

            state_dict = checkpoint[
                "model_state_dict"
            ]

        else:

            # Could itself be a state dictionary
            state_dict = checkpoint

    else:

        state_dict = checkpoint

    # Remove DataParallel prefix if present
    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):

            key = key[
                len("module.") :
            ]

        cleaned_state_dict[key] = value

    model.load_state_dict(
        cleaned_state_dict,
        strict=True,
    )

    model.to(DEVICE)

    model.eval()

    return model


# ============================================================
# LOAD METRICS
# ============================================================

def load_metrics():

    if not METRICS_FILE.exists():

        return None

    try:

        with open(
            METRICS_FILE,
            "r",
            encoding="utf-8",
        ) as f:

            return json.load(f)

    except Exception:

        return None


# ============================================================
# SINGLE IMAGE PREDICTION
# ============================================================

def predict_soil(
    image_path,
    model=None,
):

    image_path = Path(image_path)

    if not image_path.exists():

        raise FileNotFoundError(
            f"Soil image not found:\n{image_path}"
        )

    # --------------------------------------------------------
    # Load model only when needed
    # --------------------------------------------------------

    if model is None:

        model = load_soil_model()

    # --------------------------------------------------------
    # Open image
    # --------------------------------------------------------

    image = Image.open(
        image_path
    ).convert("RGB")

    # --------------------------------------------------------
    # Transform
    # --------------------------------------------------------

    tensor = transform(
        image
    ).unsqueeze(0)

    tensor = tensor.to(
        DEVICE
    )

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    with torch.no_grad():

        logits = model(
            tensor
        )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )[0]

    probabilities = (
        probabilities
        .cpu()
        .numpy()
    )

    # --------------------------------------------------------
    # Sort predictions
    # --------------------------------------------------------

    indices = np.argsort(
        probabilities
    )[::-1]

    predictions = []

    for index in indices:

        predictions.append(
            {
                "soil_class": SOIL_CLASSES[index],
                "probability": float(
                    probabilities[index]
                ),
            }
        )

    top_class = predictions[0][
        "soil_class"
    ]

    confidence = predictions[0][
        "probability"
    ]

    # --------------------------------------------------------
    # Confidence category
    # --------------------------------------------------------

    if confidence >= 0.70:

        confidence_level = "HIGH"

    elif confidence >= 0.50:

        confidence_level = "MEDIUM"

    else:

        confidence_level = "LOW"

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    result = {

        "image": str(
            image_path
        ),

        "soil_class": top_class,

        "confidence": float(
            confidence
        ),

        "confidence_percent": float(
            confidence * 100
        ),

        "confidence_level": confidence_level,

        "predictions": predictions,
    }

    return result


# ============================================================
# PRINT RESULT
# ============================================================

def print_result(result):

    print("\n")
    print("=" * 70)
    print("SOIL CLASSIFICATION RESULT")
    print("=" * 70)

    print(
        f"\nImage:"
        f" {result['image']}"
    )

    print(
        f"\nPredicted soil:"
        f" {result['soil_class']}"
    )

    print(
        f"Confidence:"
        f" {result['confidence_percent']:.2f}%"
    )

    print(
        f"Confidence level:"
        f" {result['confidence_level']}"
    )

    print("\nClass probabilities:")

    for prediction in result[
        "predictions"
    ]:

        print(
            f"  "
            f"{prediction['soil_class']:<12}"
            f"{prediction['probability'] * 100:>8.2f}%"
        )

    print("\n")


# ============================================================
# COMMAND LINE INTERFACE
# ============================================================

def main():

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Predict soil class from a soil image "
            "using the trained EfficientNet-B0 model."
        )
    )

    parser.add_argument(
        "image",
        help="Path to soil image",
    )

    args = parser.parse_args()

    print(
        f"Using device: {DEVICE}"
    )

    model = load_soil_model()

    result = predict_soil(
        args.image,
        model=model,
    )

    print_result(
        result
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()