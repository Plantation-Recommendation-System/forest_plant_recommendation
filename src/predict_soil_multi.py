from pathlib import Path
import sys
import json

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = Path("models/soil/soil_efficientnet_b0_final.pt")
METRICS_PATH = Path("models/soil/soil_model_metrics.json")

CLASSES = [
    "Alluvial",
    "Arid",
    "Black",
    "Laterite",
    "Mountain",
    "Red",
    "Yellow",
]

IMAGE_SIZE = 224

HIGH_THRESHOLD = 0.70
MEDIUM_THRESHOLD = 0.50


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# MODEL
# ============================================================

def build_model():
    model = models.efficientnet_b0(weights=None)

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        len(CLASSES)
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False
    )

    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Remove DataParallel prefix if present
    cleaned_state_dict = {}

    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[7:]
        cleaned_state_dict[key] = value

    model.load_state_dict(cleaned_state_dict)

    model.to(device)
    model.eval()

    return model


# ============================================================
# IMAGE TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# ============================================================
# SINGLE IMAGE
# ============================================================

def predict_single(model, image_path):

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        probabilities = torch.softmax(output, dim=1)[0]

    return probabilities.cpu()


# ============================================================
# MULTI IMAGE
# ============================================================

def predict_multiple(image_paths):

    model = build_model()

    all_probabilities = []

    print()
    print("Using device:", device)
    print()

    for image_path in image_paths:

        print(f"Processing: {image_path}")

        probabilities = predict_single(
            model,
            image_path
        )

        all_probabilities.append(probabilities)

    # Stack:
    # [number_of_images, number_of_classes]
    probability_matrix = torch.stack(all_probabilities)

    # Average probability across images
    mean_probabilities = probability_matrix.mean(dim=0)

    # Final prediction
    predicted_index = torch.argmax(mean_probabilities).item()

    predicted_class = CLASSES[predicted_index]
    confidence = mean_probabilities[predicted_index].item()

    if confidence >= HIGH_THRESHOLD:
        confidence_level = "HIGH"
    elif confidence >= MEDIUM_THRESHOLD:
        confidence_level = "MEDIUM"
    else:
        confidence_level = "LOW"

    # ========================================================
    # OUTPUT
    # ========================================================

    print()
    print("=" * 70)
    print("MULTI-IMAGE SOIL CLASSIFICATION RESULT")
    print("=" * 70)

    print()
    print(f"Images used: {len(image_paths)}")
    print(f"Predicted soil: {predicted_class}")
    print(f"Confidence: {confidence * 100:.2f}%")
    print(f"Confidence level: {confidence_level}")

    print()
    print("Average class probabilities:")

    sorted_results = sorted(
        zip(CLASSES, mean_probabilities.tolist()),
        key=lambda x: x[1],
        reverse=True
    )

    for class_name, probability in sorted_results:
        print(
            f"  {class_name:<14} "
            f"{probability * 100:.2f}%"
        )

    print()

    # Show individual predictions
    print("-" * 70)
    print("INDIVIDUAL IMAGE PREDICTIONS")
    print("-" * 70)

    for image_path, probabilities in zip(
        image_paths,
        all_probabilities
    ):

        idx = torch.argmax(probabilities).item()

        print(
            f"{Path(image_path).name}: "
            f"{CLASSES[idx]} "
            f"({probabilities[idx].item() * 100:.2f}%)"
        )

    print()

    # ========================================================
    # RECOMMENDATION ENGINE FLAG
    # ========================================================

    if confidence >= HIGH_THRESHOLD:

        soil_usable = True

        print("SOIL SIGNAL: USABLE")
        print(
            "The soil prediction is confident enough "
            "to enter the recommendation pipeline."
        )

    else:

        soil_usable = False

        print("SOIL SIGNAL: NOT USABLE")
        print(
            "Confidence is too low. "
            "The recommendation engine should ignore "
            "the soil signal."
        )

    print("=" * 70)

    return {
        "predicted_soil": predicted_class,
        "confidence": confidence,
        "confidence_percent": confidence * 100,
        "confidence_level": confidence_level,
        "soil_signal_usable": soil_usable,
        "num_images": len(image_paths),
        "probabilities": {
            class_name: probability
            for class_name, probability in sorted_results
        }
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) < 3:
        print()
        print("Usage:")
        print(
            "python src/predict_soil_multi.py "
            "data/soil1.jpg data/soil2.jpg data/soil3.jpg"
        )
        print()
        sys.exit(1)

    image_paths = [
        Path(arg)
        for arg in sys.argv[1:]
    ]

    # Check files
    for path in image_paths:

        if not path.exists():

            print(
                f"ERROR: Image not found: {path}"
            )

            sys.exit(1)

    if not MODEL_PATH.exists():

        print(
            f"ERROR: Model not found: {MODEL_PATH}"
        )

        sys.exit(1)

    result = predict_multiple(image_paths)

    # Optional JSON output
    output_path = Path(
        "outputs/soil_multi_prediction.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            result,
            f,
            indent=2
        )

    print()
    print(
        f"Saved result to: {output_path}"
    )