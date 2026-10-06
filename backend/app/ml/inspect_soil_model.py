from pathlib import Path

import torch


BACKEND_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    BACKEND_DIR
    / "models"
    / "efficientnet_b0_soil_final.pth"
)


checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=False,
)


print("=" * 60)
print("SOIL MODEL METADATA")
print("=" * 60)


print(
    "Epoch:",
    checkpoint.get("epoch")
)

print(
    "Model name:",
    checkpoint.get("model_name")
)

print(
    "Training stage:",
    checkpoint.get("stage")
)

print(
    "Validation macro F1:",
    checkpoint.get("val_macro_f1")
)


print("\nClasses:")

print(
    checkpoint.get("classes")
)


print("\nClass to index:")

print(
    checkpoint.get("class_to_idx")
)


state_dict = checkpoint[
    "model_state_dict"
]


print("\nClassifier-related tensors:")

for key, value in state_dict.items():

    if "classifier" in key:

        print(
            key,
            tuple(value.shape)
        )


print(
    "\n✅ SOIL MODEL METADATA INSPECTION COMPLETE"
)