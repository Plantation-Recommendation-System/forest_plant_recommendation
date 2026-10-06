from io import BytesIO
from pathlib import Path
from typing import Dict, Union

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


# =========================================================
# PATH
# =========================================================

BACKEND_DIR = Path(__file__).resolve().parents[2]

SOIL_MODEL_PATH = (
    BACKEND_DIR
    / "models"
    / "efficientnet_b0_soil_final.pth"
)


# =========================================================
# INFERENCE TRANSFORM
#
# Same evaluation transform used during training:
#
# Resize 256
# CenterCrop 224
# ToTensor
# ImageNet normalization
# =========================================================

SOIL_INFERENCE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(256),

        transforms.CenterCrop(224),

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


# =========================================================
# SOIL IMAGE SERVICE
# =========================================================

class SoilImageService:

    def __init__(
        self,
        model_path: Path = SOIL_MODEL_PATH,
    ):

        self.model_path = model_path

        self.device = torch.device(
            "cpu"
        )

        self.model = None

        self.classes = None

        self.class_to_idx = None


    # -----------------------------------------------------
    # LOAD MODEL ONCE
    # -----------------------------------------------------

    def load_model(
        self,
    ):

        if self.model is not None:
            return self.model


        if not self.model_path.exists():

            raise FileNotFoundError(
                "Soil model not found: "
                f"{self.model_path}"
            )


        checkpoint = torch.load(
            self.model_path,
            map_location=self.device,
            weights_only=False,
        )


        self.classes = checkpoint[
            "classes"
        ]

        self.class_to_idx = checkpoint[
            "class_to_idx"
        ]


        # -----------------------------------------------
        # Recreate EfficientNet-B0 architecture
        # -----------------------------------------------

        model = models.efficientnet_b0(
            weights=None
        )


        in_features = (
            model
            .classifier[1]
            .in_features
        )


        model.classifier[1] = nn.Linear(
            in_features,
            len(self.classes),
        )


        # -----------------------------------------------
        # Load our trained weights
        # -----------------------------------------------

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )


        model.to(
            self.device
        )

        model.eval()


        self.model = model

        return self.model


    # -----------------------------------------------------
    # PREPARE IMAGE
    # -----------------------------------------------------

    def prepare_image(
        self,
        image: Image.Image,
    ) -> torch.Tensor:

        if image.mode != "RGB":

            image = image.convert(
                "RGB"
            )


        tensor = (
            SOIL_INFERENCE_TRANSFORM(
                image
            )
        )


        tensor = tensor.unsqueeze(
            0
        )


        return tensor.to(
            self.device
        )


    # -----------------------------------------------------
    # PREDICT FROM PIL IMAGE
    # -----------------------------------------------------

    def predict_image(
        self,
        image: Image.Image,
    ) -> Dict:

        model = self.load_model()


        tensor = self.prepare_image(
            image
        )


        with torch.no_grad():

            logits = model(
                tensor
            )

            probabilities = (
                torch.softmax(
                    logits,
                    dim=1,
                )
                .cpu()
                .numpy()[0]
            )


        probability_map = {

            class_name.lower():
                float(
                    probabilities[index]
                )

            for index, class_name
            in enumerate(
                self.classes
            )
        }


        predicted_index = int(
            probabilities.argmax()
        )


        predicted_class = (
            self.classes[
                predicted_index
            ]
        )


        confidence = float(
            probabilities[
                predicted_index
            ]
        )


        return {

            "predicted_class":
                predicted_class,

            "confidence":
                confidence,

            "probabilities":
                probability_map,

            "classes":
                self.classes,
        }


    # -----------------------------------------------------
    # PREDICT FROM RAW UPLOADED BYTES
    #
    # This will be useful with FastAPI UploadFile.
    # -----------------------------------------------------

    def predict_bytes(
        self,
        image_bytes: bytes,
    ) -> Dict:

        try:

            image = Image.open(
                BytesIO(
                    image_bytes
                )
            )

        except Exception as error:

            raise ValueError(
                "Uploaded file is not a valid image."
            ) from error


        return self.predict_image(
            image
        )


    # -----------------------------------------------------
    # PREDICT FROM LOCAL FILE
    #
    # Useful for development/testing.
    # -----------------------------------------------------

    def predict_file(
        self,
        image_path: Union[
            str,
            Path,
        ],
    ) -> Dict:

        image_path = Path(
            image_path
        )


        if not image_path.exists():

            raise FileNotFoundError(
                f"Soil image not found: "
                f"{image_path}"
            )


        image = Image.open(
            image_path
        )


        return self.predict_image(
            image
        )


# =========================================================
# SHARED SERVICE INSTANCE
# =========================================================

soil_image_service = SoilImageService()