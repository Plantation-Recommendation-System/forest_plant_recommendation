from pathlib import Path
from typing import Any

import joblib


class JoblibModel:
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self._model: Any = None

    def load(self) -> Any:
        if self._model is None:
            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"ML model was not found: {self.model_path}"
                )
            self._model = joblib.load(self.model_path)
        return self._model

    def predict(self, features: Any) -> Any:
        return self.load().predict(features)
