from typing import Any

from app.core.config import settings
from app.ml.model_loader import JoblibModel


recommendation_model = JoblibModel(settings.ml_model_file_path)


def predict(features: Any) -> Any:
    """Run the future live scikit-learn/joblib inference model."""
    return recommendation_model.predict(features)
