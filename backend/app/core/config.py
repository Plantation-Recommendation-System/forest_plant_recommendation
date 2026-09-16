from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    app_name: str = "Plant Survival AI API"
    app_env: str = "development"
    api_v1_prefix: str = "/api/v1"
    database_url: str = ""
    ml_model_path: str = "models/recommendation_model.joblib"
    survival_file: str = "data/survival_ml_final.csv"
    cors_origins: list[str] = ["http://localhost:5173"]

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / "backend" / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def survival_file_path(self) -> Path:
        return self._resolve_project_path(self.survival_file)

    @property
    def ml_model_file_path(self) -> Path:
        return self._resolve_project_path(self.ml_model_path)

    @staticmethod
    def _resolve_project_path(value: str) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return PROJECT_ROOT / path


settings = Settings()
