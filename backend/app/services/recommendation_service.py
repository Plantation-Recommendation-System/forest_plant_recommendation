from pathlib import Path
from typing import Any

import pandas as pd

from app.core.config import settings
from app.schemas.recommendation import RecommendationRequest


class RecommendationService:
    def __init__(self, survival_file: Path):
        self.survival_file = survival_file

    def get_recommendations(
        self,
        request: RecommendationRequest,
    ) -> dict[str, Any]:
        if not self.survival_file.exists():
            raise FileNotFoundError(
                f"Survival dataset was not found: {self.survival_file}"
            )

        survival = pd.read_csv(self.survival_file, low_memory=False)
        required_columns = {"species_full", "survival_per"}
        missing = required_columns - set(survival.columns)
        if missing:
            raise ValueError(f"Survival dataset is missing columns: {sorted(missing)}")

        survival["survival_per"] = pd.to_numeric(
            survival["survival_per"], errors="coerce"
        )
        survival = survival.dropna(subset=["species_full", "survival_per"])

        summary = (
            survival.assign(
                species_normalized=survival["species_full"]
                .astype(str)
                .str.strip()
                .str.lower()
                .str.split()
                .str[:2]
                .str.join(" ")
            )
            .groupby("species_normalized", as_index=False)
            .agg(
                scientific_name=("species_full", "first"),
                observed_survival_mean=("survival_per", "mean"),
                survival_observations=("survival_per", "count"),
            )
            .sort_values(
                ["observed_survival_mean", "survival_observations"],
                ascending=[False, False],
            )
            .head(6)
            .reset_index(drop=True)
        )

        summary["recommendation_rank"] = summary.index + 1
        summary["final_score"] = summary["observed_survival_mean"] * 100
        summary["v5_survival_score"] = summary["observed_survival_mean"]
        recommendations = summary.round(4).to_dict(orient="records")

        return {
            "success": True,
            "count": len(recommendations),
            "recommendations": recommendations,
            "mode": "historical_survival_dataset_only",
        }


recommendation_service = RecommendationService(
    settings.survival_file_path
)
