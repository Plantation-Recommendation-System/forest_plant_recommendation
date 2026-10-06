from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
# =========================================================
# PATHS
# =========================================================
BACKEND_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = (
    BACKEND_DIR
    / "models"
    / "catboost_survival_final_28f.cbm"
)
METADATA_PATH = (
    BACKEND_DIR
    / "data"
    / "species_metadata.csv"
)
CATBOOST_COVERAGE_PATH = (
    BACKEND_DIR
    / "data"
    / "catboost_species_coverage.csv"
)
# =========================================================
# MANAGEMENT FEATURES
# =========================================================
MANAGEMENT_FEATURES = [
    "comp_removal",
    "shading",
    "soil_prep",
    "water_reg",
    "fertilisation",
    "protection",
]
# =========================================================
# SERVICE
# =========================================================
class SurvivalService:
    def __init__(self):
        self._model = None
        self._metadata = None
        self._catboost_coverage = None
    # -----------------------------------------------------
    # LOAD CATBOOST ONCE
    # -----------------------------------------------------
    def load_model(self) -> CatBoostRegressor:
        if self._model is None:
            if not MODEL_PATH.exists():
                raise FileNotFoundError(
                    f"CatBoost model not found: {MODEL_PATH}"
                )
            model = CatBoostRegressor()
            model.load_model(
                str(MODEL_PATH)
            )
            self._model = model
        return self._model
    # -----------------------------------------------------
    # LOAD SPECIES METADATA
    # -----------------------------------------------------
    def load_metadata(self) -> pd.DataFrame:
        if self._metadata is None:
            if not METADATA_PATH.exists():
                raise FileNotFoundError(
                    f"Species metadata not found: {METADATA_PATH}"
                )
            self._metadata = pd.read_csv(
                METADATA_PATH,
                low_memory=False,
            )
        return self._metadata.copy()
    # -----------------------------------------------------
    # LOAD CATBOOST SPECIES COVERAGE
    #
    # This file was generated from the EXACT final
    # CatBoost train/test datasets:
    #
    #   survival_train_with_climate.csv
    #   survival_test_with_climate.csv
    #
    # Possible stored statuses:
    #
    #   TRAIN_SEEN
    #   TEST_ONLY
    #
    # NO_SURVIVAL_MATCH is assigned dynamically when a
    # recommendation species does not occur in either file.
    # -----------------------------------------------------
    def load_catboost_coverage(self) -> pd.DataFrame:
        if self._catboost_coverage is None:
            if not CATBOOST_COVERAGE_PATH.exists():
                raise FileNotFoundError(
                    "CatBoost species coverage file not found: "
                    f"{CATBOOST_COVERAGE_PATH}"
                )
            df = pd.read_csv(
                CATBOOST_COVERAGE_PATH,
                low_memory=False,
            )
            required_columns = {
                "species_full",
                "train_rows",
                "test_rows",
                "catboost_coverage_status",
            }
            missing_columns = (
                required_columns
                - set(df.columns)
            )
            if missing_columns:
                raise ValueError(
                    "CatBoost coverage file is missing columns: "
                    f"{sorted(missing_columns)}"
                )
            # Normalize names only for exact matching.
            df["species_full"] = (
                df["species_full"]
                .astype(str)
                .str.strip()
            )
            # Coverage file should have one row per species.
            if df["species_full"].duplicated().any():
                duplicated = (
                    df.loc[
                        df["species_full"].duplicated(
                            keep=False
                        ),
                        "species_full",
                    ]
                    .drop_duplicates()
                    .tolist()
                )
                raise ValueError(
                    "Duplicate species found in CatBoost "
                    f"coverage file: {duplicated}"
                )
            df["train_rows"] = (
                pd.to_numeric(
                    df["train_rows"],
                    errors="raise",
                )
                .astype(int)
            )
            df["test_rows"] = (
                pd.to_numeric(
                    df["test_rows"],
                    errors="raise",
                )
                .astype(int)
            )
            allowed_statuses = {
                "TRAIN_SEEN",
                "TEST_ONLY",
            }
            invalid_statuses = (
                set(
                    df[
                        "catboost_coverage_status"
                    ]
                    .dropna()
                    .astype(str)
                )
                - allowed_statuses
            )
            if invalid_statuses:
                raise ValueError(
                    "Invalid CatBoost coverage statuses: "
                    f"{sorted(invalid_statuses)}"
                )
            self._catboost_coverage = df
        return self._catboost_coverage.copy()
    # -----------------------------------------------------
    # MANAGEMENT ACTIONS
    #
    # Frontend/backend can send:
    #
    # ["comp_removal", "protection"]
    #
    # We automatically construct all six binary fields.
    #
    # treatment = 1 whenever at least one management
    # action is active.
    # -----------------------------------------------------
    def management_features(
        self,
        actions: Optional[Iterable[str]],
    ) -> Dict[str, int]:
        actions = set(
            actions or []
        )
        invalid = (
            actions
            - set(MANAGEMENT_FEATURES)
        )
        if invalid:
            raise ValueError(
                "Unknown management actions: "
                f"{sorted(invalid)}"
            )
        result = {
            feature: int(
                feature in actions
            )
            for feature
            in MANAGEMENT_FEATURES
        }
        result["treatment"] = int(
            any(
                result[feature] == 1
                for feature
                in MANAGEMENT_FEATURES
            )
        )
        return result
    # -----------------------------------------------------
    # BUILD EXACT CATBOOST MODEL MATRIX
    # -----------------------------------------------------
    def build_input_matrix(
        self,
        candidates: pd.DataFrame,
        site_climate: Dict[str, float],
        latitude: float,
        longitude: float,
        country: str,
        province: str,
        forest_condition: str,
        management_actions: Optional[Iterable[str]],
        target_horizon_months: float,
        planting_density: Optional[float] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        model = self.load_model()
        metadata = self.load_metadata()
        coverage = self.load_catboost_coverage()
        # ================================================
        # REQUIRED CANDIDATE FIELD
        # ================================================
        if (
            "project_species_resolved"
            not in candidates.columns
        ):
            raise ValueError(
                "Candidates must contain "
                "'project_species_resolved'."
            )
        # ================================================
        # SPECIES METADATA
        # ================================================
        work = candidates.copy()
        metadata_cols = [
            "project_species_resolved",
            "genus",
            "species",
            "family",
            "w_meanWD",
        ]
        missing_metadata_columns = [
            column
            for column
            in metadata_cols
            if column not in metadata.columns
        ]
        if missing_metadata_columns:
            raise ValueError(
                "Species metadata file is missing columns: "
                f"{missing_metadata_columns}"
            )
        work = work.merge(
            metadata[
                metadata_cols
            ],
            on="project_species_resolved",
            how="left",
            validate="many_to_one",
        )
        missing_metadata = work[
            [
                "genus",
                "species",
                "family",
                "w_meanWD",
            ]
        ].isna().any(axis=1)
        if missing_metadata.any():
            missing_species = work.loc[
                missing_metadata,
                "project_species_resolved",
            ].tolist()
            raise ValueError(
                "Missing CatBoost metadata for species: "
                f"{missing_species}"
            )
        # ================================================
        # MANAGEMENT
        # ================================================
        management = (
            self.management_features(
                management_actions
            )
        )
        # ================================================
        # PLANTING DENSITY
        # ================================================
        if planting_density is None:
            planting_density_value = np.nan
            planting_density_missing = 1
        else:
            planting_density_value = float(
                planting_density
            )
            if planting_density_value <= 0:
                raise ValueError(
                    "Planting density must be positive."
                )
            planting_density_missing = 0
        # ================================================
        # BUILD ONE ROW PER CANDIDATE SPECIES
        # ================================================
        rows = []
        for _, candidate in work.iterrows():
            row = {
                "country":
                    str(country),
                "province":
                    str(province),
                "forest_condition":
                    str(forest_condition),
                "comp_removal":
                    management[
                        "comp_removal"
                    ],
                "shading":
                    management[
                        "shading"
                    ],
                "soil_prep":
                    management[
                        "soil_prep"
                    ],
                "water_reg":
                    management[
                        "water_reg"
                    ],
                "fertilisation":
                    management[
                        "fertilisation"
                    ],
                "protection":
                    management[
                        "protection"
                    ],
                "planting_density":
                    planting_density_value,
                # Standardized single-species comparison
                # scenario used during recommendation.
                "planting_sp_no":
                    1.0,
                "species_full":
                    str(
                        candidate[
                            "project_species_resolved"
                        ]
                    ).strip(),
                "genus":
                    str(
                        candidate[
                            "genus"
                        ]
                    ),
                "species":
                    str(
                        candidate[
                            "species"
                        ]
                    ),
                "family":
                    str(
                        candidate[
                            "family"
                        ]
                    ),
                "lat_dec":
                    float(latitude),
                "lon_dec":
                    float(longitude),
                "w_meanWD":
                    float(
                        candidate[
                            "w_meanWD"
                        ]
                    ),
                "treatment":
                    management[
                        "treatment"
                    ],
                "planting_density_missing":
                    planting_density_missing,
                "target_horizon_months":
                    float(
                        target_horizon_months
                    ),
            }
            # ============================================
            # ADD SITE CLIMATE FEATURES
            # ============================================
            for (
                climate_name,
                climate_value,
            ) in site_climate.items():
                row[
                    climate_name
                ] = float(
                    climate_value
                )
            rows.append(
                row
            )
        matrix = pd.DataFrame(
            rows
        )
        # ================================================
        # EXACT MODEL FEATURE ORDER
        # ================================================
        model_features = list(
            model.feature_names_
        )
        missing_features = [
            feature
            for feature
            in model_features
            if feature not in matrix.columns
        ]
        if missing_features:
            raise ValueError(
                "Missing CatBoost input features: "
                f"{missing_features}"
            )
        # IMPORTANT:
        # The saved model itself determines the final
        # feature order.
        matrix = matrix[
            model_features
        ].copy()
        # ================================================
        # CATBOOST CATEGORICAL FIELDS
        # ================================================
        cat_indices = list(
            model.get_cat_feature_indices()
        )
        categorical_features = [
            model_features[index]
            for index
            in cat_indices
        ]
        for column in categorical_features:
            if matrix[
                column
            ].isna().any():
                raise ValueError(
                    "Categorical feature contains "
                    f"missing values: {column}"
                )
            matrix[
                column
            ] = (
                matrix[
                    column
                ]
                .astype(str)
            )
        # ================================================
        # ATTACH CATBOOST COVERAGE INFORMATION
        #
        # This information is SUPPORT METADATA ONLY.
        # It is NOT passed to CatBoost as a predictor.
        # ================================================
        support = work[
            [
                "project_species_resolved"
            ]
        ].copy()
        support[
            "species_full"
        ] = (
            support[
                "project_species_resolved"
            ]
            .astype(str)
            .str.strip()
        )
        coverage_for_merge = coverage[
            [
                "species_full",
                "train_rows",
                "test_rows",
                "catboost_coverage_status",
            ]
        ].copy()
        support = support.merge(
            coverage_for_merge,
            on="species_full",
            how="left",
            validate="many_to_one",
        )
        # Species absent from BOTH exact CatBoost train
        # and test datasets receive NO_SURVIVAL_MATCH.
        support[
            "catboost_coverage_status"
        ] = (
            support[
                "catboost_coverage_status"
            ]
            .fillna(
                "NO_SURVIVAL_MATCH"
            )
        )
        support[
            "train_rows"
        ] = (
            support[
                "train_rows"
            ]
            .fillna(0)
            .astype(int)
        )
        support[
            "test_rows"
        ] = (
            support[
                "test_rows"
            ]
            .fillna(0)
            .astype(int)
        )
        # Keep this boolean for compatibility with any
        # existing code that still expects it.
        support[
            "seen_in_catboost_training"
        ] = (
            support[
                "catboost_coverage_status"
            ]
            == "TRAIN_SEEN"
        )
        return matrix, support
    # -----------------------------------------------------
    # PREDICT SURVIVAL
    # -----------------------------------------------------
    def predict_candidates(
        self,
        candidates: pd.DataFrame,
        site_climate: Dict[str, float],
        latitude: float,
        longitude: float,
        country: str,
        province: str,
        forest_condition: str,
        management_actions: Optional[Iterable[str]],
        target_horizon_months: float,
        planting_density: Optional[float] = None,
    ) -> pd.DataFrame:
        model = self.load_model()
        matrix, support = (
            self.build_input_matrix(
                candidates=candidates,
                site_climate=site_climate,
                latitude=latitude,
                longitude=longitude,
                country=country,
                province=province,
                forest_condition=forest_condition,
                management_actions=management_actions,
                target_horizon_months=target_horizon_months,
                planting_density=planting_density,
            )
        )
        # ================================================
        # CATBOOST PREDICTION
        #
        # IMPORTANT:
        # The verified frozen .cbm model is used directly.
        # No retraining or re-ranking occurs here.
        # ================================================
        predictions = model.predict(
            matrix
        )
        result = candidates.copy().reset_index(
            drop=True
        )
        support = support.reset_index(
            drop=True
        )
        # ================================================
        # PREDICTION OUTPUT
        # ================================================
        result[
            "predicted_survival_pct_raw"
        ] = predictions
        result[
            "predicted_survival_pct"
        ] = (
            result[
                "predicted_survival_pct_raw"
            ]
            .round(1)
        )
        # Do NOT silently clip predictions.
        result[
            "prediction_range_status"
        ] = np.where(
            result[
                "predicted_survival_pct_raw"
            ]
            < 0,
            "below_0",
            np.where(
                result[
                    "predicted_survival_pct_raw"
                ]
                > 100,
                "above_100",
                "within_0_100",
            ),
        )
        # ================================================
        # CATBOOST COVERAGE / EVIDENCE
        # ================================================
        result[
            "catboost_coverage_status"
        ] = support[
            "catboost_coverage_status"
        ].values
        result[
            "catboost_train_rows"
        ] = support[
            "train_rows"
        ].values
        result[
            "catboost_test_rows"
        ] = support[
            "test_rows"
        ].values
        # Backward-compatible boolean.
        result[
            "seen_in_catboost_training"
        ] = support[
            "seen_in_catboost_training"
        ].values
        # Human/backend-readable supporting evidence label.
        evidence_map = {
            "TRAIN_SEEN":
                "supporting_estimate_train_seen",
            "TEST_ONLY":
                "supporting_estimate_test_only",
            "NO_SURVIVAL_MATCH":
                "supporting_estimate_no_survival_match",
        }
        result[
            "catboost_evidence_status"
        ] = (
            result[
                "catboost_coverage_status"
            ]
            .map(
                evidence_map
            )
        )
        # ================================================
        # INSPECTION RANK ONLY
        #
        # CatBoost does NOT replace the environmental
        # climate/soil ranking.
        # ================================================
        result[
            "catboost_survival_rank"
        ] = (
            result[
                "predicted_survival_pct_raw"
            ]
            .rank(
                ascending=False,
                method="min",
            )
            .astype("Int64")
        )
        return result
survival_service = SurvivalService()
