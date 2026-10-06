from pathlib import Path
from typing import Dict

import pandas as pd

from soiltexture import getTexture


# =========================================================
# PATH
# =========================================================

BACKEND_DIR = Path(__file__).resolve().parents[2]

SOIL_OBSERVATION_PATH = (
    BACKEND_DIR
    / "data"
    / "species_soil_texture_observations.csv"
)


# =========================================================
# PROJECT 3-CLASS HARMONIZATION
#
# USDA texture classes are collapsed into the same
# Sand / Loam / Clay labels used by the image model.
#
# This is project harmonization, not an official USDA
# 3-class soil classification.
# =========================================================

SAND_CLASSES = {
    "sand",
    "loamy sand",
}

LOAM_CLASSES = {
    "sandy loam",
    "loam",
    "silt",
    "silt loam",
    "clay loam",
    "sandy clay loam",
    "silty clay loam",
}

CLAY_CLASSES = {
    "clay",
    "sandy clay",
    "silty clay",
}


def collapse_texture(
    usda_texture: str,
) -> str:

    value = str(
        usda_texture
    ).strip().lower()

    if value in SAND_CLASSES:
        return "sand"

    if value in LOAM_CLASSES:
        return "loam"

    if value in CLAY_CLASSES:
        return "clay"

    return "unknown"


# =========================================================
# SERVICE
# =========================================================

class SoilCompatibilityService:

    def __init__(
        self,
        observation_path: Path = SOIL_OBSERVATION_PATH,
    ):

        self.observation_path = observation_path

        self._species_distribution = None


    # -----------------------------------------------------
    # BUILD SPECIES SOIL DISTRIBUTIONS
    # -----------------------------------------------------

    def load_species_distributions(
        self,
    ) -> pd.DataFrame:

        if self._species_distribution is not None:

            return (
                self._species_distribution
                .copy()
            )


        if not self.observation_path.exists():

            raise FileNotFoundError(
                "Soil observation file not found: "
                f"{self.observation_path}"
            )


        df = pd.read_csv(
            self.observation_path,
            low_memory=False,
        )


        required = [
            "project_species_resolved",
            "soilgrids_sand_pct",
            "soilgrids_clay_pct",
        ]


        missing = [
            col
            for col in required
            if col not in df.columns
        ]


        if missing:

            raise ValueError(
                f"Missing soil columns: {missing}"
            )


        work = df[
            required
        ].copy()


        work[
            "soilgrids_sand_pct"
        ] = pd.to_numeric(
            work[
                "soilgrids_sand_pct"
            ],
            errors="coerce",
        )


        work[
            "soilgrids_clay_pct"
        ] = pd.to_numeric(
            work[
                "soilgrids_clay_pct"
            ],
            errors="coerce",
        )


        work = work.dropna(
            subset=[
                "soilgrids_sand_pct",
                "soilgrids_clay_pct",
            ]
        )


        # -----------------------------------------------
        # USDA TEXTURE CLASS
        # soiltexture expects sand + clay percentages
        # -----------------------------------------------

        work[
            "usda_texture"
        ] = work.apply(

            lambda row:
                getTexture(
                    row[
                        "soilgrids_sand_pct"
                    ],
                    row[
                        "soilgrids_clay_pct"
                    ],
                ),

            axis=1,
        )


        work[
            "broad_texture"
        ] = (
            work[
                "usda_texture"
            ]
            .apply(
                collapse_texture
            )
        )


        work = work[
            work[
                "broad_texture"
            ]
            !=
            "unknown"
        ].copy()


        # -----------------------------------------------
        # COUNT EACH CLASS PER SPECIES
        # -----------------------------------------------

        counts = (

            work
            .groupby(
                [
                    "project_species_resolved",
                    "broad_texture",
                ]
            )
            .size()
            .unstack(
                fill_value=0
            )
        )


        for texture in [
            "sand",
            "loam",
            "clay",
        ]:

            if texture not in counts.columns:
                counts[
                    texture
                ] = 0


        counts = counts[
            [
                "sand",
                "loam",
                "clay",
            ]
        ]


        counts[
            "total_texture_observations"
        ] = (
            counts[
                [
                    "sand",
                    "loam",
                    "clay",
                ]
            ]
            .sum(
                axis=1
            )
        )


        # -----------------------------------------------
        # FRACTIONS
        # -----------------------------------------------

        for texture in [
            "sand",
            "loam",
            "clay",
        ]:

            counts[
                f"{texture}_fraction"
            ] = (

                counts[
                    texture
                ]

                /

                counts[
                    "total_texture_observations"
                ]
            )


        counts = (
            counts
            .reset_index()
        )


        self._species_distribution = (
            counts
        )


        return counts.copy()


    # -----------------------------------------------------
    # VALIDATE USER IMAGE PROBABILITIES
    # -----------------------------------------------------

    def validate_user_probabilities(
        self,
        probabilities: Dict[str, float],
    ) -> Dict[str, float]:

        required = {
            "sand",
            "loam",
            "clay",
        }


        missing = (
            required
            -
            set(
                probabilities.keys()
            )
        )


        if missing:

            raise ValueError(
                "Missing soil probabilities: "
                f"{sorted(missing)}"
            )


        cleaned = {
            key:
                float(
                    probabilities[
                        key
                    ]
                )

            for key
            in required
        }


        if any(
            value < 0
            for value in cleaned.values()
        ):

            raise ValueError(
                "Soil probabilities cannot be negative."
            )


        total = sum(
            cleaned.values()
        )


        if total <= 0:

            raise ValueError(
                "Soil probability total must be greater than zero."
            )


        # Normalize for safety
        cleaned = {
            key:
                value / total

            for key, value
            in cleaned.items()
        }


        return cleaned


    # -----------------------------------------------------
    # MATCH USER SOIL TO SPECIES
    #
    # overlap =
    # min(user_sand, species_sand)
    # +
    # min(user_loam, species_loam)
    # +
    # min(user_clay, species_clay)
    #
    # Range: 0 to 1
    # -----------------------------------------------------

    def match_candidates(
        self,
        candidates: pd.DataFrame,
        user_probabilities: Dict[str, float],
    ) -> pd.DataFrame:

        probs = (
            self
            .validate_user_probabilities(
                user_probabilities
            )
        )


        distributions = (
            self
            .load_species_distributions()
        )


        if (
            "project_species_resolved"
            not in
            candidates.columns
        ):

            raise ValueError(
                "Candidates must contain "
                "'project_species_resolved'."
            )


        result = candidates.merge(
            distributions,
            on="project_species_resolved",
            how="left",
        )


        def compute_overlap(
            row,
        ):

            required_cols = [
                "sand_fraction",
                "loam_fraction",
                "clay_fraction",
            ]


            if any(
                pd.isna(
                    row[col]
                )
                for col
                in required_cols
            ):

                return None


            return (

                min(
                    probs[
                        "sand"
                    ],
                    row[
                        "sand_fraction"
                    ],
                )

                +

                min(
                    probs[
                        "loam"
                    ],
                    row[
                        "loam_fraction"
                    ],
                )

                +

                min(
                    probs[
                        "clay"
                    ],
                    row[
                        "clay_fraction"
                    ],
                )
            )


        result[
            "soil_texture_overlap"
        ] = result.apply(
            compute_overlap,
            axis=1,
        )


        result[
            "soil_texture_match_pct"
        ] = (

            result[
                "soil_texture_overlap"
            ]

            * 100
        )


        # -----------------------------------------------
        # Operational descriptive labels
        # Not biological thresholds.
        # -----------------------------------------------

        def match_label(
            overlap,
        ):

            if pd.isna(
                overlap
            ):
                return "insufficient_soil_evidence"

            if overlap >= 0.80:
                return "strong_texture_match"

            if overlap >= 0.60:
                return "moderate_texture_match"

            return "weak_texture_match"


        result[
            "soil_texture_match_label"
        ] = (

            result[
                "soil_texture_overlap"
            ]
            .apply(
                match_label
            )
        )


        return result


    # -----------------------------------------------------
    # QUESTIONNAIRE CROSS-CHECK
    #
    # Only wet-soil behavior maps directly to our
    # texture classes.
    #
    # Drainage and usual moisture stay as separate
    # contextual information and do NOT alter the
    # texture score.
    # -----------------------------------------------------

    def questionnaire_consistency(
        self,
        predicted_class: str,
        wet_behavior: str,
    ) -> str:

        behavior_to_texture = {

            "loose_gritty":
                "sand",

            "weak_ball":
                "loam",

            "sticky":
                "clay",
        }


        questionnaire_texture = (
            behavior_to_texture.get(
                wet_behavior
            )
        )


        if questionnaire_texture is None:

            return "questionnaire_unknown"


        predicted = (
            str(
                predicted_class
            )
            .strip()
            .lower()
        )


        if (
            predicted
            ==
            questionnaire_texture
        ):

            return "consistent"


        return "inconsistent"


soil_compatibility_service = (
    SoilCompatibilityService()
)