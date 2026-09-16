"""
V7 FINAL RECOMMENDER

Purpose
-------
Combine regional candidate discovery with the existing
survival/evidence recommendation architecture.

IMPORTANT
---------
This system distinguishes:

A_ML_SUPPORTED
    Regional evidence + sufficient survival evidence.
    Eligible for the main recommendation ranking.

B_REGIONAL_ONLY
    Regional evidence but insufficient survival evidence.
    NOT assigned a fabricated survival score.

C_LOCAL_EXTENDED
    Local/extended evidence.
    Not treated as ML-supported unless survival evidence exists.

No missing survival score is replaced with 50.

The final recommendation is an evidence-aware ranking,
NOT a calibrated probability of survival.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================
ROOT = Path(r"C:\Users\Dhrub Karn\Documents\Plant_Survival_Ai")

DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"


REGIONAL_FILE = (
    OUTPUTS / "v7_regional_candidates.csv"
)

# Existing V5 candidate prediction file.
# This is useful for the Pune candidates that have already
# been passed through the V5 model.
V5_EXISTING_FILE = (
    OUTPUTS / "candidate_v5_predictions.csv"
)

EVIDENCE_FILE = (
    OUTPUTS / "candidate_evidence_scores.csv"
)

SOIL_FILE = (
    OUTPUTS / "candidate_soil_compatibility.csv"
)

FINAL_FILE = (
    OUTPUTS / "v7_final_ranking.csv"
)

MAIN_POOL_FILE = (
    OUTPUTS / "v7_main_recommendation_pool.csv"
)

REGIONAL_ONLY_FILE = (
    OUTPUTS / "v7_regional_only_candidates.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

# Main score weights.
#
# These are deliberately the same conceptual weighting
# structure used in the existing V6 prototype:
#
# survival 40
# evidence 15
# soil 15
# climate 10
# pollution 10
# user 10
#
# For now, unavailable components are renormalized.
#
# This means missing climate data does NOT silently become
# zero and unfairly punish every species.
WEIGHTS = {
    "survival": 0.40,
    "evidence": 0.15,
    "soil": 0.15,
    "climate": 0.10,
    "pollution": 0.10,
    "user": 0.10,
}


# ============================================================
# HELPERS
# ============================================================

def find_column(
    df,
    candidates,
):
    """
    Return the first available column.
    """

    for column in candidates:

        if column in df.columns:

            return column

    return None


def numeric_series(
    df,
    column,
):
    """
    Convert a column to numeric safely.
    """

    if column is None:

        return pd.Series(
            np.nan,
            index=df.index,
            dtype=float,
        )

    return pd.to_numeric(
        df[column],
        errors="coerce",
    )


def minmax_0_100(
    series,
):
    """
    Normalize a numeric series to 0-100.

    If all values are identical, return 50 rather
    than creating an artificial ordering.
    """

    values = pd.to_numeric(
        series,
        errors="coerce",
    )

    valid = values.dropna()

    if valid.empty:

        return pd.Series(
            np.nan,
            index=series.index,
            dtype=float,
        )

    minimum = valid.min()
    maximum = valid.max()

    if maximum == minimum:

        result = pd.Series(
            np.nan,
            index=series.index,
            dtype=float,
        )

        result.loc[
            values.notna()
        ] = 50.0

        return result

    return (
        (values - minimum)
        / (maximum - minimum)
        * 100
    )


# ============================================================
# REGIONAL EVIDENCE SCORE
# ============================================================

def calculate_regional_evidence(
    df,
):
    """
    Convert regional evidence into a bounded evidence score.

    This is NOT a survival score.

    Local evidence is weighted more strongly than extended
    evidence.

    Components:
        within 10 km
        within 25 km
        within 50 km
        unique coordinates
        dataset diversity
    """

    within10 = numeric_series(
        df,
        "within_10km",
    )

    within25 = numeric_series(
        df,
        "within_25km",
    )

    within50 = numeric_series(
        df,
        "within_50km",
    )

    coordinates = numeric_series(
        df,
        "unique_coordinates",
    )

    datasets = numeric_series(
        df,
        "unique_datasets",
    )

    # --------------------------------------------------------
    # Presence indicators
    # --------------------------------------------------------

    local = (
        within10
        .fillna(0)
        > 0
    ).astype(float)

    regional = (
        within25
        .fillna(0)
        > 0
    ).astype(float)

    extended = (
        within50
        .fillna(0)
        > 0
    ).astype(float)

    # --------------------------------------------------------
    # Diversity
    # --------------------------------------------------------

    coordinate_score = minmax_0_100(
        coordinates
    )

    dataset_score = minmax_0_100(
        datasets
    )

    # --------------------------------------------------------
    # Weighted evidence
    # --------------------------------------------------------

    score = (
        local * 50
        +
        regional * 25
        +
        extended * 10
        +
        coordinate_score.fillna(0) * 0.10
        +
        dataset_score.fillna(0) * 0.15
    )

    return score.clip(
        lower=0,
        upper=100,
    )


# ============================================================
# LOAD EXISTING V5 PREDICTIONS
# ============================================================

def load_existing_v5():
    """
    Load previously generated V5 predictions if available.

    IMPORTANT:
    Only existing genuine V5 predictions are used.

    We do NOT create missing V5 predictions here.
    """

    if not V5_EXISTING_FILE.exists():

        print(
            "No existing V5 candidate prediction file."
        )

        return pd.DataFrame()

    df = pd.read_csv(
        V5_EXISTING_FILE,
        low_memory=False,
    )

    species_column = find_column(
        df,
        [
            "species_normalized",
            "species_full",
            "species",
        ],
    )

    prediction_column = find_column(
        df,
        [
            "v5_predicted_survival",
            "predicted_survival",
            "v5_prediction",
        ],
    )

    if (
        species_column is None
        or prediction_column is None
    ):

        print(
            "Existing V5 file does not contain "
            "recognized species/prediction columns."
        )

        return pd.DataFrame()

    result = df[
        [
            species_column,
            prediction_column,
        ]
    ].copy()

    result.columns = [
        "species_normalized",
        "v5_survival_score",
    ]

    result[
        "v5_survival_score"
    ] = pd.to_numeric(
        result[
            "v5_survival_score"
        ],
        errors="coerce",
    )

    return (
        result
        .dropna(
            subset=[
                "species_normalized"
            ]
        )
        .drop_duplicates(
            subset=[
                "species_normalized"
            ]
        )
    )


# ============================================================
# LOAD OTHER EXISTING SCORES
# ============================================================

def load_optional_scores(
    path,
    possible_species,
    possible_score,
    output_name,
):

    if not path.exists():

        print(
            f"{path.name}: not found."
        )

        return pd.DataFrame()

    df = pd.read_csv(
        path,
        low_memory=False,
    )

    species_column = find_column(
        df,
        possible_species,
    )

    score_column = find_column(
        df,
        possible_score,
    )

    if (
        species_column is None
        or score_column is None
    ):

        print(
            f"{path.name}: required columns "
            f"not found."
        )

        return pd.DataFrame()

    result = df[
        [
            species_column,
            score_column,
        ]
    ].copy()

    result.columns = [
        "species_normalized",
        output_name,
    ]

    result[
        output_name
    ] = pd.to_numeric(
        result[
            output_name
        ],
        errors="coerce",
    )

    return (
        result
        .dropna(
            subset=[
                "species_normalized"
            ]
        )
        .drop_duplicates(
            subset=[
                "species_normalized"
            ]
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("V7 FINAL RECOMMENDER")
    print("=" * 70)

    # ========================================================
    # 1. LOAD REGIONAL CANDIDATES
    # ========================================================

    if not REGIONAL_FILE.exists():

        raise FileNotFoundError(
            f"Missing regional candidate file:\n"
            f"{REGIONAL_FILE}\n\n"
            "Run v7_regional_candidates.py first."
        )

    regional = pd.read_csv(
        REGIONAL_FILE,
        low_memory=False,
    )

    if regional.empty:

        raise RuntimeError(
            "Regional candidate file is empty."
        )

    print()
    print(
        f"Regional candidates loaded: "
        f"{len(regional)}"
    )

    # ========================================================
    # 2. REGIONAL EVIDENCE
    # ========================================================

    regional[
        "regional_evidence_score"
    ] = calculate_regional_evidence(
        regional
    )

    # ========================================================
    # 3. EXISTING V5
    # ========================================================

    v5 = load_existing_v5()

    if not v5.empty:

        print(
            f"Existing V5 predictions: "
            f"{len(v5)}"
        )

        regional = regional.merge(
            v5,
            on="species_normalized",
            how="left",
        )

    else:

        regional[
            "v5_survival_score"
        ] = np.nan

    # ========================================================
    # 4. EVIDENCE SCORE
    # ========================================================

    evidence = load_optional_scores(
        EVIDENCE_FILE,
        [
            "species_normalized",
            "species",
            "species_full",
        ],
        [
            "evidence_score",
            "candidate_evidence_score",
            "historical_evidence_score",
        ],
        "historical_evidence_score",
    )

    if not evidence.empty:

        regional = regional.merge(
            evidence,
            on="species_normalized",
            how="left",
        )

    else:

        regional[
            "historical_evidence_score"
        ] = np.nan

    # ========================================================
    # 5. SOIL
    # ========================================================

    soil = load_optional_scores(
        SOIL_FILE,
        [
            "species_normalized",
            "species",
            "species_full",
        ],
        [
            "soil_compatibility_score",
            "soil_score",
            "compatibility_score",
        ],
        "soil_score",
    )

    if not soil.empty:

        regional = regional.merge(
            soil,
            on="species_normalized",
            how="left",
        )

    else:

        regional[
            "soil_score"
        ] = np.nan

    # ========================================================
    # 6. CLIMATE / POLLUTION / USER
    # ========================================================
    #
    # These are intentionally left as optional components.
    #
    # If a future module supplies them, this final engine
    # can consume the scores.
    #
    # We do NOT invent values.
    #

    for column in [
        "climate_score",
        "pollution_score",
        "user_fit_score",
    ]:

        if column not in regional.columns:

            regional[column] = np.nan

    # ========================================================
    # 7. IDENTIFY MAIN POOL
    # ========================================================

    regional[
        "ml_supported"
    ] = (
        regional[
            "candidate_tier"
        ]
        == "A_ML_SUPPORTED"
    )

    main_pool = regional[
        regional[
            "ml_supported"
        ]
        &
        regional[
            "v5_survival_score"
        ].notna()
    ].copy()

    regional_only = regional[
        ~(
            regional[
                "ml_supported"
            ]
            &
            regional[
                "v5_survival_score"
            ].notna()
        )
    ].copy()

    print()
    print(
        f"Main ML recommendation pool: "
        f"{len(main_pool)}"
    )

    print(
        f"Regional-only / insufficient-score pool: "
        f"{len(regional_only)}"
    )

    # ========================================================
    # 8. SCORE MAIN POOL
    # ========================================================

    if not main_pool.empty:

        # ----------------------------------------------------
        # Available score columns
        # ----------------------------------------------------

        score_columns = {

            "survival":
                "v5_survival_score",

            "evidence":
                "historical_evidence_score",

            "soil":
                "soil_score",

            "climate":
                "climate_score",

            "pollution":
                "pollution_score",

            "user":
                "user_fit_score",
        }

        # ----------------------------------------------------
        # Build weighted score using ONLY available evidence.
        # ----------------------------------------------------

        weighted_sum = pd.Series(
            0.0,
            index=main_pool.index,
        )

        available_weight = pd.Series(
            0.0,
            index=main_pool.index,
        )

        for component, column in (
            score_columns.items()
        ):

            if column not in main_pool.columns:

                continue

            values = pd.to_numeric(
                main_pool[column],
                errors="coerce",
            )

            available = values.notna()

            weighted_sum.loc[
                available
            ] += (
                values.loc[
                    available
                ]
                * WEIGHTS[component]
            )

            available_weight.loc[
                available
            ] += WEIGHTS[component]

        # ----------------------------------------------------
        # Renormalize available components.
        # ----------------------------------------------------

        main_pool[
            "final_score"
        ] = np.where(
            available_weight > 0,
            weighted_sum
            / available_weight
            * 100,
            np.nan,
        )

        # ----------------------------------------------------
        # Survival safety information
        # ----------------------------------------------------

        main_pool[
            "survival_evidence_flag"
        ] = np.where(
            main_pool[
                "v5_survival_score"
            ] >= 60,
            "stronger",
            "lower",
        )

        # ----------------------------------------------------
        # Sort
        # ----------------------------------------------------

        main_pool = (
            main_pool
            .sort_values(
                [
                    "final_score",
                    "v5_survival_score",
                    "regional_evidence_score",
                ],
                ascending=[
                    False,
                    False,
                    False,
                ],
                na_position="last",
            )
            .reset_index(
                drop=True
            )
        )

        main_pool[
            "recommendation_rank"
        ] = (
            np.arange(
                len(main_pool)
            )
            + 1
        )

    # ========================================================
    # 9. REGIONAL-ONLY LABEL
    # ========================================================

    if not regional_only.empty:

        regional_only[
            "recommendation_status"
        ] = (
            "REGIONAL_EVIDENCE_ONLY"
        )

        regional_only[
            "final_score"
        ] = np.nan

        regional_only[
            "recommendation_rank"
        ] = np.nan

        regional_only[
            "reason"
        ] = (
            "Regional presence detected, "
            "but insufficient survival evidence "
            "for a main ML recommendation."
        )

    # ========================================================
    # 10. SAVE MAIN POOL
    # ========================================================

    main_pool.to_csv(
        MAIN_POOL_FILE,
        index=False,
    )

    regional_only.to_csv(
        REGIONAL_ONLY_FILE,
        index=False,
    )

    # ========================================================
    # 11. FINAL FILE
    # ========================================================

    if not main_pool.empty:

        final = main_pool.copy()

        final[
            "recommendation_status"
        ] = "MAIN_RECOMMENDATION"

        final[
            "reason"
        ] = (
            "Regional evidence plus "
            "survival-model support."
        )

    else:

        final = pd.DataFrame()

    final.to_csv(
        FINAL_FILE,
        index=False,
    )

    # ========================================================
    # 12. REPORT
    # ========================================================

    print()
    print("=" * 70)
    print("V7 FINAL RECOMMENDER COMPLETE")
    print("=" * 70)

    print()

    print(
        f"Main recommendations: "
        f"{len(main_pool)}"
    )

    print(
        f"Regional-only candidates: "
        f"{len(regional_only)}"
    )

    if not main_pool.empty:

        print()
        print(
            "MAIN RECOMMENDATIONS:"
        )

        columns = [
            "recommendation_rank",
            "species_normalized",
            "final_score",
            "v5_survival_score",
            "regional_evidence_score",
            "candidate_tier",
        ]

        print(
            main_pool[
                [
                    c
                    for c in columns
                    if c in main_pool.columns
                ]
            ]
            .head(20)
            .to_string(
                index=False
            )
        )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "A regional-only species is NOT given "
        "a fabricated survival score."
    )

    print(
        "Final score is a suitability ranking, "
        "not a calibrated survival probability."
    )

    print()
    print(
        "Files:"
    )

    print(
        f"  {FINAL_FILE}"
    )

    print(
        f"  {MAIN_POOL_FILE}"
    )

    print(
        f"  {REGIONAL_ONLY_FILE}"
    )


if __name__ == "__main__":
    main()