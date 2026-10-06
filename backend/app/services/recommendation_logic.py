
import pandas as pd
import numpy as np


REQUIRED_COLUMNS = [
    "project_species_resolved",
    "environmental_rank",
    "climate_tier",
    "soil_texture_match_pct",
    "soil_texture_match_label",
    "predicted_survival_pct",
    "seen_in_catboost_training"
]


CLIMATE_TEXT = {
    "A_strict_match": "Strong climate match",
    "B_near_match": "Near climate match",
    "C_partial_match": "Partial climate match",
    "D_weak_match": "Weak climate match"
}


def _to_bool_series(series):
    """
    Safely normalize True/False values from CSV or DataFrame.
    """
    return (
        series
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("true")
    )


def _soil_interpretation(row):

    label = row["soil_texture_match_label"]
    pct = row["soil_texture_match_pct"]

    if pd.isna(pct):
        return "Soil-texture evidence uncertain"

    if label == "strong_texture_match":
        return f"Strong soil-texture agreement ({pct:.1f}%)"

    if label == "moderate_texture_match":
        return f"Moderate soil-texture agreement ({pct:.1f}%)"

    if label == "weak_texture_match":
        return f"Weak soil-texture agreement ({pct:.1f}%)"

    return "Soil-texture evidence uncertain"


def _catboost_interpretation(row):

    survival = row["predicted_survival_pct"]
    seen = row["seen_in_catboost_training"]

    if pd.isna(survival):
        return "No CatBoost survival estimate available"

    if seen:
        return (
            f"Supporting survival estimate: {survival:.1f}% "
            f"(species represented in CatBoost training)"
        )

    return (
        f"Supporting survival estimate: {survival:.1f}% "
        f"(species unseen during CatBoost training; "
        f"interpret cautiously)"
    )


def _recommendation_basis(row):

    climate = row["climate_tier"]
    soil = row["soil_texture_match_label"]

    if climate == "A_strict_match":
        return (
            "Recommended primarily because the site has "
            "a strict climate match with the species profile; "
            "soil compatibility provides secondary support."
        )

    if climate == "B_near_match":
        return (
            "Recommended because climate conditions are "
            "close to the species profile and soil evidence "
            "provides additional support."
        )

    if (
        climate == "C_partial_match"
        and soil == "strong_texture_match"
    ):
        return (
            "Secondary recommendation: climate match is partial, "
            "but strong soil-texture compatibility supports "
            "keeping the species as a candidate."
        )

    return (
        "Additional candidate requiring greater caution."
    )


def _overall_evidence(row):

    climate = row["climate_tier"]
    soil = row["soil_texture_match_label"]
    seen = row["seen_in_catboost_training"]

    if (
        climate == "A_strict_match"
        and soil == "strong_texture_match"
    ):

        if seen:
            return "strong_environmental_evidence"

        return "strong_environmental_evidence_model_caution"

    if climate == "B_near_match":
        return "good_environmental_evidence"

    if (
        climate == "C_partial_match"
        and soil == "strong_texture_match"
    ):
        return "moderate_environmental_evidence"

    return "limited_environmental_evidence"


def finalize_recommendations(
    candidates,
    top_n=10,
    scenario_status="live_assessment"
):
    """
    Finalize species recommendations.

    Ranking principle:
        1. Climate screening is primary.
        2. Soil refinement is secondary.
        3. Existing environmental_rank is preserved.
        4. CatBoost is supporting evidence only.
        5. No weighted climate/soil/CatBoost score is used.

    Parameters
    ----------
    candidates : pandas.DataFrame
        Candidate species after environmental screening and
        CatBoost prediction.

    top_n : int
        Number of recommendations displayed by default.

    scenario_status : str
        Example:
        - "demo_pipeline_only"
        - "live_assessment"

    Returns
    -------
    pandas.DataFrame
    """

    df = candidates.copy()

    missing = [
        col
        for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required recommendation columns: {missing}"
        )

    if len(df) == 0:
        return df

    # Normalize model exposure flag
    df["seen_in_catboost_training"] = (
        _to_bool_series(
            df["seen_in_catboost_training"]
        )
    )

    # -----------------------------------------------------
    # PRESERVE ENVIRONMENTAL ORDER
    # -----------------------------------------------------

    df = (
        df
        .sort_values(
            "environmental_rank",
            ascending=True
        )
        .reset_index(
            drop=True
        )
    )

    df["final_recommendation_rank"] = (
        range(
            1,
            len(df) + 1
        )
    )

    # -----------------------------------------------------
    # DISPLAY GROUP
    # top_n is UI behavior, not biological cutoff
    # -----------------------------------------------------

    df["recommendation_display_group"] = np.where(
        df["final_recommendation_rank"] <= top_n,
        "top_recommendation",
        "additional_candidate"
    )

    # -----------------------------------------------------
    # INTERPRETATIONS
    # -----------------------------------------------------

    df["climate_interpretation"] = (
        df["climate_tier"]
        .map(CLIMATE_TEXT)
        .fillna("Climate evidence uncertain")
    )

    df["soil_interpretation"] = (
        df.apply(
            _soil_interpretation,
            axis=1
        )
    )

    df["catboost_interpretation"] = (
        df.apply(
            _catboost_interpretation,
            axis=1
        )
    )

    df["catboost_caution"] = np.where(
        df["seen_in_catboost_training"],
        (
            "Supporting estimate only; "
            "held-out model generalization was weak."
        ),
        (
            "Higher uncertainty: species was unseen during "
            "CatBoost training and held-out model "
            "generalization was weak."
        )
    )

    df["recommendation_basis"] = (
        df.apply(
            _recommendation_basis,
            axis=1
        )
    )

    df["overall_evidence_label"] = (
        df.apply(
            _overall_evidence,
            axis=1
        )
    )

    df["scenario_status"] = scenario_status

    return df


def get_top_recommendations(
    candidates,
    top_n=10,
    scenario_status="live_assessment"
):
    """
    Convenience wrapper for the website/API.
    """

    final = finalize_recommendations(
        candidates=candidates,
        top_n=top_n,
        scenario_status=scenario_status
    )

    return (
        final[
            final["final_recommendation_rank"] <= top_n
        ]
        .copy()
        .reset_index(drop=True)
    )


def to_api_records(df):
    """
    Convert result DataFrame to JSON-safe records.
    """

    safe = df.copy()

    safe = safe.where(
        pd.notnull(safe),
        None
    )

    return safe.to_dict(
        orient="records"
    )
