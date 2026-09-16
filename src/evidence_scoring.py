import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

BASE = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    BASE
    / "outputs"
    / "candidate_v5_predictions.csv"
)

OUTPUT_FILE = (
    BASE
    / "outputs"
    / "candidate_evidence_scores.csv"
)


# ============================================================
# SCORING PARAMETERS
# ============================================================

# Maximum number of observations at which historical
# evidence is treated as having reached high confidence.
#
# This is NOT a statistical confidence interval.
# It is only a conservative evidence-strength transformation.
OBSERVATION_REFERENCE = 15

# Reference number of trees for local Pune presence.
TREE_REFERENCE = 1000


# ============================================================
# NORMALIZATION FUNCTIONS
# ============================================================

def min_max_normalize(series):
    """
    Normalize a numeric series to 0-100.

    If all values are identical, return 50 for every item
    because there is no ranking information in that signal.
    """

    minimum = series.min()
    maximum = series.max()

    if pd.isna(minimum) or pd.isna(maximum):
        return pd.Series(
            50.0,
            index=series.index
        )

    if maximum == minimum:
        return pd.Series(
            50.0,
            index=series.index
        )

    return (
        (series - minimum)
        / (maximum - minimum)
        * 100
    )


def observation_strength(n):
    """
    Convert number of survival observations into an
    evidence-strength score between 0 and 100.

    Uses a saturating curve rather than treating 18 observations
    as 18 times stronger than one observation.
    """

    n = max(float(n), 0.0)

    return (
        100
        * (1 - np.exp(-n / OBSERVATION_REFERENCE))
    )


def pune_presence_score(tree_count):
    """
    Convert Pune census tree count into a 0-100 local-presence
    signal using log scaling.

    This prevents very common species from dominating purely
    because they have thousands of census records.
    """

    tree_count = max(float(tree_count), 0.0)

    return (
        100
        * np.log1p(tree_count)
        / np.log1p(TREE_REFERENCE)
    ).clip(0, 100)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("EVIDENCE-AWARE SPECIES SCORING")
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}\n\n"
            f"Run predict_candidates.py first."
        )

    df = pd.read_csv(INPUT_FILE)

    print(f"\nInput shape: {df.shape}")

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required_columns = [
        "species_normalized",
        "pune_tree_count",
        "survival_observations",
        "observed_survival_mean",
        "v5_predicted_survival",
        "evidence_tier",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    numeric_columns = [
        "pune_tree_count",
        "survival_observations",
        "observed_survival_mean",
        "v5_predicted_survival",
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Check for invalid values
    # --------------------------------------------------------

    if df["v5_predicted_survival"].isna().any():

        raise ValueError(
            "Some candidates have no V5 prediction."
        )

    # --------------------------------------------------------
    # 1. V5 MODEL SIGNAL
    # --------------------------------------------------------

    print("\nCalculating V5 model signal...")

    df["model_signal"] = min_max_normalize(
        df["v5_predicted_survival"]
    )

    # --------------------------------------------------------
    # 2. HISTORICAL SURVIVAL SIGNAL
    # --------------------------------------------------------

    print("Calculating historical survival signal...")

    # Historical survival is already expressed as a percentage.
    df["historical_survival_signal"] = (
        df["observed_survival_mean"]
        .clip(0, 100)
    )

    # --------------------------------------------------------
    # 3. OBSERVATION EVIDENCE STRENGTH
    # --------------------------------------------------------

    print("Calculating observation evidence strength...")

    df["observation_strength"] = (
        df["survival_observations"]
        .fillna(0)
        .apply(observation_strength)
    )

    # --------------------------------------------------------
    # 4. PUNE LOCAL PRESENCE
    # --------------------------------------------------------

    print("Calculating Pune presence signal...")

    df["pune_presence_signal"] = (
        df["pune_tree_count"]
        .fillna(0)
        .apply(pune_presence_score)
    )

    # --------------------------------------------------------
    # 5. EVIDENCE-ADJUSTED HISTORICAL SIGNAL
    # --------------------------------------------------------
    #
    # We shrink the historical survival signal toward the
    # overall candidate-set average when there are few
    # observations.
    #
    # This prevents a species with 1-2 observations from
    # being treated exactly like one with many observations.
    # --------------------------------------------------------

    overall_survival = (
        df["historical_survival_signal"]
        .mean()
    )

    df["historical_evidence_adjusted"] = (
        df["historical_survival_signal"]
        * (
            df["observation_strength"] / 100
        )
        +
        overall_survival
        * (
            1
            - df["observation_strength"] / 100
        )
    )

    # --------------------------------------------------------
    # 6. FINAL EVIDENCE SCORE
    # --------------------------------------------------------
    #
    # IMPORTANT:
    #
    # These weights are deliberately presented as a
    # transparent V1 ranking heuristic.
    #
    # They are NOT learned weights and NOT statistically
    # validated probabilities.
    # --------------------------------------------------------

    MODEL_WEIGHT = 0.40
    HISTORICAL_WEIGHT = 0.40
    PUNE_WEIGHT = 0.20

    df["evidence_score"] = (
        MODEL_WEIGHT
        * df["model_signal"]
        +
        HISTORICAL_WEIGHT
        * df["historical_evidence_adjusted"]
        +
        PUNE_WEIGHT
        * df["pune_presence_signal"]
    )

    # --------------------------------------------------------
    # 7. Evidence confidence label
    # --------------------------------------------------------

    def confidence_label(row):

        observations = row["survival_observations"]
        trees = row["pune_tree_count"]

        if observations >= 10 and trees >= 100:
            return "High"

        if observations >= 5 or trees >= 100:
            return "Moderate"

        return "Low"

    df["evidence_confidence"] = (
        df.apply(
            confidence_label,
            axis=1
        )
    )

    # --------------------------------------------------------
    # 8. Rank
    # --------------------------------------------------------

    df = df.sort_values(
        "evidence_score",
        ascending=False
    ).reset_index(drop=True)

    df.insert(
        0,
        "evidence_rank",
        range(1, len(df) + 1)
    )

    # --------------------------------------------------------
    # 9. Interpretation label
    # --------------------------------------------------------

    def interpretation(row):

        if (
            row["historical_survival_signal"] >= 70
            and row["survival_observations"] >= 5
        ):
            return "Strong survival evidence"

        if (
            row["historical_survival_signal"] >= 50
            and row["survival_observations"] >= 5
        ):
            return "Moderate survival evidence"

        if row["survival_observations"] >= 5:
            return "Limited survival evidence"

        return "Very limited survival evidence"

    df["survival_evidence_interpretation"] = (
        df.apply(
            interpretation,
            axis=1
        )
    )

    # --------------------------------------------------------
    # 10. Save
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("EVIDENCE-AWARE RESULTS")
    print("=" * 70)

    display_columns = [
        "evidence_rank",
        "species_normalized",
        "evidence_score",
        "v5_predicted_survival",
        "historical_survival_signal",
        "survival_observations",
        "pune_tree_count",
        "evidence_confidence",
    ]

    print(
        df[
            display_columns
        ].to_string(index=False)
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("COMPLETE")
    print("=" * 70)

    print(f"\nSaved to:")
    print(OUTPUT_FILE)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()