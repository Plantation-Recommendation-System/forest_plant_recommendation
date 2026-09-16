from pathlib import Path
import pandas as pd
import numpy as np

from scipy.stats import spearmanr


# ============================================================
# CONFIG
# ============================================================

V5_FILE = Path(
    "outputs/candidate_v5_predictions.csv"
)

EVIDENCE_FILE = Path(
    "outputs/candidate_evidence_scores.csv"
)

SOIL_FILE = Path(
    "outputs/candidate_soil_compatibility.csv"
)

SURVIVAL_TEST_FILE = Path(
    "data/survival_test.csv"
)

OUTPUT_FILE = Path(
    "outputs/recommendation_validation.csv"
)


# ============================================================
# SETTINGS
# ============================================================

# We need a common survival horizon.
#
# The original ranking experiment was problematic because
# different species had different observation durations.
#
# For this first validation, we restrict observations to
# reasonably comparable duration records.

MIN_DURATION_MONTHS = 10

MAX_DURATION_MONTHS = 24


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print()
    print("=" * 70)
    print("LOADING RECOMMENDATION VALIDATION DATA")
    print("=" * 70)

    v5 = pd.read_csv(V5_FILE)

    evidence = pd.read_csv(
        EVIDENCE_FILE
    )

    soil = pd.read_csv(
        SOIL_FILE
    )

    test = pd.read_csv(
        SURVIVAL_TEST_FILE
    )

    print()
    print(f"V5 predictions: {len(v5)} rows")
    print(f"Evidence:        {len(evidence)} rows")
    print(f"Soil:            {len(soil)} rows")
    print(f"Test survival:   {len(test)} rows")

    return v5, evidence, soil, test


# ============================================================
# SPECIES COLUMN
# ============================================================

def find_species_column(df):

    candidates = [
        "species_normalized",
        "species_full",
        "species",
        "scientific_name",
        "species_name",
    ]

    for column in candidates:

        if column in df.columns:
            return column

    raise ValueError(
        "No species column found. "
        f"Available columns: {list(df.columns)}"
    )


# ============================================================
# CREATE HISTORICAL SPECIES TARGET
# ============================================================

def create_species_targets(test):

    print()
    print("-" * 70)
    print("CREATING HELD-OUT SPECIES TARGETS")
    print("-" * 70)

    species_column = find_species_column(
        test
    )

    # --------------------------------------------------------
    # Duration filtering
    # --------------------------------------------------------

    if "duration_months" not in test.columns:

        raise ValueError(
            "duration_months is missing from survival_test.csv"
        )

    filtered = test[
        (
            test["duration_months"]
            >= MIN_DURATION_MONTHS
        )
        &
        (
            test["duration_months"]
            <= MAX_DURATION_MONTHS
        )
    ].copy()

    print()
    print(
        f"Duration window: "
        f"{MIN_DURATION_MONTHS}–"
        f"{MAX_DURATION_MONTHS} months"
    )

    print(
        f"Rows before filtering: {len(test)}"
    )

    print(
        f"Rows after filtering:  {len(filtered)}"
    )

    # --------------------------------------------------------
    # Need location
    # --------------------------------------------------------

    required = [
        "lat_dec",
        "lon_dec",
        species_column,
        "survival_per",
    ]

    missing = [
        c for c in required
        if c not in filtered.columns
    ]

    if missing:

        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # --------------------------------------------------------
    # Create location ID
    # --------------------------------------------------------

    filtered[
        "location_id"
    ] = (
        filtered["lat_dec"].round(4).astype(str)
        + "_"
        + filtered["lon_dec"].round(4).astype(str)
    )

    # --------------------------------------------------------
    # Aggregate actual survival by
    # location + species
    # --------------------------------------------------------

    actual = (
        filtered
        .groupby(
            [
                "location_id",
                species_column,
            ],
            as_index=False
        )
        .agg(
            actual_survival=(
                "survival_per",
                "mean"
            ),
            observations=(
                "survival_per",
                "count"
            )
        )
    )

    print(
        f"Location-species combinations: "
        f"{len(actual)}"
    )

    return actual, species_column


# ============================================================
# PREPARE MODEL RANKINGS
# ============================================================

def prepare_model_data(
    v5,
    evidence,
    soil
):

    print()
    print("-" * 70)
    print("PREPARING MODEL SCORES")
    print("-" * 70)

    v5_species = find_species_column(v5)

    evidence_species = find_species_column(
        evidence
    )

    soil_species = find_species_column(
        soil
    )

    # --------------------------------------------------------
    # Standardize species names
    # --------------------------------------------------------

    for df, column in [
        (v5, v5_species),
        (evidence, evidence_species),
        (soil, soil_species),
    ]:

        df["_species"] = (
            df[column]
            .astype(str)
            .str.strip()
            .str.lower()
        )

    # --------------------------------------------------------
    # V5 score
    # --------------------------------------------------------

    v5_score_column = (
        "v5_predicted_survival"
    )

    if v5_score_column not in v5.columns:

        raise ValueError(
            f"{v5_score_column} missing from V5 file"
        )

    v5_clean = v5[
        [
            "_species",
            v5_score_column,
        ]
    ].copy()

    v5_clean = (
        v5_clean
        .groupby(
            "_species",
            as_index=False
        )[v5_score_column]
        .mean()
    )

    # --------------------------------------------------------
    # Evidence score
    # --------------------------------------------------------

    if "evidence_score" not in evidence.columns:

        raise ValueError(
            "evidence_score missing from evidence file"
        )

    evidence_clean = evidence[
        [
            "_species",
            "evidence_score",
        ]
    ].copy()

    evidence_clean = (
        evidence_clean
        .groupby(
            "_species",
            as_index=False
        )["evidence_score"]
        .mean()
    )

    # --------------------------------------------------------
    # Soil-adjusted score
    # --------------------------------------------------------

    if "soil_adjusted_score" not in soil.columns:

        raise ValueError(
            "soil_adjusted_score missing from soil file"
        )

    soil_clean = soil[
        [
            "_species",
            "soil_adjusted_score",
        ]
    ].copy()

    soil_clean = (
        soil_clean
        .groupby(
            "_species",
            as_index=False
        )["soil_adjusted_score"]
        .mean()
    )

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    scores = v5_clean.merge(
        evidence_clean,
        on="_species",
        how="outer"
    )

    scores = scores.merge(
        soil_clean,
        on="_species",
        how="outer"
    )

    return scores


# ============================================================
# NORMALIZATION
# ============================================================

def minmax(series):

    series = pd.to_numeric(
        series,
        errors="coerce"
    )

    minimum = series.min()
    maximum = series.max()

    if pd.isna(minimum) or pd.isna(maximum):
        return pd.Series(
            np.nan,
            index=series.index
        )

    if maximum == minimum:

        return pd.Series(
            0.5,
            index=series.index
        )

    return (
        (series - minimum)
        /
        (maximum - minimum)
    )


# ============================================================
# BUILD COMPARABLE RANKING
# ============================================================

def evaluate_location(
    location_id,
    actual_location,
    scores
):

    # --------------------------------------------------------
    # Actual species
    # --------------------------------------------------------

    actual = actual_location.copy()

    actual[
        "_species"
    ] = (
        actual[
            actual.columns[
                1
            ]
        ]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # --------------------------------------------------------
    # Only species present in both
    # --------------------------------------------------------

    merged = actual.merge(
        scores,
        on="_species",
        how="inner"
    )

    if len(merged) < 3:
        return None

    # --------------------------------------------------------
    # Actual ranking
    # --------------------------------------------------------

    merged[
        "actual_rank"
    ] = (
        merged[
            "actual_survival"
        ]
        .rank(
            method="average",
            ascending=False
        )
    )

    # --------------------------------------------------------
    # Model scores
    # --------------------------------------------------------

    # Model A:
    # V5 alone

    merged[
        "v5_rank"
    ] = (
        merged[
            "v5_predicted_survival"
        ]
        .rank(
            method="average",
            ascending=False
        )
    )

    # Model B:
    # Evidence score

    merged[
        "evidence_rank"
    ] = (
        merged[
            "evidence_score"
        ]
        .rank(
            method="average",
            ascending=False
        )
    )

    # Model C:
    # Soil-adjusted score

    merged[
        "soil_rank"
    ] = (
        merged[
            "soil_adjusted_score"
        ]
        .rank(
            method="average",
            ascending=False
        )
    )

    # --------------------------------------------------------
    # Spearman
    # --------------------------------------------------------

    def spearman(
        predicted_rank
    ):

        if len(merged) < 3:
            return np.nan

        result = spearmanr(
            merged[
                predicted_rank
            ],
            merged[
                "actual_rank"
            ]
        )

        return result.statistic

    # --------------------------------------------------------
    # Top-k
    # --------------------------------------------------------

    def top_k(
        predicted_rank,
        k
    ):

        predicted_best = (
            merged
            .sort_values(
                predicted_rank
            )
            .head(k)
            [
                "_species"
            ]
            .tolist()
        )

        actual_best = (
            merged
            .sort_values(
                "actual_rank"
            )
            .head(k)
            [
                "_species"
            ]
            .tolist()
        )

        return int(
            any(
                species in actual_best
                for species in predicted_best
            )
        )

    # --------------------------------------------------------
    # Best actual species
    # --------------------------------------------------------

    actual_best = (
        merged
        .sort_values(
            "actual_rank"
        )
        .iloc[0]
    )

    best_species = (
        actual_best["_species"]
    )

    # --------------------------------------------------------
    # Rank of actual best species
    # --------------------------------------------------------

    def rank_of_best(
        predicted_rank
    ):

        row = merged[
            merged["_species"]
            == best_species
        ]

        if len(row) == 0:
            return np.nan

        return float(
            row.iloc[0][
                predicted_rank
            ]
        )

    return {
        "location_id": location_id,
        "n_species": len(merged),

        "v5_spearman": spearman(
            "v5_rank"
        ),

        "evidence_spearman": spearman(
            "evidence_rank"
        ),

        "soil_spearman": spearman(
            "soil_rank"
        ),

        "v5_top1": top_k(
            "v5_rank",
            1
        ),

        "v5_top3": top_k(
            "v5_rank",
            3
        ),

        "evidence_top1": top_k(
            "evidence_rank",
            1
        ),

        "evidence_top3": top_k(
            "evidence_rank",
            3
        ),

        "soil_top1": top_k(
            "soil_rank",
            1
        ),

        "soil_top3": top_k(
            "soil_rank",
            3
        ),

        "v5_rank_of_actual_best": rank_of_best(
            "v5_rank"
        ),

        "evidence_rank_of_actual_best": rank_of_best(
            "evidence_rank"
        ),

        "soil_rank_of_actual_best": rank_of_best(
            "soil_rank"
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    v5, evidence, soil, test = load_data()

    actual, test_species_column = (
        create_species_targets(
            test
        )
    )

    scores = prepare_model_data(
        v5,
        evidence,
        soil
    )

    print()
    print("-" * 70)
    print("EVALUATING HELD-OUT LOCATIONS")
    print("-" * 70)

    results = []

    for location_id, group in actual.groupby(
        "location_id"
    ):

        result = evaluate_location(
            location_id,
            group,
            scores
        )

        if result is not None:
            results.append(result)

    results_df = pd.DataFrame(
        results
    )

    if len(results_df) == 0:

        print()
        print(
            "ERROR: No locations had enough "
            "overlapping species."
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Save detailed results
    # --------------------------------------------------------

    results_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("RECOMMENDATION VALIDATION SUMMARY")
    print("=" * 70)

    print()
    print(
        f"Locations evaluated: "
        f"{len(results_df)}"
    )

    print()

    summary = pd.DataFrame({

        "Model": [
            "V5 survival model",
            "Historical evidence",
            "Evidence + soil",
        ],

        "Mean Spearman": [
            results_df[
                "v5_spearman"
            ].mean(),

            results_df[
                "evidence_spearman"
            ].mean(),

            results_df[
                "soil_spearman"
            ].mean(),
        ],

        "Median Spearman": [
            results_df[
                "v5_spearman"
            ].median(),

            results_df[
                "evidence_spearman"
            ].median(),

            results_df[
                "soil_spearman"
            ].median(),
        ],

        "Top-1 Accuracy": [
            results_df[
                "v5_top1"
            ].mean(),

            results_df[
                "evidence_top1"
            ].mean(),

            results_df[
                "soil_top1"
            ].mean(),
        ],

        "Top-3 Accuracy": [
            results_df[
                "v5_top3"
            ].mean(),

            results_df[
                "evidence_top3"
            ].mean(),

            results_df[
                "soil_top3"
            ].mean(),
        ],

        "Mean Rank of Actual Best": [
            results_df[
                "v5_rank_of_actual_best"
            ].mean(),

            results_df[
                "evidence_rank_of_actual_best"
            ].mean(),

            results_df[
                "soil_rank_of_actual_best"
            ].mean(),
        ],
    })

    # Format percentages
    summary_display = summary.copy()

    summary_display[
        "Top-1 Accuracy"
    ] *= 100

    summary_display[
        "Top-3 Accuracy"
    ] *= 100

    print(
        summary_display.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.3f}"
        )
    )

    print()
    print("-" * 70)
    print("INTERPRETATION")
    print("-" * 70)

    print()

    best_spearman = summary.loc[
        summary[
            "Mean Spearman"
        ].idxmax()
    ]

    best_top3 = summary.loc[
        summary[
            "Top-3 Accuracy"
        ].idxmax()
    ]

    print(
        f"Best mean Spearman: "
        f"{best_spearman['Model']}"
    )

    print(
        f"Best Top-3 accuracy: "
        f"{best_top3['Model']}"
    )

    print()

    print(
        "Lower 'Mean Rank of Actual Best' "
        "is better."
    )

    print(
        "Higher Spearman and Top-k accuracy "
        "are better."
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "This is an evaluation of ranking behavior, "
        "not a claim of true field survival probability."
    )

    print()
    print(
        f"Detailed results saved to:"
    )

    print(
        f"  {OUTPUT_FILE}"
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()