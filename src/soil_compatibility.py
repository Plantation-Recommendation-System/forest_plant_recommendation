from pathlib import Path
import json
import sys

import pandas as pd


# ============================================================
# CONFIG
# ============================================================

SOIL_RESULT = Path("outputs/soil_multi_prediction.json")

CANDIDATE_FILE = Path(
    "outputs/candidate_evidence_scores.csv"
)

OUTPUT_FILE = Path(
    "outputs/candidate_soil_compatibility.csv"
)

FINAL_OUTPUT_FILE = Path(
    "outputs/final_candidate_ranking.csv"
)


# ============================================================
# SOIL CLASSES
# ============================================================

SOIL_CLASSES = [
    "Alluvial",
    "Arid",
    "Black",
    "Laterite",
    "Mountain",
    "Red",
    "Yellow",
]


# ============================================================
# SOIL COMPATIBILITY MATRIX
#
# Prototype ecological suitability scores.
#
# 100 = very compatible
#  80 = highly compatible
#  70 = good
#  60 = moderate
#  50 = uncertain/moderate
#  40 = relatively weak
#  30 = weak
#  20 = poor
#  10 = very poor
#
# These are NOT survival probabilities.
# ============================================================

COMPATIBILITY = {

    "Gmelina arborea": {
        "Alluvial": 100,
        "Arid": 30,
        "Black": 60,
        "Laterite": 50,
        "Mountain": 50,
        "Red": 70,
        "Yellow": 60,
    },

    "Xylia xylocarpa": {
        "Alluvial": 60,
        "Arid": 40,
        "Black": 10,
        "Laterite": 100,
        "Mountain": 50,
        "Red": 100,
        "Yellow": 60,
    },

    "Mesua ferrea": {
        "Alluvial": 70,
        "Arid": 20,
        "Black": 50,
        "Laterite": 60,
        "Mountain": 70,
        "Red": 70,
        "Yellow": 60,
    },

    "Acacia mangium": {
        "Alluvial": 50,
        "Arid": 20,
        "Black": 30,
        "Laterite": 80,
        "Mountain": 30,
        "Red": 80,
        "Yellow": 70,
    },

    "Melia azedarach": {
        "Alluvial": 80,
        "Arid": 70,
        "Black": 80,
        "Laterite": 60,
        "Mountain": 50,
        "Red": 70,
        "Yellow": 60,
    },

    "Bischofia javanica": {
        "Alluvial": 90,
        "Arid": 40,
        "Black": 90,
        "Laterite": 60,
        "Mountain": 50,
        "Red": 70,
        "Yellow": 60,
    },

    "Acacia auriculiformis": {
        "Alluvial": 90,
        "Arid": 80,
        "Black": 95,
        "Laterite": 90,
        "Mountain": 40,
        "Red": 90,
        "Yellow": 80,
    },

    "Tectona grandis": {
        "Alluvial": 100,
        "Arid": 50,
        "Black": 30,
        "Laterite": 30,
        "Mountain": 50,
        "Red": 70,
        "Yellow": 60,
    },

    "Calophyllum inophyllum": {
        "Alluvial": 60,
        "Arid": 20,
        "Black": 50,
        "Laterite": 50,
        "Mountain": 20,
        "Red": 60,
        "Yellow": 50,
    },

    "Swietenia macrophylla": {
        "Alluvial": 90,
        "Arid": 30,
        "Black": 60,
        "Laterite": 50,
        "Mountain": 30,
        "Red": 60,
        "Yellow": 50,
    },

    "Adenanthera pavonina": {
        "Alluvial": 80,
        "Arid": 40,
        "Black": 80,
        "Laterite": 60,
        "Mountain": 30,
        "Red": 70,
        "Yellow": 60,
    },

    "Artocarpus heterophyllus": {
        "Alluvial": 100,
        "Arid": 20,
        "Black": 70,
        "Laterite": 50,
        "Mountain": 30,
        "Red": 60,
        "Yellow": 50,
    },

    "Acrocarpus fraxinifolius": {
        "Alluvial": 60,
        "Arid": 10,
        "Black": 40,
        "Laterite": 60,
        "Mountain": 80,
        "Red": 80,
        "Yellow": 60,
    },

    "Careya arborea": {
        "Alluvial": 60,
        "Arid": 50,
        "Black": 60,
        "Laterite": 90,
        "Mountain": 50,
        "Red": 80,
        "Yellow": 60,
    },

    "Caryota urens": {
        "Alluvial": 80,
        "Arid": 20,
        "Black": 70,
        "Laterite": 50,
        "Mountain": 60,
        "Red": 60,
        "Yellow": 50,
    },

    "Cananga odorata": {
        "Alluvial": 80,
        "Arid": 20,
        "Black": 60,
        "Laterite": 50,
        "Mountain": 20,
        "Red": 60,
        "Yellow": 50,
    },
}


# ============================================================
# LOAD SOIL RESULT
# ============================================================

def load_soil_prediction():

    if not SOIL_RESULT.exists():

        print()
        print("ERROR: Soil prediction file not found:")
        print(f"  {SOIL_RESULT}")
        print()
        print(
            "Run predict_soil_multi.py first."
        )

        sys.exit(1)

    with open(
        SOIL_RESULT,
        "r",
        encoding="utf-8"
    ) as f:

        result = json.load(f)

    return result


# ============================================================
# NORMALIZE SOIL PROBABILITIES
# ============================================================

def normalize_probabilities(probabilities):

    """
    The multi-image predictor may store probabilities either as:

        0.8753
        0.0352
        ...

    OR as percentages:

        87.53
        3.52
        ...

    Convert everything to fractions in [0, 1].
    """

    cleaned = {}

    for soil_class, value in probabilities.items():

        if soil_class not in SOIL_CLASSES:
            continue

        try:
            value = float(value)
        except (TypeError, ValueError):
            continue

        cleaned[soil_class] = value

    total = sum(cleaned.values())

    if total <= 0:
        raise ValueError(
            "Soil probability distribution is empty or invalid."
        )

    # If values look like percentages, convert to fractions.
    if total > 1.5:
        cleaned = {
            soil_class: value / 100.0
            for soil_class, value in cleaned.items()
        }

    # Normalize one more time to guarantee sum = 1.
    total = sum(cleaned.values())

    cleaned = {
        soil_class: value / total
        for soil_class, value in cleaned.items()
    }

    return cleaned


# ============================================================
# FIND COLUMN
# ============================================================

def find_column(
    dataframe,
    candidates,
    description
):

    for column in candidates:

        if column in dataframe.columns:
            return column

    print()
    print(
        f"ERROR: Could not find {description}."
    )

    print()
    print("Expected one of:")

    for column in candidates:
        print(f"  - {column}")

    print()
    print("Available columns:")

    for column in dataframe.columns:
        print(f"  - {column}")

    print()

    return None


# ============================================================
# SPECIES NORMALIZATION
# ============================================================

def normalize_species_name(value):

    if pd.isna(value):
        return ""

    value = str(value).strip()

    # Standardize capitalization.
    parts = value.split()

    if len(parts) >= 2:

        return (
            parts[0].capitalize()
            + " "
            + parts[1].lower()
        )

    return value


# ============================================================
# CALCULATE SOIL COMPATIBILITY
# ============================================================

def calculate_compatibility(
    species,
    probabilities
):

    species = normalize_species_name(
        species
    )

    if species not in COMPATIBILITY:
        return None

    score = 0.0

    for soil_class, probability in probabilities.items():

        compatibility = COMPATIBILITY[
            species
        ][soil_class]

        score += (
            probability
            * compatibility
        )

    return score


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("SOIL → SPECIES COMPATIBILITY")
    print("=" * 70)

    # --------------------------------------------------------
    # Load files
    # --------------------------------------------------------

    if not CANDIDATE_FILE.exists():

        print()
        print(
            "ERROR: Candidate evidence file not found:"
        )
        print(
            f"  {CANDIDATE_FILE}"
        )

        sys.exit(1)

    soil_result = load_soil_prediction()

    candidates = pd.read_csv(
        CANDIDATE_FILE
    )

    # --------------------------------------------------------
    # Soil prediction
    # --------------------------------------------------------

    predicted_soil = soil_result.get(
        "predicted_soil"
    )

    confidence = float(
        soil_result.get(
            "confidence",
            0
        )
    )

    confidence_percent = float(
        soil_result.get(
            "confidence_percent",
            confidence * 100
        )
    )

    confidence_level = soil_result.get(
        "confidence_level",
        "LOW"
    )

    soil_usable = bool(
        soil_result.get(
            "soil_signal_usable",
            False
        )
    )

    raw_probabilities = soil_result.get(
        "probabilities",
        {}
    )

    # --------------------------------------------------------
    # FIX:
    # Convert percentage probabilities to fractions.
    # --------------------------------------------------------

    probabilities = normalize_probabilities(
        raw_probabilities
    )

    print()
    print(
        f"Predicted soil: {predicted_soil}"
    )

    print(
        f"Confidence: {confidence_percent:.2f}%"
    )

    print(
        f"Confidence level: {confidence_level}"
    )

    print(
        f"Signal usable: {soil_usable}"
    )

    # --------------------------------------------------------
    # Display corrected probabilities
    # --------------------------------------------------------

    print()
    print(
        "Normalized soil probabilities:"
    )

    sorted_probabilities = sorted(
        probabilities.items(),
        key=lambda x: x[1],
        reverse=True
    )

    for soil_class, probability in sorted_probabilities:

        print(
            f"  {soil_class:<14}"
            f"{probability * 100:.2f}%"
        )

    print()

    # --------------------------------------------------------
    # Find species column
    # --------------------------------------------------------

    species_column = find_column(
        candidates,
        [
            "species_normalized",
            "species_full",
            "species",
            "scientific_name",
            "species_name",
        ],
        "species column"
    )

    if species_column is None:
        sys.exit(1)

    print(
        f"Using species column: "
        f"{species_column}"
    )

    # --------------------------------------------------------
    # Find evidence score
    # --------------------------------------------------------

    evidence_column = find_column(
        candidates,
        [
            "evidence_score",
            "final_score",
            "score",
        ],
        "evidence score column"
    )

    if evidence_column is None:
        sys.exit(1)

    print(
        f"Using evidence score column: "
        f"{evidence_column}"
    )

    # --------------------------------------------------------
    # Calculate soil compatibility
    # --------------------------------------------------------

    candidates[
        "soil_compatibility"
    ] = candidates[
        species_column
    ].apply(
        lambda species:
            calculate_compatibility(
                species,
                probabilities
            )
    )

    # --------------------------------------------------------
    # Check unmatched species
    # --------------------------------------------------------

    candidates[
        "soil_compatibility_available"
    ] = candidates[
        "soil_compatibility"
    ].notna()

    unmatched = candidates[
        ~candidates[
            "soil_compatibility_available"
        ]
    ]

    if len(unmatched) > 0:

        print()
        print(
            "WARNING: Missing compatibility rules:"
        )

        for species in unmatched[
            species_column
        ].tolist():

            print(
                f"  - {species}"
            )

        print()

        # Do NOT fabricate ecological scores.
        # Keep them as NaN and exclude them
        # from soil influence.

    # --------------------------------------------------------
    # Soil influence
    # --------------------------------------------------------

    evidence_scores = pd.to_numeric(
        candidates[
            evidence_column
        ],
        errors="coerce"
    )

    # Start with original evidence score.
    candidates[
        "soil_adjusted_score"
    ] = evidence_scores

    # Only apply soil to species for which
    # compatibility evidence exists AND
    # soil signal is usable.
    if soil_usable:

        valid_soil = (
            candidates[
                "soil_compatibility_available"
            ]
        )

        candidates.loc[
            valid_soil,
            "soil_adjusted_score"
        ] = (
            evidence_scores[valid_soil] * 0.85
            +
            candidates.loc[
                valid_soil,
                "soil_compatibility"
            ] * 0.15
        )

    # --------------------------------------------------------
    # Ranking before soil
    # --------------------------------------------------------

    candidates[
        "evidence_rank"
    ] = (
        evidence_scores
        .rank(
            method="min",
            ascending=False
        )
        .astype(int)
    )

    # --------------------------------------------------------
    # Final ranking
    # --------------------------------------------------------

    candidates = candidates.sort_values(
        "soil_adjusted_score",
        ascending=False
    ).reset_index(
        drop=True
    )

    candidates[
        "final_rank"
    ] = range(
        1,
        len(candidates) + 1
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    candidates.to_csv(
        OUTPUT_FILE,
        index=False
    )

    candidates.to_csv(
        FINAL_OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Display ranking
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("FINAL SOIL-ADJUSTED RANKING")
    print("-" * 70)

    print()

    display_columns = [
        "final_rank",
        species_column,
        evidence_column,
        "soil_compatibility",
        "soil_compatibility_available",
        "soil_adjusted_score",
    ]

    optional_columns = [
        "survival_observations",
        "pune_tree_count",
    ]

    for column in optional_columns:

        if column in candidates.columns:
            display_columns.append(
                column
            )

    print(
        candidates[
            display_columns
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Impact summary
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("SOIL IMPACT SUMMARY")
    print("-" * 70)

    print()

    for _, row in candidates.iterrows():

        species = row[
            species_column
        ]

        old_score = row[
            evidence_column
        ]

        new_score = row[
            "soil_adjusted_score"
        ]

        change = (
            new_score
            - old_score
        )

        soil_score = row[
            "soil_compatibility"
        ]

        if pd.isna(soil_score):

            soil_text = "N/A"

        else:

            soil_text = (
                f"{soil_score:.2f}"
            )

        print(
            f"{int(row['final_rank']):>2}. "
            f"{str(species):<35} "
            f"{old_score:>7.2f} → "
            f"{new_score:>7.2f} "
            f"({change:+.2f}) "
            f"Soil={soil_text}"
        )

    # --------------------------------------------------------
    # Files
    # --------------------------------------------------------

    print()
    print("=" * 70)

    print(
        "Saved compatibility output:"
    )

    print(
        f"  {OUTPUT_FILE}"
    )

    print(
        "Saved final ranking:"
    )

    print(
        f"  {FINAL_OUTPUT_FILE}"
    )

    print()

    # --------------------------------------------------------
    # Methodology
    # --------------------------------------------------------

    print(
        "IMPORTANT:"
    )

    print(
        "Soil compatibility is a prototype "
        "rule-based suitability signal."
    )

    print(
        "It is NOT a calibrated survival probability."
    )

    print(
        "The V5 survival model remains separate "
        "from this soil layer."
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()