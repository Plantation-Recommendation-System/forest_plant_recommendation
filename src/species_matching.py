import pandas as pd
import re
import unicodedata
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]

SURVIVAL_FILE = BASE / "data" / "survival_ml_cleaned.csv"
PUNE_FILE = BASE / "data" / "pune_final_candidate_pool.csv"
OUTPUT_FILE = BASE / "outputs" / "species_match_results.csv"


def normalize_species_name(name):
    """
    Conservative species-name normalization.

    Keeps the first two biological tokens:
    Genus + species epithet.

    Example:
        Acacia auriculiformis
        Acacia auriculiformis Benth.

    ->  acacia auriculiformis
    """

    if pd.isna(name):
        return ""

    name = unicodedata.normalize("NFKD", str(name))
    name = name.lower().strip()

    # Remove punctuation
    name = re.sub(r"[^a-z\s]", " ", name)

    # Collapse whitespace
    tokens = name.split()

    if len(tokens) < 2:
        return ""

    return f"{tokens[0]} {tokens[1]}"


def main():

    print("Loading datasets...")

    survival = pd.read_csv(SURVIVAL_FILE)
    pune = pd.read_csv(PUNE_FILE)

    print(f"Survival rows: {len(survival)}")
    print(f"Pune candidates: {len(pune)}")

    # Normalize survival species
    survival["species_normalized"] = (
        survival["species_full"]
        .apply(normalize_species_name)
    )

    # Normalize Pune candidates
    pune["species_normalized"] = (
        pune["species_normalized"]
        .apply(normalize_species_name)
    )

    # Unique survival species
    survival_species = (
        survival[
            ["species_normalized", "species_full"]
        ]
        .drop_duplicates()
        .sort_values("species_normalized")
    )

    # Match
    results = pune.merge(
        survival_species,
        on="species_normalized",
        how="left"
    )

    results["survival_species_match"] = (
        results["species_full"].notna()
    )

    # Count observations directly from the survival dataset
    observation_counts = (
        survival.groupby("species_normalized")
        .size()
        .reset_index(name="survival_obs_from_dataset")
    )

    results = results.merge(
        observation_counts,
        on="species_normalized",
        how="left"
    )

    results["survival_obs_from_dataset"] = (
        results["survival_obs_from_dataset"]
        .fillna(0)
        .astype(int)
    )

    # Use the observation count calculated directly from the
    # survival dataset. Keep the original Pune candidate field
    # separately if it exists.
    results["survival_observations"] = results[
        "survival_obs_from_dataset"
    ]

    results.drop(
        columns=["survival_obs_from_dataset"],
        inplace=True
    )


    # Save
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    results.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nSpecies matching complete.")
    print(f"Output: {OUTPUT_FILE}")

    print("\nMatch summary:")
    print(
        results[
            [
                "species_normalized",
                "species_full",
                "survival_species_match",
                "survival_observations"
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()