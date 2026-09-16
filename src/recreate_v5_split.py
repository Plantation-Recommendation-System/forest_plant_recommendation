from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = Path(
    "data/survival_ml_cleaned.csv"
)

OUTPUT_DIR = Path(
    "data"
)

TRAIN_FILE = OUTPUT_DIR / "survival_train.csv"
VALIDATION_FILE = OUTPUT_DIR / "survival_validation.csv"
TEST_FILE = OUTPUT_DIR / "survival_test.csv"


# ============================================================
# SETTINGS
# ============================================================

TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20
RANDOM_STATE = 42

LOCATION_DECIMALS = 4


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("RECREATING V5 GEOGRAPHIC TRAIN / VALIDATION / TEST SPLIT")
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    if not INPUT_FILE.exists():

        print()
        print(
            "ERROR: Input dataset not found:"
        )
        print(
            f"  {INPUT_FILE}"
        )
        print()

        return

    df = pd.read_csv(
        INPUT_FILE
    )

    print()
    print(
        f"Dataset rows: {len(df)}"
    )

    print(
        f"Dataset columns: {len(df.columns)}"
    )

    # --------------------------------------------------------
    # Check coordinates
    # --------------------------------------------------------

    required = [
        "lat_dec",
        "lon_dec",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        print()
        print(
            f"ERROR: Missing columns: {missing}"
        )

        return

    # --------------------------------------------------------
    # Geographic group
    #
    # Same rounded coordinate = same geographic group.
    #
    # This prevents observations from the same location
    # leaking between train and test.
    # --------------------------------------------------------

    df["_location_group"] = (
        df["lat_dec"]
        .round(LOCATION_DECIMALS)
        .astype(str)
        +
        "_"
        +
        df["lon_dec"]
        .round(LOCATION_DECIMALS)
        .astype(str)
    )

    print()
    print(
        "Unique geographic groups:",
        df["_location_group"].nunique()
    )

    # --------------------------------------------------------
    # FIRST SPLIT
    #
    # 80% development
    # 20% final test
    # --------------------------------------------------------

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    development_idx, test_idx = next(
        splitter.split(
            df,
            groups=df["_location_group"]
        )
    )

    development = (
        df.iloc[
            development_idx
        ]
        .copy()
    )

    test = (
        df.iloc[
            test_idx
        ]
        .copy()
    )

    # --------------------------------------------------------
    # SECOND SPLIT
    #
    # 80% train
    # 20% validation
    #
    # Since development = 80% of total,
    # validation = 16% total.
    # --------------------------------------------------------

    splitter_val = GroupShuffleSplit(
        n_splits=1,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
    )

    train_idx, validation_idx = next(
        splitter_val.split(
            development,
            groups=development[
                "_location_group"
            ]
        )
    )

    train = (
        development.iloc[
            train_idx
        ]
        .copy()
    )

    validation = (
        development.iloc[
            validation_idx
        ]
        .copy()
    )

    # --------------------------------------------------------
    # Remove helper column
    # --------------------------------------------------------

    for dataset in [
        train,
        validation,
        test,
    ]:

        dataset.drop(
            columns=[
                "_location_group"
            ],
            inplace=True
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    train.to_csv(
        TRAIN_FILE,
        index=False
    )

    validation.to_csv(
        VALIDATION_FILE,
        index=False
    )

    test.to_csv(
        TEST_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("SPLIT RESULT")
    print("-" * 70)

    print()

    print(
        f"Train:      {len(train):>5} rows"
    )

    print(
        f"Validation: {len(validation):>5} rows"
    )

    print(
        f"Test:       {len(test):>5} rows"
    )

    print(
        f"Total:      "
        f"{len(train) + len(validation) + len(test):>5} rows"
    )

    print()

    # --------------------------------------------------------
    # Location counts
    # --------------------------------------------------------

    def location_count(dataset):

        return (
            dataset[
                [
                    "lat_dec",
                    "lon_dec"
                ]
            ]
            .drop_duplicates()
            .shape[0]
        )

    print(
        f"Train locations:      "
        f"{location_count(train)}"
    )

    print(
        f"Validation locations: "
        f"{location_count(validation)}"
    )

    print(
        f"Test locations:       "
        f"{location_count(test)}"
    )

    # --------------------------------------------------------
    # Check leakage
    # --------------------------------------------------------

    def get_locations(dataset):

        return set(
            zip(
                dataset["lat_dec"].round(
                    LOCATION_DECIMALS
                ),
                dataset["lon_dec"].round(
                    LOCATION_DECIMALS
                ),
            )
        )

    train_locations = get_locations(
        train
    )

    validation_locations = get_locations(
        validation
    )

    test_locations = get_locations(
        test
    )

    train_test_overlap = (
        train_locations
        & test_locations
    )

    train_val_overlap = (
        train_locations
        & validation_locations
    )

    validation_test_overlap = (
        validation_locations
        & test_locations
    )

    print()
    print(
        "Geographic leakage check:"
    )

    print(
        f"Train ↔ Test:       "
        f"{len(train_test_overlap)}"
    )

    print(
        f"Train ↔ Validation: "
        f"{len(train_val_overlap)}"
    )

    print(
        f"Validation ↔ Test:  "
        f"{len(validation_test_overlap)}"
    )

    # --------------------------------------------------------
    # Species overlap
    # --------------------------------------------------------

    if "species_full" in df.columns:

        train_species = set(
            train[
                "species_full"
            ]
            .dropna()
            .astype(str)
        )

        validation_species = set(
            validation[
                "species_full"
            ]
            .dropna()
            .astype(str)
        )

        test_species = set(
            test[
                "species_full"
            ]
            .dropna()
            .astype(str)
        )

        print()
        print(
            "Species overlap:"
        )

        print(
            f"Train species:      "
            f"{len(train_species)}"
        )

        print(
            f"Validation species: "
            f"{len(validation_species)}"
        )

        print(
            f"Test species:       "
            f"{len(test_species)}"
        )

        print(
            f"Validation ∩ Train: "
            f"{len(validation_species & train_species)}"
        )

        print(
            f"Test ∩ Train:       "
            f"{len(test_species & train_species)}"
        )

    # --------------------------------------------------------
    # Files
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("FILES CREATED")
    print("-" * 70)

    print()
    print(
        f"  {TRAIN_FILE}"
    )

    print(
        f"  {VALIDATION_FILE}"
    )

    print(
        f"  {TEST_FILE}"
    )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()