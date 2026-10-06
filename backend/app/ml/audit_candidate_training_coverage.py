from pathlib import Path

import pandas as pd


BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_DIR = BACKEND_DIR.parent

TRAIN_PATH = (
    PROJECT_DIR
    / "data"
    / "survival_ml_final_train_groupsafe.csv"
)

TEST_PATH = (
    PROJECT_DIR
    / "data"
    / "survival_ml_final_test_groupsafe.csv"
)

FULL_PATH = (
    PROJECT_DIR
    / "data"
    / "survival_ml_final.csv"
)


CANDIDATES = [
    "Terminalia bellirica",
    "Neolamarckia cadamba",
    "Holoptelea integrifolia",
    "Ficus racemosa",
    "Gmelina arborea",
    "Careya arborea",
    "Syzygium cumini",
    "Bauhinia variegata",
    "Lagerstroemia speciosa",
    "Dalbergia latifolia",
    "Pongamia pinnata",
    "Alstonia scholaris",
    "Caryota urens",
    "Ficus hispida",
]


def normalize(value):
    return (
        str(value)
        .strip()
        .lower()
        .replace("  ", " ")
    )


train = pd.read_csv(
    TRAIN_PATH,
    low_memory=False,
)

test = pd.read_csv(
    TEST_PATH,
    low_memory=False,
)

full = pd.read_csv(
    FULL_PATH,
    low_memory=False,
)


train["_species_norm"] = (
    train["species_full"]
    .apply(normalize)
)

test["_species_norm"] = (
    test["species_full"]
    .apply(normalize)
)

full["_species_norm"] = (
    full["species_full"]
    .apply(normalize)
)


results = []


for species in CANDIDATES:

    key = normalize(species)

    train_rows = int(
        (train["_species_norm"] == key).sum()
    )

    test_rows = int(
        (test["_species_norm"] == key).sum()
    )

    full_rows = int(
        (full["_species_norm"] == key).sum()
    )


    if train_rows > 0:

        status = "TRAIN_SEEN"

    elif test_rows > 0:

        status = "TEST_ONLY"

    elif full_rows > 0:

        status = "FULL_DATA_ONLY"

    else:

        status = "NO_EXACT_SURVIVAL_NAME_MATCH"


    results.append(
        {
            "species": species,
            "train_rows": train_rows,
            "test_rows": test_rows,
            "full_rows": full_rows,
            "status": status,
        }
    )


result = pd.DataFrame(
    results
)


print("=" * 75)
print("CURRENT 14 CANDIDATES — SURVIVAL DATA AUDIT")
print("=" * 75)

print(
    result.to_string(
        index=False
    )
)


print("\n" + "=" * 75)
print("STATUS COUNTS")
print("=" * 75)

print(
    result["status"]
    .value_counts()
)


print(
    "\n✅ CANDIDATE COVERAGE AUDIT COMPLETE"
)