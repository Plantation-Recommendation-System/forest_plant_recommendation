from pathlib import Path

import pandas as pd


BACKEND_DIR = Path(__file__).resolve().parents[2]

PROJECT_DIR = BACKEND_DIR.parent


TRAIN_PATH = (
    PROJECT_DIR
    / "data"
    / "survival_ml_final_train_groupsafe.csv"
)

OUTPUT_PATH = (
    BACKEND_DIR
    / "data"
    / "catboost_seen_species.csv"
)


print("=" * 60)
print("CATBOOST TRAINING-SPECIES AUDIT")
print("=" * 60)


# =========================================================
# 1. CHECK TRAIN FILE
# =========================================================

print(
    "Training file:",
    TRAIN_PATH
)

print(
    "Exists:",
    TRAIN_PATH.exists()
)


if not TRAIN_PATH.exists():

    raise FileNotFoundError(
        f"Training file not found: {TRAIN_PATH}"
    )


# =========================================================
# 2. LOAD
# =========================================================

train = pd.read_csv(
    TRAIN_PATH,
    low_memory=False,
)


print(
    "Training rows:",
    len(train)
)

print(
    "Columns:",
    train.columns.tolist()
)


# =========================================================
# 3. CHECK SPECIES COLUMN
# =========================================================

if "species_full" not in train.columns:

    raise ValueError(
        "species_full column was not found."
    )


# =========================================================
# 4. UNIQUE SPECIES SEEN DURING TRAINING
# =========================================================

seen = (

    train[
        ["species_full"]
    ]

    .dropna()

    .assign(
        species_full=lambda x:
            x[
                "species_full"
            ]
            .astype(str)
            .str.strip()
    )

    .query(
        "species_full != ''"
    )

    .drop_duplicates()

    .sort_values(
        "species_full"
    )

    .reset_index(
        drop=True
    )
)


seen = seen.rename(
    columns={
        "species_full":
            "project_species_resolved"
    }
)


seen[
    "seen_in_catboost_training"
] = True


# =========================================================
# 5. SAVE
# =========================================================

seen.to_csv(
    OUTPUT_PATH,
    index=False,
)


print("\n" + "=" * 60)
print("RESULT")
print("=" * 60)

print(
    "Unique species seen:",
    len(seen)
)

print(
    "Saved:",
    OUTPUT_PATH.exists()
)

print(
    "Output:",
    OUTPUT_PATH
)


print(
    "\n✅ CATBOOST TRAINING-SPECIES FILE READY"
)