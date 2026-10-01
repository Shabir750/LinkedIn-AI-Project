import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/linkedin_ai_dataset")

TRAIN_FILE = DATA_DIR / "train.csv"
VALIDATION_FILE = DATA_DIR / "validation.csv"
TEST_FILE = DATA_DIR / "test.csv"


def check_dataset(name, file_path):
    df = pd.read_csv(file_path)

    print(f"\n{name}")
    print("-" * 40)

    print(f"Records: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    print(f"Unique image IDs: {df['image_id'].nunique()}")

    return df


def main():
    train_df = check_dataset("TRAIN DATASET", TRAIN_FILE)
    validation_df = check_dataset("VALIDATION DATASET", VALIDATION_FILE)
    test_df = check_dataset("TEST DATASET", TEST_FILE)

    # Check for overlapping image IDs
    train_ids = set(train_df["image_id"])
    validation_ids = set(validation_df["image_id"])
    test_ids = set(test_df["image_id"])

    print("\nChecking for overlap...")
    print("-" * 40)

    print(
        f"Train ∩ Validation: "
        f"{len(train_ids.intersection(validation_ids))}"
    )

    print(
        f"Train ∩ Test: "
        f"{len(train_ids.intersection(test_ids))}"
    )

    print(
        f"Validation ∩ Test: "
        f"{len(validation_ids.intersection(test_ids))}"
    )

    print("\nDataset split verification completed!")


if __name__ == "__main__":
    main()