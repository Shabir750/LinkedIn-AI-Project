import pandas as pd
from pathlib import Path


TRAINING_FILE = Path(
    "data/linkedin_ai_dataset/training_dataset.csv"
)


def main():

    print("Loading training dataset...")

    df = pd.read_csv(TRAINING_FILE)

    print(f"\nTotal records: {len(df)}")
    print(f"Total columns: {len(df.columns)}")

    # Check image IDs
    print(
        f"Unique image IDs: "
        f"{df['image_id'].nunique()}"
    )

    # Check missing values
    missing_values = df.isnull().sum().sum()

    print(
        f"Missing values: {missing_values}"
    )

    # Check duplicate rows
    duplicates = df.duplicated().sum()

    print(
        f"Duplicate rows: {duplicates}"
    )

    # Check similarity score
    print(
        f"\nAverage similarity: "
        f"{df['similarity_score'].mean():.4f}"
    )

    print(
        f"Minimum similarity: "
        f"{df['similarity_score'].min():.4f}"
    )

    print(
        f"Maximum similarity: "
        f"{df['similarity_score'].max():.4f}"
    )

    print("\nTraining dataset verification completed!")


if __name__ == "__main__":
    main()