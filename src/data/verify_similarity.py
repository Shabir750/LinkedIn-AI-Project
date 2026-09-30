import pandas as pd
from pathlib import Path


SIMILARITY_FILE = Path(
    "data/linkedin_ai_dataset/image_text_similarity.csv"
)


def main():

    print("Loading similarity data...")

    df = pd.read_csv(SIMILARITY_FILE)

    print(f"\nTotal records: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    # Check required columns
    required_columns = [
        "image_id",
        "similarity_score"
    ]

    for column in required_columns:
        if column in df.columns:
            print(f"✓ {column} column exists")
        else:
            print(f"✗ {column} column missing")

    # Check missing values
    missing_values = df.isnull().sum().sum()

    print(f"\nMissing values: {missing_values}")

    # Check duplicate image IDs
    duplicates = df["image_id"].duplicated().sum()

    print(f"Duplicate image IDs: {duplicates}")

    # Check similarity score range
    min_score = df["similarity_score"].min()
    max_score = df["similarity_score"].max()

    print(f"Minimum similarity: {min_score:.4f}")
    print(f"Maximum similarity: {max_score:.4f}")

    # Check numeric values
    numeric_check = pd.api.types.is_numeric_dtype(
        df["similarity_score"]
    )

    print(f"Similarity scores numeric: {numeric_check}")

    print("\nSimilarity verification completed!")


if __name__ == "__main__":
    main()