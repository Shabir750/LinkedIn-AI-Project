import pandas as pd
from pathlib import Path


SIMILARITY_FILE = Path(
    "data/linkedin_ai_dataset/image_text_similarity.csv"
)


def main():

    print("Loading similarity data...")

    df = pd.read_csv(SIMILARITY_FILE)

    print(f"\nTotal records: {len(df)}")

    # Basic statistics
    print("\nSimilarity Statistics")
    print("---------------------")

    print(
        f"Average: {df['similarity_score'].mean():.4f}"
    )

    print(
        f"Minimum: {df['similarity_score'].min():.4f}"
    )

    print(
        f"Maximum: {df['similarity_score'].max():.4f}"
    )

    print(
        f"Median: {df['similarity_score'].median():.4f}"
    )

    # Top 10 matches
    print("\nTop 10 Image-Text Matches")
    print("-------------------------")

    top_matches = df.sort_values(
        "similarity_score",
        ascending=False
    ).head(10)

    print(
        top_matches[
            ["image_id", "similarity_score"]
        ].to_string(index=False)
    )

    # Bottom 10 matches
    print("\nBottom 10 Image-Text Matches")
    print("----------------------------")

    bottom_matches = df.sort_values(
        "similarity_score",
        ascending=True
    ).head(10)

    print(
        bottom_matches[
            ["image_id", "similarity_score"]
        ].to_string(index=False)
    )

    print("\nSimilarity analysis completed!")


if __name__ == "__main__":
    main()