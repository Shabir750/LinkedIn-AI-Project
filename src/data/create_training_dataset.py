import pandas as pd
from pathlib import Path


MULTIMODAL_FILE = Path(
    "data/linkedin_ai_dataset/multimodal_features.csv"
)

SIMILARITY_FILE = Path(
    "data/linkedin_ai_dataset/image_text_similarity.csv"
)

OUTPUT_FILE = Path(
    "data/linkedin_ai_dataset/training_dataset.csv"
)


def main():

    print("Loading multimodal features...")
    multimodal_df = pd.read_csv(MULTIMODAL_FILE)

    print("Loading similarity scores...")
    similarity_df = pd.read_csv(SIMILARITY_FILE)

    print(f"\nMultimodal records: {len(multimodal_df)}")
    print(f"Similarity records: {len(similarity_df)}")

    # Merge using image_id
    training_df = multimodal_df.merge(
        similarity_df,
        on="image_id",
        how="inner"
    )

    print(
        f"\nTraining dataset records: "
        f"{len(training_df)}"
    )

    print(
        f"Training dataset columns: "
        f"{len(training_df.columns)}"
    )

    # Save dataset
    training_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nTraining dataset created successfully!")
    print(f"Output file: {OUTPUT_FILE}")

    # Display first few rows
    print("\nFirst 5 records:")
    print(training_df.head())


if __name__ == "__main__":
    main()