import pandas as pd
import numpy as np
from pathlib import Path

IMAGE_FILE = Path(
    "data/linkedin_ai_dataset/image_features.csv"
)

TEXT_FILE = Path(
    "data/linkedin_ai_dataset/text_features.csv"
)

OUTPUT_FILE = Path(
    "data/linkedin_ai_dataset/image_text_similarity.csv"
)


def cosine_similarity(vector_a, vector_b):
    vector_a = np.array(vector_a)
    vector_b = np.array(vector_b)

    numerator = np.dot(vector_a, vector_b)

    denominator = (
        np.linalg.norm(vector_a)
        * np.linalg.norm(vector_b)
    )

    if denominator == 0:
        return 0.0

    return numerator / denominator


def main():
    print("Loading image features...")
    image_df = pd.read_csv(IMAGE_FILE)

    print("Loading text features...")
    text_df = pd.read_csv(TEXT_FILE)

    print(f"\nImage records: {len(image_df)}")
    print(f"Text records: {len(text_df)}")

    similarities = []

    for index in range(len(image_df)):

        image_vector = image_df.iloc[index, 1:].values
        text_vector = text_df.iloc[index, 1:].values

        score = cosine_similarity(
            image_vector,
            text_vector
        )

        similarities.append(score)

    result_df = pd.DataFrame({
        "image_id": image_df["image_id"],
        "similarity_score": similarities
    })

    result_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nImage-text similarity calculation completed!")

    print(f"Total pairs: {len(result_df)}")

    print(
        f"Average similarity: "
        f"{result_df['similarity_score'].mean():.4f}"
    )

    print(
        f"Minimum similarity: "
        f"{result_df['similarity_score'].min():.4f}"
    )

    print(
        f"Maximum similarity: "
        f"{result_df['similarity_score'].max():.4f}"
    )

    print(f"Output file: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()