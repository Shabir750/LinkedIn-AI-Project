import pandas as pd
from pathlib import Path


TRAINING_FILE = Path(
    "data/linkedin_ai_dataset/training_dataset.csv"
)


def main():

    print("Loading training dataset...")

    df = pd.read_csv(TRAINING_FILE)

    # Top 5 image-text matches
    print("\nTop 5 Image-Text Matches")
    print("========================")

    top = df.sort_values(
        "similarity_score",
        ascending=False
    ).head(5)

    print(
        top[
            [
                "image_id",
                "clean_caption",
                "similarity_score"
            ]
        ].to_string(index=False)
    )

    # Bottom 5 image-text matches
    print("\nBottom 5 Image-Text Matches")
    print("===========================")

    bottom = df.sort_values(
        "similarity_score",
        ascending=True
    ).head(5)

    print(
        bottom[
            [
                "image_id",
                "clean_caption",
                "similarity_score"
            ]
        ].to_string(index=False)
    )

    print("\nTraining data inspection completed!")


if __name__ == "__main__":
    main()