import pandas as pd
from pathlib import Path

IMAGE_FEATURES_FILE = Path(
    "data/linkedin_ai_dataset/image_features.csv"
)

TEXT_FEATURES_FILE = Path(
    "data/linkedin_ai_dataset/encoded_captions.csv"
)

OUTPUT_FILE = Path(
    "data/linkedin_ai_dataset/multimodal_features.csv"
)


def main():
    print("Loading image features...")
    image_df = pd.read_csv(IMAGE_FEATURES_FILE)

    print("Loading text features...")
    text_df = pd.read_csv(TEXT_FEATURES_FILE)

    print(f"\nImage records: {len(image_df)}")
    print(f"Text records: {len(text_df)}")

    # Keep the useful text columns
    text_df = text_df[
        [
            "image_id",
            "clean_caption",
            "input_ids",
            "attention_mask"
        ]
    ]

    # Merge using image_id
    multimodal_df = pd.merge(
        image_df,
        text_df,
        on="image_id",
        how="inner"
    )

    multimodal_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nMultimodal dataset created!")
    print(f"Total records: {len(multimodal_df)}")
    print(f"Total columns: {len(multimodal_df.columns)}")
    print(f"Output file: {OUTPUT_FILE}")

    print("\nColumns:")
    print(multimodal_df.columns.tolist())


if __name__ == "__main__":
    main()