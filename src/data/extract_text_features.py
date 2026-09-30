import torch
import pandas as pd
from pathlib import Path
from transformers import CLIPProcessor, CLIPModel

INPUT_FILE = Path(
    "data/linkedin_ai_dataset/text_preprocessed.csv"
)

OUTPUT_FILE = Path(
    "data/linkedin_ai_dataset/text_features.csv"
)

MODEL_NAME = "openai/clip-vit-base-patch32"


def main():
    print("Loading CLIP model...")

    model = CLIPModel.from_pretrained(MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    model.eval()

    df = pd.read_csv(INPUT_FILE)

    features = []

    print(f"\nExtracting text features from {len(df)} captions...")

    with torch.no_grad():

        for index, row in df.iterrows():

            caption = str(row["clean_caption"])

            inputs = processor(
                text=caption,
                return_tensors="pt",
                padding=True,
                truncation=True
            )

            text_features = model.get_text_features(
                **inputs
            )

            text_features = (
                text_features.pooler_output
                .squeeze()
                .tolist()
            )

            features.append(text_features)

            print(f"Processed {index + 1}/{len(df)}")

    feature_df = pd.DataFrame(features)

    feature_df.insert(
        0,
        "image_id",
        df["image_id"].values
    )

    feature_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nText feature extraction completed!")
    print(f"Total captions: {len(feature_df)}")
    print(f"Feature dimensions: {feature_df.shape[1] - 1}")
    print(f"Output file: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()