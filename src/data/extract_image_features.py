import torch
import pandas as pd
from pathlib import Path
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

DATASET_FILE = Path(
    "data/linkedin_ai_dataset/prepared_dataset.csv"
)

OUTPUT_FILE = Path(
    "data/linkedin_ai_dataset/image_features.csv"
)

MODEL_NAME = "openai/clip-vit-base-patch32"


def main():
    print("Loading CLIP model...")

    model = CLIPModel.from_pretrained(MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    model.eval()

    df = pd.read_csv(DATASET_FILE)

    features = []

    print(f"\nExtracting features from {len(df)} images...")
     
    with torch.no_grad():

      for index, row in df.iterrows():

        image_path = Path(row["image_path"])

        image = Image.open(image_path).convert("RGB")

        inputs = processor(
            images=image,
            return_tensors="pt"
        )

        image_features = model.get_image_features(
            **inputs
        )

        image_features = image_features.pooler_output.squeeze().tolist()

        features.append(image_features)

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

    print("\nImage feature extraction completed!")
    print(f"Total images: {len(feature_df)}")
    print(f"Feature dimensions: {feature_df.shape[1] - 1}")
    print(f"Output file: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()