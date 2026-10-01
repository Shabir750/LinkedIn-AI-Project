import torch
import pandas as pd
from pathlib import Path
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import joblib


# -----------------------------
# File paths
# -----------------------------

IMAGE_FILE = Path(
    "data/user_input/image.jpg"
)

TEXT_FILE = Path(
    "data/user_input/input.txt"
)

MODEL_FILE = Path(
    "models/multimodal_ridge.joblib"
)


# -----------------------------
# CLIP model
# -----------------------------

MODEL_NAME = "openai/clip-vit-base-patch32"


def main():

    print("Loading CLIP model...")

    clip_model = CLIPModel.from_pretrained(MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    clip_model.eval()


    # -----------------------------
    # Load image
    # -----------------------------

    print("\nLoading user image...")

    image = Image.open(IMAGE_FILE).convert("RGB")


    # -----------------------------
    # Extract image features
    # -----------------------------

    print("Extracting image features...")

    with torch.no_grad():

        image_inputs = processor(
            images=image,
            return_tensors="pt"
        )

        image_features = clip_model.get_image_features(
            **image_inputs
        )

        image_features = (
            image_features.pooler_output
            .squeeze()
            .tolist()
        )


    print(f"Image features: {len(image_features)}")


    # -----------------------------
    # Load user text
    # -----------------------------

    print("\nLoading user text...")

    with open(TEXT_FILE, "r", encoding="utf-8") as file:
        text = file.read().strip()


    # -----------------------------
    # Extract text features
    # -----------------------------

    print("Extracting text features...")

    with torch.no_grad():

        text_inputs = processor(
            text=text,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        text_features = clip_model.get_text_features(
            **text_inputs
        )

        text_features = (
            text_features.pooler_output
            .squeeze()
            .tolist()
        )


    print(f"Text features: {len(text_features)}")


    # -----------------------------
    # Load trained model
    # -----------------------------

    print("\nLoading trained model...")

    model = joblib.load(MODEL_FILE)

    print("Saved model loaded successfully!")


    # -----------------------------
    # Create feature dataframe
    # -----------------------------

    image_columns = [
        str(i) for i in range(512)
    ]

    text_columns = [
        f"text_{i}" for i in range(512)
    ]


    image_df = pd.DataFrame(
        [image_features],
        columns=image_columns
    )

    text_df = pd.DataFrame(
        [text_features],
        columns=text_columns
    )


    # -----------------------------
    # Combine image + text
    # -----------------------------

    input_data = pd.concat(
        [image_df, text_df],
        axis=1
    )


    print(
        f"\nInput shape: {input_data.shape}"
    )


    # -----------------------------
    # Prediction
    # -----------------------------

    prediction = model.predict(input_data)[0]


    print("\n" + "=" * 50)
    print("NEW USER INPUT PREDICTION")
    print("=" * 50)

    print(f"Text: {text}")

    print(
        f"Predicted similarity: {prediction:.4f}"
    )


if __name__ == "__main__":
    main()