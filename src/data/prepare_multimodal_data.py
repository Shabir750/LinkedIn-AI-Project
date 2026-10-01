import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/linkedin_ai_dataset")

TRAIN_FILE = DATA_DIR / "train.csv"
VALIDATION_FILE = DATA_DIR / "validation.csv"
TEST_FILE = DATA_DIR / "test.csv"

TEXT_FEATURE_FILE = DATA_DIR / "text_features.csv"


def prepare_data(split_file, text_feature_file):

    # Load split dataset
    split_df = pd.read_csv(split_file)

    # Load CLIP text features
    text_df = pd.read_csv(text_feature_file)

    # Original image feature columns
    image_feature_columns = [str(i) for i in range(512)]

    # Rename image features
    image_rename = {
        str(i): f"image_feature_{i}"
        for i in range(512)
    }

    split_df = split_df.rename(columns=image_rename)

    # Rename text features
    text_rename = {
        str(i): f"text_feature_{i}"
        for i in range(512)
    }

    text_df = text_df.rename(columns=text_rename)

    # Text feature column names
    text_feature_columns = [
        f"text_feature_{i}"
        for i in range(512)
    ]

    # Image feature column names
    image_feature_columns = [
        f"image_feature_{i}"
        for i in range(512)
    ]

    # Keep required columns
    image_df = split_df[
        ["image_id"] +
        image_feature_columns +
        ["similarity_score"]
    ]

    text_df = text_df[
        ["image_id"] +
        text_feature_columns
    ]

    # Merge image and text features
    merged_df = image_df.merge(
        text_df,
        on="image_id",
        how="inner"
    )

    # Combine both modalities
    feature_columns = (
        image_feature_columns +
        text_feature_columns
    )

    X = merged_df[feature_columns]

    # Target
    y = merged_df["similarity_score"]

    return X, y


def main():

    X_train, y_train = prepare_data(
        TRAIN_FILE,
        TEXT_FEATURE_FILE
    )

    X_validation, y_validation = prepare_data(
        VALIDATION_FILE,
        TEXT_FEATURE_FILE
    )

    X_test, y_test = prepare_data(
        TEST_FILE,
        TEXT_FEATURE_FILE
    )

    print("MULTIMODAL DATA PREPARATION")
    print("=" * 50)

    print("\nTraining:")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")

    print("\nValidation:")
    print(f"X_validation shape: {X_validation.shape}")
    print(f"y_validation shape: {y_validation.shape}")

    print("\nTest:")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    print("\nFeatures:")
    print("Image features: 512")
    print("Text features: 512")
    print("Total features: 1024")

    print("\nMultimodal data preparation completed!")


if __name__ == "__main__":
    main()