import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/linkedin_ai_dataset")

TRAIN_FILE = DATA_DIR / "train.csv"
VALIDATION_FILE = DATA_DIR / "validation.csv"
TEST_FILE = DATA_DIR / "test.csv"


def prepare_data(file_path):
    df = pd.read_csv(file_path)

    # Our 512 image features are columns 0 to 511
    image_feature_columns = [str(i) for i in range(512)]

    # X = 512 image features
    X = df[image_feature_columns]

    # y = similarity score
    y = df["similarity_score"]

    return X, y


def main():
    X_train, y_train = prepare_data(TRAIN_FILE)
    X_validation, y_validation = prepare_data(VALIDATION_FILE)
    X_test, y_test = prepare_data(TEST_FILE)

    print("MODEL DATA PREPARATION")
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

    print("\nTarget:")
    print("Target column: similarity_score")

    print("\nModel data preparation completed!")


if __name__ == "__main__":
    main()