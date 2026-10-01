import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

INPUT_FILE = Path("data/linkedin_ai_dataset/training_dataset.csv")

TRAIN_FILE = Path("data/linkedin_ai_dataset/train.csv")
VALIDATION_FILE = Path("data/linkedin_ai_dataset/validation.csv")
TEST_FILE = Path("data/linkedin_ai_dataset/test.csv")


def main():
    print("Loading training dataset...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Total records: {len(df)}")

    # 70% training, 30% temporary
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=42,
        shuffle=True
    )

    # Split remaining 30% equally:
    # 15% validation, 15% test
    validation_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        shuffle=True
    )

    # Save datasets
    train_df.to_csv(TRAIN_FILE, index=False)
    validation_df.to_csv(VALIDATION_FILE, index=False)
    test_df.to_csv(TEST_FILE, index=False)

    print(f"Training records: {len(train_df)}")
    print(f"Validation records: {len(validation_df)}")
    print(f"Test records: {len(test_df)}")

    print("\nDataset split completed!")


if __name__ == "__main__":
    main()