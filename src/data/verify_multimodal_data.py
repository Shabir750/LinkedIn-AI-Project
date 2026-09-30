import pandas as pd
from pathlib import Path

INPUT_FILE = Path(
    "data/linkedin_ai_dataset/multimodal_features.csv"
)


def main():
    df = pd.read_csv(INPUT_FILE)

    print("Multimodal dataset verification")
    print("--------------------------------")

    print(f"Total records: {len(df)}")
    print(f"Total columns: {len(df.columns)}")

    print("\nRequired columns:")

    required_columns = [
        "image_id",
        "clean_caption",
        "input_ids",
        "attention_mask"
    ]

    for column in required_columns:
        print(f"{column}: {column in df.columns}")

    print("\nImage feature count:")
    print(len(df.columns) - 4)

    print("\nMissing values:")
    print(df.isnull().sum().sum())

    print("\nDuplicate image IDs:")
    print(df["image_id"].duplicated().sum())

    print("\nFirst record:")
    print(df.iloc[0][
        [
            "image_id",
            "clean_caption"
        ]
    ])

    print("\nVerification completed!")


if __name__ == "__main__":
    main()