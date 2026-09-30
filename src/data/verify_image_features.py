import pandas as pd
from pathlib import Path

INPUT_FILE = Path(
    "data/linkedin_ai_dataset/image_features.csv"
)


def main():
    df = pd.read_csv(INPUT_FILE)

    print("Image feature verification")
    print("-------------------------")

    print(f"Total records: {len(df)}")
    print(f"Total columns: {len(df.columns)}")

    print("\nFirst column:")
    print(df.columns[0])

    print("\nFeature dimensions:")
    print(len(df.columns) - 1)

    print("\nMissing values:")
    print(df.isnull().sum().sum())

    print("\nFeature data types:")
    print(df.iloc[:, 1:].dtypes.value_counts())

    print("\nFirst image ID:")
    print(df["image_id"].iloc[0])

    print("\nFirst 10 feature values:")
    print(df.iloc[0, 1:11].tolist())

    print("\nVerification completed!")


if __name__ == "__main__":
    main()