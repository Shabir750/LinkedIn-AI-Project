import pandas as pd
from pathlib import Path

INPUT_FILE = Path("data/linkedin_ai_dataset/tokenized_captions.csv")


def main():
    df = pd.read_csv(INPUT_FILE)

    print("Tokenization verification")
    print("-------------------------")

    print(f"Total records: {len(df)}")
    print(f"Total columns: {len(df.columns)}")

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nMissing values:")
    print(df[["clean_caption", "tokens"]].isnull().sum())

    print("\nFirst record:")
    print("Clean caption:")
    print(df["clean_caption"].iloc[0])

    print("\nTokens:")
    print(df["tokens"].iloc[0])

    print("\nVerification completed!")


if __name__ == "__main__":
    main()