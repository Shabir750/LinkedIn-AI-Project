import pandas as pd
from pathlib import Path
import ast

INPUT_FILE = Path(
    "data/linkedin_ai_dataset/encoded_captions.csv"
)


def main():
    df = pd.read_csv(INPUT_FILE)

    print("Encoding verification")
    print("---------------------")

    print(f"Total records: {len(df)}")

    print("\nRequired columns:")
    required_columns = [
        "clean_caption",
        "input_ids",
        "attention_mask"
    ]

    for column in required_columns:
        print(f"{column}: {column in df.columns}")

    print("\nMissing values:")
    print(df[required_columns].isnull().sum())

    # Convert stored strings back to Python lists
    ids = ast.literal_eval(df["input_ids"].iloc[0])
    mask = ast.literal_eval(df["attention_mask"].iloc[0])

    print("\nFirst record:")
    print("Caption:")
    print(df["clean_caption"].iloc[0])

    print("\nInput IDs:")
    print(ids)

    print("\nAttention Mask:")
    print(mask)

    print("\nSequence length:")
    print(len(ids))

    print("\nAttention mask length:")
    print(len(mask))

    print("\nVerification completed!")


if __name__ == "__main__":
    main()