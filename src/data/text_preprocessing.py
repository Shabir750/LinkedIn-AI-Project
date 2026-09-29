import pandas as pd
import re
from pathlib import Path


INPUT_FILE = Path(
    "data/linkedin_ai_dataset/prepared_dataset.csv"
)

OUTPUT_FILE = Path(
    "data/linkedin_ai_dataset/text_preprocessed.csv"
)


def clean_text(text):
    """Clean a caption."""

    # Convert to string
    text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    # Remove leading/trailing spaces
    text = text.strip()

    return text


def main():

    # Load dataset
    df = pd.read_csv(INPUT_FILE)

    # Keep original caption
    df["original_caption"] = df["caption"]

    # Clean caption
    df["clean_caption"] = df["caption"].apply(clean_text)

    # Save processed dataset
    df.to_csv(OUTPUT_FILE, index=False)

    print("\nText preprocessing completed!")
    print(f"Total records: {len(df)}")
    print(f"Output file: {OUTPUT_FILE}")

    print("\nExample:")
    print("Original:")
    print(df["original_caption"].iloc[0])

    print("\nCleaned:")
    print(df["clean_caption"].iloc[0])


if __name__ == "__main__":
    main()