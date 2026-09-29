import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer

INPUT_FILE = Path("data/linkedin_ai_dataset/text_preprocessed.csv")
OUTPUT_FILE = Path("data/linkedin_ai_dataset/tokenized_captions.csv")
MODEL_NAME = "bert-base-uncased"


def main():
    df = pd.read_csv(INPUT_FILE)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    tokenized_data = []

    for caption in df["clean_caption"]:
        tokens = tokenizer.tokenize(str(caption))
        tokenized_data.append(" ".join(tokens))

    df["tokens"] = tokenized_data

    df.to_csv(OUTPUT_FILE, index=False)

    print("\nTokenization completed!")
    print(f"Total records: {len(df)}")
    print(f"Output file: {OUTPUT_FILE}")

    print("\nExample:")
    print("Caption:")
    print(df["clean_caption"].iloc[0])

    print("\nTokens:")
    print(df["tokens"].iloc[0])


if __name__ == "__main__":
    main()