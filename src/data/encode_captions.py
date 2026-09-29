import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer

INPUT_FILE = Path("data/linkedin_ai_dataset/text_preprocessed.csv")
OUTPUT_FILE = Path("data/linkedin_ai_dataset/encoded_captions.csv")

MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128


def main():
    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    df = pd.read_csv(INPUT_FILE)

    input_ids = []
    attention_masks = []

    for caption in df["clean_caption"]:
        encoded = tokenizer(
            str(caption),
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH
        )

        input_ids.append(encoded["input_ids"])
        attention_masks.append(encoded["attention_mask"])

    df["input_ids"] = input_ids
    df["attention_mask"] = attention_masks

    df.to_csv(OUTPUT_FILE, index=False)

    print("\nEncoding completed!")
    print(f"Total records: {len(df)}")
    print(f"Maximum sequence length: {MAX_LENGTH}")
    print(f"Output file: {OUTPUT_FILE}")

    print("\nExample:")
    print("Caption:")
    print(df["clean_caption"].iloc[0])

    print("\nInput IDs:")
    print(df["input_ids"].iloc[0])

    print("\nAttention Mask:")
    print(df["attention_mask"].iloc[0])


if __name__ == "__main__":
    main()