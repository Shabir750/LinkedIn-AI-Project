from datasets import load_dataset
import pandas as pd
from pathlib import Path


DATASET_NAME = "Spawning/PD12M"
NUM_SAMPLES = 100

OUTPUT_DIR = Path("data/linkedin_ai_dataset")
OUTPUT_FILE = OUTPUT_DIR / "multimodal.csv"


def main():

    print("Connecting to PD12M...")

    dataset = load_dataset(
        DATASET_NAME,
        split="train",
        streaming=True
    )

    print("Connection successful!")

    records = []

    for index, sample in enumerate(dataset):

        if index >= NUM_SAMPLES:
            break

        records.append({
            "id": sample["id"],
            "caption": sample["caption"],
            "image_url": sample["url"],
            "width": sample["width"],
            "height": sample["height"],
            "mime_type": sample["mime_type"],
            "license": sample["license"],
            "source": sample["source"]
        })

        print(f"Collected sample {index + 1}/{NUM_SAMPLES}")

    # Create output directory
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Convert to DataFrame
    df = pd.DataFrame(records)

    # Save CSV
    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nDataset created successfully!")
    print(f"Rows: {len(df)}")
    print(f"File: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()