import pandas as pd
from pathlib import Path


CSV_FILE = Path("data/linkedin_ai_dataset/multimodal.csv")
IMAGE_DIR = Path("data/linkedin_ai_dataset/images")
OUTPUT_FILE = Path("data/linkedin_ai_dataset/prepared_dataset.csv")


def main():
    # Load metadata
    df = pd.read_csv(CSV_FILE)

    records = []

    for _, row in df.iterrows():

        image_id = str(row["id"])
        image_file = IMAGE_DIR / f"{image_id}.jpg"

        # Only include images that actually exist
        if image_file.exists():
            records.append({
                "image_id": image_id,
                "image_path": str(image_file),
                "caption": row["caption"],
                "width": row["width"],
                "height": row["height"],
                "mime_type": row["mime_type"],
                "license": row["license"],
                "source": row["source"]
            })

    # Create final dataset
    prepared_df = pd.DataFrame(records)

    # Save dataset
    prepared_df.to_csv(OUTPUT_FILE, index=False)

    print("\nDataset preparation completed!")
    print(f"Total matched images: {len(prepared_df)}")
    print(f"Output file: {OUTPUT_FILE}")

    print("\nColumns:")
    print(prepared_df.columns.tolist())


if __name__ == "__main__":
    main()