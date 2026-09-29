from PIL import Image
from pathlib import Path
import pandas as pd


DATASET_FILE = Path(
    "data/linkedin_ai_dataset/prepared_dataset.csv"
)

OUTPUT_DIR = Path(
    "data/linkedin_ai_dataset/processed_images"
)

IMAGE_SIZE = (224, 224)


def main():

    # Load the prepared dataset
    df = pd.read_csv(DATASET_FILE)

    # Make sure output folder exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    processed = 0
    failed = 0

    for _, row in df.iterrows():

        image_id = str(row["image_id"])
        image_path = Path(row["image_path"])

        output_path = OUTPUT_DIR / f"{image_id}.jpg"

        try:
            # Open original image
            image = Image.open(image_path)

            # Convert to RGB
            image = image.convert("RGB")

            # Resize
            image = image.resize(IMAGE_SIZE)

            # Save processed image
            image.save(output_path, "JPEG")

            processed += 1

            print(
                f"Processed {processed}/{len(df)}: {image_id}"
            )

        except Exception as error:

            failed += 1

            print(
                f"Failed: {image_id} | Error: {error}"
            )

    print("\nImage preprocessing completed!")
    print(f"Processed images: {processed}")
    print(f"Failed images: {failed}")
    print(f"Image size: {IMAGE_SIZE}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()