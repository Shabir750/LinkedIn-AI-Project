import pandas as pd
import requests
from pathlib import Path

CSV_FILE = Path("data/linkedin_ai_dataset/multimodal.csv")
IMAGE_DIR = Path("data/linkedin_ai_dataset/images")
NUM_IMAGES = 100


def main():
    df = pd.read_csv(CSV_FILE)

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    for index, row in df.head(NUM_IMAGES).iterrows():

        image_url = row["image_url"]
        image_id = row["id"]

        output_file = IMAGE_DIR / f"{image_id}.jpg"

        try:
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()

            with open(output_file, "wb") as file:
                file.write(response.content)

            print(
                f"Downloaded {index + 1}/{NUM_IMAGES}: "
                f"{output_file.name}"
            )

        except Exception as error:
            print(f"Failed: {image_id}")
            print(f"Error: {error}")

    print("\nImage download completed.")


if __name__ == "__main__":
    main()