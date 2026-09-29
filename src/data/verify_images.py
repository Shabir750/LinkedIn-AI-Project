from PIL import Image
from pathlib import Path

IMAGE_DIR = Path("data/linkedin_ai_dataset/images")


def main():
    files = list(IMAGE_DIR.glob("*"))

    valid = 0
    invalid = 0

    for file in files:
        try:
            with Image.open(file) as image:
                image.verify()

            valid += 1

        except Exception as error:
            invalid += 1
            print(f"Invalid image: {file.name}")
            print(f"Error: {error}")

    print("\nVerification complete!")
    print(f"Total files: {len(files)}")
    print(f"Valid images: {valid}")
    print(f"Invalid images: {invalid}")


if __name__ == "__main__":
    main()