import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


SIMILARITY_FILE = Path(
    "data/linkedin_ai_dataset/image_text_similarity.csv"
)

OUTPUT_FILE = Path(
    "data/linkedin_ai_dataset/similarity_distribution.png"
)


def main():

    print("Loading similarity data...")

    df = pd.read_csv(SIMILARITY_FILE)

    similarity_scores = df["similarity_score"]

    print(f"Total scores: {len(similarity_scores)}")

    # Create histogram
    plt.figure(figsize=(10, 6))

    plt.hist(
        similarity_scores,
        bins=10,
        edgecolor="black"
    )

    plt.title("Image-Text Similarity Distribution")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Number of Samples")

    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    # Save figure
    plt.savefig(OUTPUT_FILE, dpi=150)

    print("\nSimilarity distribution plot created!")
    print(f"Output file: {OUTPUT_FILE}")

    # Display plot
    plt.show()


if __name__ == "__main__":
    main()