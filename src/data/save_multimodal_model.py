import os
import joblib
import pandas as pd
from sklearn.linear_model import Ridge


# ==========================================
# Paths
# ==========================================

DATA_DIR = "data/linkedin_ai_dataset"
MODEL_DIR = "models"

TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
TEXT_FILE = os.path.join(DATA_DIR, "text_features.csv")

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "multimodal_ridge.joblib"
)


# ==========================================
# Load datasets
# ==========================================

train_df = pd.read_csv(TRAIN_FILE)
text_df = pd.read_csv(TEXT_FILE)

print("Train data:", train_df.shape)
print("Text features:", text_df.shape)


# ==========================================
# Image features
# ==========================================

image_feature_columns = [str(i) for i in range(512)]

X_image = train_df[image_feature_columns]


# ==========================================
# Text features
# ==========================================

text_feature_columns = [str(i) for i in range(512)]

X_text = text_df[text_feature_columns].copy()

# Rename text features so they don't conflict
# with image feature names.

X_text.columns = [
    f"text_{i}" for i in range(512)
]


# ==========================================
# Match text features with training records
# ==========================================

text_df = text_df[["image_id"] + text_feature_columns]

text_df.columns = (
    ["image_id"] +
    [f"text_{i}" for i in range(512)]
)


# Merge using image_id

merged_df = train_df[["image_id"] + image_feature_columns + [
    "similarity_score"
]].merge(
    text_df,
    on="image_id",
    how="inner"
)


# ==========================================
# Create final feature matrix
# ==========================================

text_feature_columns = [
    f"text_{i}" for i in range(512)
]

feature_columns = (
    image_feature_columns +
    text_feature_columns
)

X_train = merged_df[feature_columns]
y_train = merged_df["similarity_score"]


# ==========================================
# Verify data
# ==========================================

print("\nMULTIMODAL TRAINING DATA")
print("=" * 50)

print("Records:", len(X_train))
print("Image features:", len(image_feature_columns))
print("Text features:", len(text_feature_columns))
print("Total features:", len(feature_columns))

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)


# ==========================================
# Train final Ridge model
# ==========================================

model = Ridge(alpha=100)

model.fit(X_train, y_train)


# ==========================================
# Save model
# ==========================================

os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(model, MODEL_FILE)


print("\nFINAL MODEL")
print("=" * 50)
print("Model: Ridge Regression")
print("Alpha: 100")
print("Features: 1024")
print("Model saved to:")
print(MODEL_FILE)