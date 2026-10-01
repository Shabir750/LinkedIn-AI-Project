import joblib
import pandas as pd


# ==========================================
# Paths
# ==========================================

MODEL_FILE = "models/multimodal_ridge.joblib"

TEST_FILE = "data/linkedin_ai_dataset/test.csv"

TEXT_FILE = "data/linkedin_ai_dataset/text_features.csv"


# ==========================================
# Load saved model
# ==========================================

model = joblib.load(MODEL_FILE)

print("Saved model loaded successfully!")


# ==========================================
# Load data
# ==========================================

test_df = pd.read_csv(TEST_FILE)

text_df = pd.read_csv(TEXT_FILE)


# ==========================================
# Prepare image features
# ==========================================

image_columns = [
    str(i)
    for i in range(512)
]

image_data = test_df[
    ["image_id"] + image_columns
].copy()


# ==========================================
# Prepare text features
# ==========================================

text_columns = [
    str(i)
    for i in range(512)
]

text_data = text_df[
    ["image_id"] + text_columns
].copy()


# ==========================================
# Rename text features
# ==========================================

# Image features remain:
# 0 ... 511
#
# Text features become:
# text_0 ... text_511

text_data.rename(
    columns={
        str(i): f"text_{i}"
        for i in range(512)
    },
    inplace=True
)


# ==========================================
# Merge image + text
# ==========================================

data = image_data.merge(
    text_data,
    on="image_id",
    how="inner"
)


# ==========================================
# Feature columns
# ==========================================

image_features = [
    str(i)
    for i in range(512)
]

text_features = [
    f"text_{i}"
    for i in range(512)
]

feature_columns = (
    image_features +
    text_features
)


# ==========================================
# Create X
# ==========================================

X = data[feature_columns]


print("Input shape:", X.shape)


# ==========================================
# Prediction
# ==========================================

predictions = model.predict(X)


# ==========================================
# Display predictions
# ==========================================

print("\nMULTIMODAL PREDICTIONS")
print("=" * 50)

for image_id, prediction in zip(
    data["image_id"],
    predictions
):

    print(
        f"Image ID: {image_id}   "
        f"Predicted Similarity: {prediction:.4f}"
    )