import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error


# Load test data
test_df = pd.read_csv(
    "data/linkedin_ai_dataset/test.csv"
)

text_features = pd.read_csv(
    "data/linkedin_ai_dataset/text_features.csv"
)


# --------------------------------------------------
# Prepare image features
# --------------------------------------------------

image_columns = [str(i) for i in range(512)]

test_image = test_df[
    ["image_id"] + image_columns
].copy()

test_image.rename(
    columns={
        str(i): f"image_feature_{i}"
        for i in range(512)
    },
    inplace=True
)


# --------------------------------------------------
# Prepare text features
# --------------------------------------------------

text_features.rename(
    columns={
        str(i): f"text_feature_{i}"
        for i in range(512)
    },
    inplace=True
)


# --------------------------------------------------
# We also need training data
# --------------------------------------------------

train_df = pd.read_csv(
    "data/linkedin_ai_dataset/train.csv"
)

train_image = train_df[
    ["image_id"] + image_columns
].copy()

train_image.rename(
    columns={
        str(i): f"image_feature_{i}"
        for i in range(512)
    },
    inplace=True
)


# --------------------------------------------------
# Merge image + text features
# --------------------------------------------------

train_data = train_image.merge(
    text_features,
    on="image_id",
    how="inner"
)

test_data = test_image.merge(
    text_features,
    on="image_id",
    how="inner"
)


# --------------------------------------------------
# Feature columns
# --------------------------------------------------

image_features = [
    f"image_feature_{i}"
    for i in range(512)
]

text_features_columns = [
    f"text_feature_{i}"
    for i in range(512)
]

feature_columns = (
    image_features +
    text_features_columns
)


# --------------------------------------------------
# X and y
# --------------------------------------------------

X_train = train_data[feature_columns]

y_train = train_df["similarity_score"]

X_test = test_data[feature_columns]

y_test = test_df["similarity_score"]


# --------------------------------------------------
# Train best model
# --------------------------------------------------

model = Ridge(alpha=100)

model.fit(
    X_train,
    y_train
)


# --------------------------------------------------
# Predictions
# --------------------------------------------------

predictions = model.predict(
    X_test
)


# --------------------------------------------------
# MAE
# --------------------------------------------------

mae = mean_absolute_error(
    y_test,
    predictions
)

print("MULTIMODAL RIDGE PREDICTION ANALYSIS")
print("=" * 50)

print(f"Test MAE: {mae:.4f}")


# --------------------------------------------------
# Print actual vs predicted
# --------------------------------------------------

print("\nACTUAL vs PREDICTED")
print("=" * 50)

for actual, predicted in zip(
    y_test,
    predictions
):

    print(
        f"Actual: {actual:.4f}   "
        f"Predicted: {predicted:.4f}"
    )


# --------------------------------------------------
# Plot
# --------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    predictions
)

# Perfect prediction line
min_value = min(
    y_test.min(),
    predictions.min()
)

max_value = max(
    y_test.max(),
    predictions.max()
)

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.xlabel(
    "Actual Similarity"
)

plt.ylabel(
    "Predicted Similarity"
)

plt.title(
    "Multimodal Ridge: Actual vs Predicted"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "data/linkedin_ai_dataset/"
    "multimodal_actual_vs_predicted.png"
)

plt.show()


print(
    "\nPlot saved successfully!"
)