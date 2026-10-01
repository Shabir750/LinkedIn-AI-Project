import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load datasets
train_df = pd.read_csv(
    "data/linkedin_ai_dataset/train.csv"
)

test_df = pd.read_csv(
    "data/linkedin_ai_dataset/test.csv"
)

text_features = pd.read_csv(
    "data/linkedin_ai_dataset/text_features.csv"
)


# --------------------------------------------------
# Prepare image features
# --------------------------------------------------

image_feature_columns = [str(i) for i in range(512)]

train_image = train_df[
    ["image_id"] + image_feature_columns
].copy()

test_image = test_df[
    ["image_id"] + image_feature_columns
].copy()


# Rename image feature columns
rename_image_columns = {
    str(i): f"image_feature_{i}"
    for i in range(512)
}

train_image.rename(
    columns=rename_image_columns,
    inplace=True
)

test_image.rename(
    columns=rename_image_columns,
    inplace=True
)


# --------------------------------------------------
# Prepare text features
# --------------------------------------------------

text_feature_columns = [str(i) for i in range(512)]

text_features.rename(
    columns={
        str(i): f"text_feature_{i}"
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

image_columns = [
    f"image_feature_{i}"
    for i in range(512)
]

text_columns = [
    f"text_feature_{i}"
    for i in range(512)
]

feature_columns = image_columns + text_columns


# --------------------------------------------------
# Prepare X and y
# --------------------------------------------------

X_train = train_data[feature_columns]

y_train = train_df["similarity_score"]

X_test = test_data[feature_columns]

y_test = test_df["similarity_score"]


# --------------------------------------------------
# Train final model
# --------------------------------------------------

print("FINAL MULTIMODAL RIDGE MODEL")
print("=" * 50)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Features: {X_train.shape[1]}")
print("Selected alpha: 100")


model = Ridge(alpha=100)

model.fit(
    X_train,
    y_train
)


# --------------------------------------------------
# Predictions
# --------------------------------------------------

predictions = model.predict(X_test)


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

mae = mean_absolute_error(
    y_test,
    predictions
)

mse = mean_squared_error(
    y_test,
    predictions
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_test,
    predictions
)


# --------------------------------------------------
# Results
# --------------------------------------------------

print("\nFINAL TEST RESULTS")
print("=" * 50)

print(f"MAE:  {mae:.4f}")
print(f"MSE:  {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")

print("\nMultimodal Ridge evaluation completed!")