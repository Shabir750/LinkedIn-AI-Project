import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------------------------
# Load datasets
# --------------------------------------------------

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

image_columns = [str(i) for i in range(512)]

train_image = train_df[
    ["image_id"] + image_columns
].copy()

test_image = test_df[
    ["image_id"] + image_columns
].copy()


rename_image = {
    str(i): f"image_feature_{i}"
    for i in range(512)
}

train_image.rename(
    columns=rename_image,
    inplace=True
)

test_image.rename(
    columns=rename_image,
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

image_feature_columns = [
    f"image_feature_{i}"
    for i in range(512)
]

text_feature_columns = [
    f"text_feature_{i}"
    for i in range(512)
]

feature_columns = (
    image_feature_columns +
    text_feature_columns
)


# --------------------------------------------------
# X and y
# --------------------------------------------------

X_train = train_data[feature_columns]

y_train = train_df["similarity_score"]

X_test = test_data[feature_columns]

y_test = test_df["similarity_score"]


print("FINAL PCA + RIDGE MODEL")
print("=" * 50)

print(f"Original features: {X_train.shape[1]}")
print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")

print("PCA components: 60")
print("Ridge alpha: 100")


# --------------------------------------------------
# Standardization
# --------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


# --------------------------------------------------
# PCA
# --------------------------------------------------

pca = PCA(
    n_components=60
)

X_train_pca = pca.fit_transform(
    X_train_scaled
)

X_test_pca = pca.transform(
    X_test_scaled
)


print(
    f"Explained variance: "
    f"{pca.explained_variance_ratio_.sum():.4f}"
)


# --------------------------------------------------
# Ridge
# --------------------------------------------------

model = Ridge(
    alpha=100
)

model.fit(
    X_train_pca,
    y_train
)


# --------------------------------------------------
# Test prediction
# --------------------------------------------------

predictions = model.predict(
    X_test_pca
)


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

print("\nPCA + Ridge evaluation completed!")