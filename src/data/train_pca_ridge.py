import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load datasets
train_df = pd.read_csv(
    "data/linkedin_ai_dataset/train.csv"
)

validation_df = pd.read_csv(
    "data/linkedin_ai_dataset/validation.csv"
)

text_features = pd.read_csv(
    "data/linkedin_ai_dataset/text_features.csv"
)


# -------------------------------
# Prepare image features
# -------------------------------

image_columns = [str(i) for i in range(512)]

train_image = train_df[
    ["image_id"] + image_columns
].copy()

validation_image = validation_df[
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

validation_image.rename(
    columns=rename_image,
    inplace=True
)


# -------------------------------
# Prepare text features
# -------------------------------

text_features.rename(
    columns={
        str(i): f"text_feature_{i}"
        for i in range(512)
    },
    inplace=True
)


# -------------------------------
# Merge image + text
# -------------------------------

train_data = train_image.merge(
    text_features,
    on="image_id",
    how="inner"
)

validation_data = validation_image.merge(
    text_features,
    on="image_id",
    how="inner"
)


# -------------------------------
# Feature columns
# -------------------------------

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


# -------------------------------
# X and y
# -------------------------------

X_train = train_data[feature_columns]

y_train = train_df["similarity_score"]

X_validation = validation_data[feature_columns]

y_validation = validation_df["similarity_score"]


print("PCA + RIDGE MODEL")
print("=" * 50)

print(f"Original features: {X_train.shape[1]}")
print(f"Training samples: {X_train.shape[0]}")
print(f"Validation samples: {X_validation.shape[0]}")


# -------------------------------
# Standardization
# -------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_validation_scaled = scaler.transform(
    X_validation
)


# -------------------------------
# PCA
# -------------------------------

pca = PCA(
    n_components=50
)

X_train_pca = pca.fit_transform(
    X_train_scaled
)

X_validation_pca = pca.transform(
    X_validation_scaled
)


print(f"PCA components: {X_train_pca.shape[1]}")
print(
    f"Explained variance: "
    f"{pca.explained_variance_ratio_.sum():.4f}"
)


# -------------------------------
# Ridge
# -------------------------------

model = Ridge(alpha=100)

model.fit(
    X_train_pca,
    y_train
)


# -------------------------------
# Validation prediction
# -------------------------------

predictions = model.predict(
    X_validation_pca
)


# -------------------------------
# Evaluation
# -------------------------------

mae = mean_absolute_error(
    y_validation,
    predictions
)

mse = mean_squared_error(
    y_validation,
    predictions
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_validation,
    predictions
)


print("\nVALIDATION RESULTS")
print("=" * 50)

print(f"MAE:  {mae:.4f}")
print(f"MSE:  {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")

print("\nPCA + Ridge training completed!")