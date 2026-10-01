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

validation_df = pd.read_csv(
    "data/linkedin_ai_dataset/validation.csv"
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

validation_data = validation_image.merge(
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

X_validation = validation_data[feature_columns]

y_validation = validation_df["similarity_score"]


print("PCA + RIDGE HYPERPARAMETER TUNING")
print("=" * 60)

print(f"Original features: {X_train.shape[1]}")
print(f"Training samples: {X_train.shape[0]}")
print(f"Validation samples: {X_validation.shape[0]}")


# --------------------------------------------------
# Standardization
# --------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_validation_scaled = scaler.transform(
    X_validation
)


# --------------------------------------------------
# PCA component values
# --------------------------------------------------

components = [10, 25, 50, 60, 70]

results = []


# --------------------------------------------------
# Try different PCA sizes
# --------------------------------------------------

for n_components in components:

    pca = PCA(
        n_components=n_components
    )

    X_train_pca = pca.fit_transform(
        X_train_scaled
    )

    X_validation_pca = pca.transform(
        X_validation_scaled
    )


    # Ridge
    model = Ridge(alpha=100)

    model.fit(
        X_train_pca,
        y_train
    )


    # Validation prediction
    predictions = model.predict(
        X_validation_pca
    )


    # Metrics
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

    explained_variance = (
        pca.explained_variance_ratio_.sum()
    )


    results.append(
        {
            "components": n_components,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2,
            "variance": explained_variance
        }
    )


    print(
        f"Components={n_components:<3} "
        f"Variance={explained_variance:.4f} "
        f"MAE={mae:.4f} "
        f"RMSE={rmse:.4f} "
        f"R²={r2:.4f}"
    )


# --------------------------------------------------
# Select best configuration
# --------------------------------------------------

best_result = min(
    results,
    key=lambda x: x["MAE"]
)


print("\nBEST PCA CONFIGURATION")
print("=" * 60)

print(
    f"Components: "
    f"{best_result['components']}"
)

print(
    f"Explained variance: "
    f"{best_result['variance']:.4f}"
)

print(
    f"Validation MAE: "
    f"{best_result['MAE']:.4f}"
)

print(
    f"Validation RMSE: "
    f"{best_result['RMSE']:.4f}"
)

print(
    f"Validation R²: "
    f"{best_result['R2']:.4f}"
)

print("\nPCA tuning completed!")