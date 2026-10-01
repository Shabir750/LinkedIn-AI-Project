import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load datasets
train_df = pd.read_csv("data/linkedin_ai_dataset/train.csv")
validation_df = pd.read_csv("data/linkedin_ai_dataset/validation.csv")

text_features = pd.read_csv(
    "data/linkedin_ai_dataset/text_features.csv"
)


# Rename image features
image_feature_columns = [str(i) for i in range(512)]

train_image = train_df[["image_id"] + image_feature_columns].copy()
validation_image = validation_df[
    ["image_id"] + image_feature_columns
].copy()

train_image.rename(
    columns={
        str(i): f"image_feature_{i}"
        for i in range(512)
    },
    inplace=True
)

validation_image.rename(
    columns={
        str(i): f"image_feature_{i}"
        for i in range(512)
    },
    inplace=True
)


# Rename text features
text_feature_columns = [str(i) for i in range(512)]

text_features.rename(
    columns={
        str(i): f"text_feature_{i}"
        for i in range(512)
    },
    inplace=True
)


# Merge image + text features
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


# Feature columns
image_columns = [
    f"image_feature_{i}"
    for i in range(512)
]

text_columns = [
    f"text_feature_{i}"
    for i in range(512)
]

feature_columns = image_columns + text_columns


# Prepare X and y
X_train = train_data[feature_columns]
y_train = train_df["similarity_score"]

X_validation = validation_data[feature_columns]
y_validation = validation_df["similarity_score"]


print("MULTIMODAL RIDGE HYPERPARAMETER TUNING")
print("=" * 50)

print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_validation)}")
print(f"Features: {X_train.shape[1]}")


# Alpha values
alphas = [0.01, 0.1, 1, 10, 100]


results = []


for alpha in alphas:

    model = Ridge(alpha=alpha)

    model.fit(X_train, y_train)

    predictions = model.predict(X_validation)

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

    results.append(
        {
            "alpha": alpha,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2
        }
    )

    print(
        f"alpha={alpha:<6} "
        f"MAE={mae:.4f} "
        f"RMSE={rmse:.4f} "
        f"R²={r2:.4f}"
    )


# Find best alpha using validation MAE
best_result = min(
    results,
    key=lambda x: x["MAE"]
)


print("\nBEST ALPHA")
print("=" * 50)

print(f"Alpha: {best_result['alpha']}")
print(f"Validation MAE: {best_result['MAE']:.4f}")
print(f"Validation RMSE: {best_result['RMSE']:.4f}")
print(f"Validation R²: {best_result['R2']:.4f}")