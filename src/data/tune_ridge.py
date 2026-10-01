import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_DIR = Path("data/linkedin_ai_dataset")

TRAIN_FILE = DATA_DIR / "train.csv"
VALIDATION_FILE = DATA_DIR / "validation.csv"


def prepare_data(file_path):
    df = pd.read_csv(file_path)

    # 512 CLIP image features
    image_feature_columns = [str(i) for i in range(512)]

    X = df[image_feature_columns]
    y = df["similarity_score"]

    return X, y


def main():

    X_train, y_train = prepare_data(TRAIN_FILE)
    X_validation, y_validation = prepare_data(VALIDATION_FILE)

    alpha_values = [0.01, 0.1, 1, 10, 100]

    results = []

    print("RIDGE HYPERPARAMETER TUNING")
    print("=" * 60)

    for alpha in alpha_values:

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

        rmse = mse ** 0.5

        r2 = r2_score(
            y_validation,
            predictions
        )

        results.append({
            "alpha": alpha,
            "mae": mae,
            "rmse": rmse,
            "r2": r2
        })

        print(
            f"alpha={alpha:<6} "
            f"MAE={mae:.4f}  "
            f"RMSE={rmse:.4f}  "
            f"R²={r2:.4f}"
        )

    # Find best alpha based on validation MAE
    best_result = min(
        results,
        key=lambda result: result["mae"]
    )

    print("\nBEST ALPHA")
    print("=" * 60)

    print(f"Alpha: {best_result['alpha']}")
    print(f"Validation MAE: {best_result['mae']:.4f}")
    print(f"Validation RMSE: {best_result['rmse']:.4f}")
    print(f"Validation R²: {best_result['r2']:.4f}")

    print("\nRidge hyperparameter tuning completed!")


if __name__ == "__main__":
    main()