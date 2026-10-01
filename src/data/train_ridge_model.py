import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_DIR = Path("data/linkedin_ai_dataset")

TRAIN_FILE = DATA_DIR / "train.csv"
VALIDATION_FILE = DATA_DIR / "validation.csv"
TEST_FILE = DATA_DIR / "test.csv"


def prepare_data(file_path):
    df = pd.read_csv(file_path)

    # 512 CLIP image features
    image_feature_columns = [str(i) for i in range(512)]

    X = df[image_feature_columns]
    y = df["similarity_score"]

    return X, y


def evaluate_model(model, X, y):
    predictions = model.predict(X)

    mae = mean_absolute_error(y, predictions)
    mse = mean_squared_error(y, predictions)
    rmse = mse ** 0.5
    r2 = r2_score(y, predictions)

    return mae, mse, rmse, r2


def main():

    X_train, y_train = prepare_data(TRAIN_FILE)
    X_validation, y_validation = prepare_data(VALIDATION_FILE)
    X_test, y_test = prepare_data(TEST_FILE)

    print("Training Ridge Regression model...")

    # Ridge regularization
    model = Ridge(alpha=1.0)

    # Train
    model.fit(X_train, y_train)

    # Validation
    val_mae, val_mse, val_rmse, val_r2 = evaluate_model(
        model,
        X_validation,
        y_validation
    )

    # Test
    test_mae, test_mse, test_rmse, test_r2 = evaluate_model(
        model,
        X_test,
        y_test
    )

    print("\nVALIDATION RESULTS")
    print("=" * 40)
    print(f"MAE:  {val_mae:.4f}")
    print(f"MSE:  {val_mse:.4f}")
    print(f"RMSE: {val_rmse:.4f}")
    print(f"R²:   {val_r2:.4f}")

    print("\nTEST RESULTS")
    print("=" * 40)
    print(f"MAE:  {test_mae:.4f}")
    print(f"MSE:  {test_mse:.4f}")
    print(f"RMSE: {test_rmse:.4f}")
    print(f"R²:   {test_r2:.4f}")

    print("\nRidge model training completed!")


if __name__ == "__main__":
    main()