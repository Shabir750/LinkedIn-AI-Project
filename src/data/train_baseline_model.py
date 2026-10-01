import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
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


def main():

    # Load data
    X_train, y_train = prepare_data(TRAIN_FILE)
    X_validation, y_validation = prepare_data(VALIDATION_FILE)
    X_test, y_test = prepare_data(TEST_FILE)

    print("Training Linear Regression model...")
    
    # Create model
    model = LinearRegression()

    # Train model
    model.fit(X_train, y_train)

    # Predictions
    validation_predictions = model.predict(X_validation)
    test_predictions = model.predict(X_test)

    # Validation metrics
    validation_mae = mean_absolute_error(
        y_validation,
        validation_predictions
    )

    validation_mse = mean_squared_error(
        y_validation,
        validation_predictions
    )

    validation_rmse = validation_mse ** 0.5

    validation_r2 = r2_score(
        y_validation,
        validation_predictions
    )

    # Test metrics
    test_mae = mean_absolute_error(
        y_test,
        test_predictions
    )

    test_mse = mean_squared_error(
        y_test,
        test_predictions
    )

    test_rmse = test_mse ** 0.5

    test_r2 = r2_score(
        y_test,
        test_predictions
    )

    print("\nVALIDATION RESULTS")
    print("=" * 40)
    print(f"MAE:  {validation_mae:.4f}")
    print(f"MSE:  {validation_mse:.4f}")
    print(f"RMSE: {validation_rmse:.4f}")
    print(f"R²:   {validation_r2:.4f}")

    print("\nTEST RESULTS")
    print("=" * 40)
    print(f"MAE:  {test_mae:.4f}")
    print(f"MSE:  {test_mse:.4f}")
    print(f"RMSE: {test_rmse:.4f}")
    print(f"R²:   {test_r2:.4f}")

    print("\nBaseline model training completed!")


if __name__ == "__main__":
    main()