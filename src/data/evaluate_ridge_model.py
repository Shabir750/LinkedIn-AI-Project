import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_DIR = Path("data/linkedin_ai_dataset")

TRAIN_FILE = DATA_DIR / "train.csv"
TEST_FILE = DATA_DIR / "test.csv"


def prepare_data(file_path):
    df = pd.read_csv(file_path)

    # 512 CLIP image features
    image_feature_columns = [str(i) for i in range(512)]

    X = df[image_feature_columns]
    y = df["similarity_score"]

    return X, y


def main():

    # Load training and test data
    X_train, y_train = prepare_data(TRAIN_FILE)
    X_test, y_test = prepare_data(TEST_FILE)

    print("Training final Ridge model...")
    print("Selected alpha: 10")

    # Use the alpha selected using validation data
    model = Ridge(alpha=10)

    model.fit(X_train, y_train)

    # Test predictions
    predictions = model.predict(X_test)

    # Calculate metrics
    mae = mean_absolute_error(y_test, predictions)

    mse = mean_squared_error(y_test, predictions)

    rmse = mse ** 0.5

    r2 = r2_score(y_test, predictions)

    print("\nFINAL TEST RESULTS")
    print("=" * 40)

    print(f"MAE:  {mae:.4f}")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")

    print("\nFinal Ridge evaluation completed!")


if __name__ == "__main__":
    main()