import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_DIR = Path("data/linkedin_ai_dataset")

TRAIN_FILE = DATA_DIR / "train.csv"
VALIDATION_FILE = DATA_DIR / "validation.csv"
TEST_FILE = DATA_DIR / "test.csv"
TEXT_FEATURE_FILE = DATA_DIR / "text_features.csv"


def prepare_data(split_file):

    split_df = pd.read_csv(split_file)
    text_df = pd.read_csv(TEXT_FEATURE_FILE)

    # Rename image features
    split_df = split_df.rename(
        columns={str(i): f"image_feature_{i}" for i in range(512)}
    )

    # Rename text features
    text_df = text_df.rename(
        columns={str(i): f"text_feature_{i}" for i in range(512)}
    )

    image_features = [
        f"image_feature_{i}" for i in range(512)
    ]

    text_features = [
        f"text_feature_{i}" for i in range(512)
    ]

    image_df = split_df[
        ["image_id"] + image_features + ["similarity_score"]
    ]

    text_df = text_df[
        ["image_id"] + text_features
    ]

    merged_df = image_df.merge(
        text_df,
        on="image_id",
        how="inner"
    )

    X = merged_df[image_features + text_features]
    y = merged_df["similarity_score"]

    return X, y


def evaluate(model, X, y):

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

    print("Training multimodal Ridge model...")
    print("Features: 1024")
    print("Selected alpha: 10")

    model = Ridge(alpha=10)

    model.fit(X_train, y_train)

    val_mae, val_mse, val_rmse, val_r2 = evaluate(
        model,
        X_validation,
        y_validation
    )

    test_mae, test_mse, test_rmse, test_r2 = evaluate(
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

    print("\nMultimodal Ridge training completed!")


if __name__ == "__main__":
    main()