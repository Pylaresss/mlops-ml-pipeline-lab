"""
train_model.py

A clean, modular training script designed for reproducible MLOps pipelines.

This script:
- Loads pre-split training and test datasets
- Separates features and target
- Trains a Linear Regression model
- Evaluates performance (MSE on train & test)
- Saves the trained model to a standardized modelstore

Intended usage inside a CI/CD or DVC/MLflow pipeline.
"""

from pathlib import Path
from typing import Union
import pandas as pd
import joblib
import sys
import argparse
import logging
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# -------------------------------------------------------------------
#  Global Configuration
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATASTORE_DIR = PROJECT_ROOT / "datastores"

MODELSTORE_DIR = PROJECT_ROOT / "modelstores"
MODELSTORE_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = DATASTORE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------
#  Logging Configuration
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "training.log", mode="a"),
    ],
)

logger = logging.getLogger("train_model")
logger.info(f"PROJECT_ROOT detected at: {PROJECT_ROOT}")


# -------------------------------------------------------------------
#  Functions
# -------------------------------------------------------------------
def load_data(input_data_path: Union[str, Path]) -> pd.DataFrame:
    """
    Loads a CSV file into a Pandas DataFrame.

    Args:
        input_data_path (Union[str, Path]): Path to CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    try:
        input_path = Path(input_data_path)
        logger.info(f"Loading dataset from: {input_path}")

        df = pd.read_csv(input_path)
        logger.info(f"Dataset loaded successfully shape={df.shape}")

        return df

    except FileNotFoundError:
        logger.error(f"ERROR: File not found {input_path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"ERROR loading {input_path}: {e}")
        sys.exit(1)


def train_model(
    input_train_data: str, input_test_data: str, model_filename: str
) -> None:
    """
    Loads train/test datasets, trains a regression model,
    evaluates its performance, and saves it to modelstore.

    Args:
        input_train_data (str): Path to training CSV.
        input_test_data  (str): Path to testing CSV.
        model_filename   (str): Filename to save the trained model with extension .joblib
    """
    logger.info("Loading train and test datasets...")
    df_train = load_data(input_train_data)
    df_test = load_data(input_test_data)

    logger.info("Splitting features and target column")
    target_col = "MEDV"
    if target_col not in df_train.columns or target_col not in df_test.columns:
        raise ValueError(
            f"Target column '{target_col}' must exist in both train and test datasets"
        )

    X_train = df_train.drop(columns=[target_col])
    y_train = df_train[target_col]

    X_test = df_test.drop(columns=[target_col])
    y_test = df_test[target_col]

    logger.info("Training Linear Regression model...")
    # 1. Define the model (e.g., LinearRegression)
    model = LinearRegression()

    # 2. Train the model using fit method
    model.fit(X_train, y_train)

    logger.info("Evaluating model on TRAIN set...")
    train_pred = model.predict(X_train)
    train_mse = mean_squared_error(y_train, train_pred)
    logger.info(f"Train MSE: {train_mse:.4f}")

    logger.info("Evaluating model on TEST set...")
    test_pred = model.predict(X_test)
    test_mse  = mean_squared_error(y_test, test_pred)
    logger.info(f"Test MSE: {test_mse:.4f}")

    # Save model
    save_path = MODELSTORE_DIR / model_filename
    joblib.dump(model, save_path)
    logger.info(f"Model saved to: {save_path}")


# -------------------------------------------------------------------
#  CLI Interface
# -------------------------------------------------------------------
def parse_arguments() -> argparse.Namespace:
    """
    Parses CLI arguments for the training pipeline.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train a Linear Regression model for MLOps pipeline"
    )

    parser.add_argument(
        "--input_train_data",
        type=str,
        required=True,
        help="Path to the training CSV file.",
    )

    parser.add_argument(
        "--input_test_data",
        type=str,
        required=True,
        help="Path to the testing CSV file.",
    )

    parser.add_argument(
        "--model_filename",
        type=str,
        required=True,
        default='LinearRegression.joblib',
        help="Output model filename with extension .joblib (stored in modelstores/).",
    )

    return parser.parse_args()


# -------------------------------------------------------------------
#  Main Entry Point
# -------------------------------------------------------------------
def main() -> None:
    """
    Main function for CLI execution.
    Loads datasets, trains a model, and saves it as an artifact.
    """
    args = parse_arguments()

    train_model(
        input_train_data=args.input_train_data,
        input_test_data=args.input_test_data,
        model_filename=args.model_filename,
    )


    


if __name__ == "__main__":
    main()
