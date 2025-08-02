"""Main training script for NYC taxi ride duration prediction."""

from datetime import datetime
from pathlib import Path

import mlflow
from loguru import logger
from prefect import flow
from sklearn.linear_model import LinearRegression

from e2e_taxi_ride_duration_prediction.ingestion import get_nyc_taxi_data
from e2e_taxi_ride_duration_prediction.mlflow_utils import setup_mlflow
from e2e_taxi_ride_duration_prediction.preprocessing import basic_preprocessing
from e2e_taxi_ride_duration_prediction.training import (
    dict_vectorize_features,
    save_model_and_vectorizer,
    time_series_train_test_split,
    train_model,
    validate_model,
    vectorize_target,
)

logger.add("logs/train_model.log")


@flow
def main(
    start_year: int = 2025,
    start_month: int = 1,
    end_year: int = 2025,
    end_month: int = 3,
    train_end_year: int = 2025,
    train_end_month: int = 2,
    test_start_year: int = 2025,
    test_start_month: int = 2,
    test_end_year: int = 2025,
    test_end_month: int = 3,
):
    """Run the complete ML training pipeline with configurable parameters."""
    ROOT_DIR = Path(__file__).parent.parent
    MODEL_DIR = ROOT_DIR / "models"

    MODEL_DIR.mkdir(exist_ok=True)

    # Setup MLflow tracking URI and experiment
    setup_mlflow()

    with mlflow.start_run():
        # Data ingestion
        logger.info(
            f"Loading NYC taxi data from {start_year}-{start_month:02d} to {end_year}-{end_month:02d}"
        )
        lf = get_nyc_taxi_data(
            root=ROOT_DIR, start=(start_year, start_month), end=(end_year, end_month)
        )

        # Preprocessing
        logger.info("Preprocessing data")
        processed_lf = basic_preprocessing(
            lf,
            start=datetime(start_year, start_month, 1),
            end=datetime(end_year, end_month, 28),
        )

        # Train/test split
        logger.info("Creating train/test split")
        X_train, X_test, y_train, y_test = time_series_train_test_split(
            processed_lf,
            train_start=datetime(start_year, start_month, 1),
            test_start=datetime(test_start_year, test_start_month, 1),
            test_end=datetime(test_end_year, test_end_month, 1),
            train_end=datetime(train_end_year, train_end_month, 1),
        )

        # Vectorization
        logger.info("Vectorizing features")
        X_train_vec, X_test_vec, fitted_dict_vectorizer = dict_vectorize_features(
            X_train, X_test, features=["pickup_dropoff_pair", "trip_distance"]
        )
        y_train_vec, y_test_vec = vectorize_target(y_train, y_test)

        # Training
        logger.info("Training model")
        model = train_model(LinearRegression(), X_train_vec, y_train_vec)

        # Evaluation
        logger.info("Evaluating model")
        results = validate_model(model, X_test_vec, y_test_vec)

        # Save outputs
        model_path = MODEL_DIR / "baseline_taxi_duration_model_and_vectorizer.joblib"
        save_model_and_vectorizer((model, fitted_dict_vectorizer), model_path)

        logger.info(f"Model saved: {model_path}")

        return model, results, fitted_dict_vectorizer


if __name__ == "__main__":
    main()
