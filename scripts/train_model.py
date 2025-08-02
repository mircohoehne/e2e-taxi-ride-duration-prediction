"""Main training script for NYC taxi ride duration prediction."""

import json
from datetime import datetime
from pathlib import Path

import mlflow
from loguru import logger
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


def main():
    """Run the complete ML training pipeline."""
    ROOT_DIR = Path(__file__).parent.parent
    MODEL_DIR = ROOT_DIR / "models"
    RESULTS_DIR = ROOT_DIR / "results"

    MODEL_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    # Setup MLflow tracking URI and experiment
    setup_mlflow()

    with mlflow.start_run():
        # Data ingestion
        logger.info("Loading NYC taxi data")
        lf = get_nyc_taxi_data(root=ROOT_DIR, start=(2025, 1), end=(2025, 3))

        # Preprocessing
        logger.info("Preprocessing data")
        processed_lf = basic_preprocessing(
            lf, start=datetime(2025, 1, 1), end=datetime(2025, 3, 31)
        )

        # Train/test split
        logger.info("Creating train/test split")
        X_train, X_test, y_train, y_test = time_series_train_test_split(
            processed_lf,
            train_start=datetime(2025, 1, 1),
            test_start=datetime(2025, 2, 1),
            test_end=datetime(2025, 3, 1),
            train_end=datetime(2025, 2, 1),
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

        results_path = RESULTS_DIR / "baseline_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Model saved: {model_path}")
        logger.info(f"Results saved: {results_path}")

        return model, results, fitted_dict_vectorizer


if __name__ == "__main__":
    model, results, fitted_dict_vectorizer = main()
