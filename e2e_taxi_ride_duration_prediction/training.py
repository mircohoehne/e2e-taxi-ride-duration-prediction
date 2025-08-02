from datetime import datetime
from pathlib import Path
from typing import Any, Union

import joblib
import mlflow
import numpy as np
import numpy.typing as npt
import polars as pl
from loguru import logger
from prefect import task
from scipy.sparse import csr_matrix
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)

from e2e_taxi_ride_duration_prediction.models import SklearnCompatibleRegressor

pl.Config.set_engine_affinity("streaming")


@task
def time_series_train_test_split(
    lf: pl.LazyFrame,
    train_start: datetime,
    test_start: datetime,
    test_end: datetime,
    train_end: datetime | None = None,
    datetime_filter_column: str = "tpep_pickup_datetime",
    target_column: str = "duration",
) -> tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame, pl.LazyFrame]:
    if train_end is None:
        logger.info("train_end not set. Using test_start as train_end")
        train_end = test_start

    X_train = lf.filter(
        (pl.col(datetime_filter_column) >= train_start)
        & (pl.col(datetime_filter_column) < train_end)
    )
    X_test = lf.filter(
        (pl.col(datetime_filter_column) >= test_start)
        & (pl.col(datetime_filter_column) < test_end)
    )

    y_train = X_train.select(target_column)
    y_test = X_test.select(target_column)

    X_train = X_train.select(pl.exclude(target_column))
    X_test = X_test.select(pl.exclude(target_column))

    return X_train, X_test, y_train, y_test


@task
def dict_vectorize_features(
    train_lf: pl.LazyFrame,
    test_lf: pl.LazyFrame,
    features: list[str] | None = None,
) -> tuple[
    Union[csr_matrix, np.ndarray], Union[csr_matrix, np.ndarray], DictVectorizer
]:
    dict_vectorizer = DictVectorizer()
    if features:
        train_dicts = train_lf.select(features).collect().to_dicts()
        test_dicts = test_lf.select(features).collect().to_dicts()
    else:
        train_dicts = train_lf.collect().to_dicts()
        test_dicts = test_lf.collect().to_dicts()

    X_train = dict_vectorizer.fit_transform(train_dicts)
    X_test = dict_vectorizer.transform(test_dicts)

    return X_train, X_test, dict_vectorizer


@task
def vectorize_target(
    train_target_lf: pl.LazyFrame,
    test_target_lf: pl.LazyFrame,
) -> tuple[npt.NDArray, npt.NDArray]:
    y_train = train_target_lf.collect().to_numpy().ravel()
    y_test = test_target_lf.collect().to_numpy().ravel()

    return y_train, y_test


@task
def train_model(
    model: SklearnCompatibleRegressor,
    X_train: Union[csr_matrix, np.ndarray],
    y_train: npt.NDArray,
) -> Any:
    """Train sklearn-compatible model."""
    logger.info(f"Training {type(model).__name__}")
    model.fit(X_train, y_train)
    return model


@task
def validate_model(
    model: SklearnCompatibleRegressor,
    X_test: Union[csr_matrix, np.ndarray],
    y_test: npt.NDArray,
) -> dict[str, float]:
    """Validate sklearn-compatible model."""
    logger.info("Calculating predictions")
    y_pred = model.predict(X_test)
    results = {
        "test_mean_squared_error": mean_squared_error(y_test, y_pred),
        "test_mean_absolute_error": mean_absolute_error(y_test, y_pred),
        "test_r2_score": r2_score(y_test, y_pred),
        "test_root_mean_squared_error": root_mean_squared_error(y_test, y_pred),
    }
    logger.info(f"Results: {results}")
    try:
        active_run = mlflow.active_run()
        if active_run:
            logger.info(f"Logging metrics to run: {active_run.info.run_id}")
            mlflow.log_metrics(results)
            logger.info("Successfully logged metrics to MLflow")
        else:
            logger.warning("No active MLflow run found")
    except Exception as e:
        logger.warning(f"Failed to log metrics to MLflow: {e}")

    return results


@task
def save_model_and_vectorizer(
    model: tuple[SklearnCompatibleRegressor, DictVectorizer],
    save_path: str | Path | None,
):
    """Save any model and vectorizer pair."""
    joblib.dump(model, save_path)
    logger.info(f"saved model and vectorizer to {save_path}.")
