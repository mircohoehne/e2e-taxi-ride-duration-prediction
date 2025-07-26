from datetime import datetime
from pathlib import Path

import joblib
import numpy.typing as npt
import polars as pl
from loguru import logger
from scipy.sparse import csr_matrix
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

pl.Config.set_engine_affinity("streaming")


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


def dict_vectorize_features(
    train_lf: pl.LazyFrame,
    test_lf: pl.LazyFrame,
    features: list[str] | None = None,
) -> tuple[csr_matrix, csr_matrix]:
    dict_vectorizer = DictVectorizer()
    if features:
        train_dicts = train_lf.select(features).collect().to_dicts()
        test_dicts = test_lf.select(features).collect().to_dicts()
    else:
        train_dicts = train_lf.collect().to_dicts()
        test_dicts = test_lf.collect().to_dicts()

    X_train = dict_vectorizer.fit_transform(train_dicts)
    X_test = dict_vectorizer.transform(test_dicts)

    return X_train, X_test


def vectorize_target(
    train_target_lf: pl.LazyFrame,
    test_target_lf: pl.LazyFrame,
) -> tuple[npt.NDArray, npt.NDArray]:
    y_train = train_target_lf.collect().to_numpy().ravel()
    y_test = test_target_lf.collect().to_numpy().ravel()

    return y_train, y_test


def train_linear_regression(
    X_train: csr_matrix,
    y_train: npt.NDArray,
    parameters: dict | None = None,
) -> LinearRegression:
    # TODO: Add parameter validation for LinearRegression kwargs
    if parameters:
        linear_regression_model = LinearRegression(**parameters)
    else:
        linear_regression_model = LinearRegression()
    linear_regression_model.fit(X_train, y_train)

    return linear_regression_model


def validate_linear_regression(
    X_test: csr_matrix,
    y_test: npt.NDArray,
    model: LinearRegression,
) -> dict[str, float]:
    logger.info("Calculating predictions")
    y_pred = model.predict(X_test)
    results = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": root_mean_squared_error(y_test, y_pred),
        "r2_score": r2_score(y_test, y_pred),
    }
    logger.info(f"Results: {results}")
    return results


def save_model(model: LinearRegression, save_path: str | Path | None):
    joblib.dump(model, save_path)
    logger.info(f"saved model to {save_path}.")
