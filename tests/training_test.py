import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock

import joblib
import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal
from scipy.sparse import csr_matrix
from sklearn.linear_model import LinearRegression

from e2e_taxi_ride_duration_prediction.training import (
    dict_vectorize_features,
    save_model_and_vectorizer,
    time_series_train_test_split,
    train_model,
    validate_model,
    vectorize_target,
)


@pytest.fixture
def test_data_with_target(
    test_data: pl.LazyFrame, test_target_data: pl.LazyFrame
) -> pl.LazyFrame:
    return pl.concat([test_data, test_target_data], how="horizontal").lazy()


def test_time_series_train_test_split(test_data_with_target):
    X_train_result, X_test_result, y_train_result, y_test_result = (
        time_series_train_test_split(
            lf=test_data_with_target,
            train_start=datetime(2024, 12, 31),
            train_end=datetime(2025, 1, 2),
            test_start=datetime(2026, 1, 1),
            test_end=datetime(2026, 1, 3),
        )
    )
    X_train_expected = test_data_with_target.slice(0, 3).select(pl.exclude("duration"))
    X_test_expected = test_data_with_target.slice(3, 1).select(pl.exclude("duration"))
    y_train_expected = test_data_with_target.slice(0, 3).select("duration")
    y_test_expected = test_data_with_target.slice(3, 1).select("duration")

    assert_frame_equal(X_train_result, X_train_expected)
    assert_frame_equal(X_test_result, X_test_expected)
    assert_frame_equal(y_train_result, y_train_expected)
    assert_frame_equal(y_test_result, y_test_expected)


def test_time_series_train_test_split_no_train_end(test_data_with_target):
    X_train_expected = test_data_with_target.slice(0, 3).select(pl.exclude("duration"))
    X_test_expected = test_data_with_target.slice(3, 2).select(pl.exclude("duration"))
    y_train_expected = test_data_with_target.slice(0, 3).select("duration")
    y_test_expected = test_data_with_target.slice(3, 2).select("duration")

    X_train_result, X_test_result, y_train_result, y_test_result = (
        time_series_train_test_split(
            lf=test_data_with_target,
            train_start=datetime(2024, 12, 31),
            test_start=datetime(2025, 1, 2),
            test_end=datetime(2026, 1, 3),
        )
    )
    assert_frame_equal(X_train_result, X_train_expected)
    assert_frame_equal(X_test_result, X_test_expected)
    assert_frame_equal(y_train_result, y_train_expected)
    assert_frame_equal(y_test_result, y_test_expected)


def test_dict_vectorize_features(test_data):
    train_lf = test_data.slice(0, 3)
    test_lf = test_data.slice(3, 2)
    features = ["VendorID", "RatecodeID"]

    X_train, X_test, vectorizer = dict_vectorize_features(train_lf, test_lf, features)

    assert isinstance(X_train, csr_matrix)
    assert isinstance(X_test, csr_matrix)
    assert X_train.shape[0] == 3
    assert X_test.shape[0] == 2
    assert X_train.shape[1] == X_test.shape[1]
    assert len(vectorizer.get_feature_names_out()) > 0


def test_dict_vectorize_features_all_columns(test_data):
    train_lf = test_data.slice(0, 3).select(["VendorID", "RatecodeID", "trip_distance"])
    test_lf = test_data.slice(3, 2).select(["VendorID", "RatecodeID", "trip_distance"])

    X_train, X_test, vectorizer = dict_vectorize_features(train_lf, test_lf)

    assert isinstance(X_train, csr_matrix)
    assert isinstance(X_test, csr_matrix)
    assert X_train.shape[0] == 3
    assert X_test.shape[0] == 2
    assert X_train.shape[1] == X_test.shape[1]
    assert len(vectorizer.get_feature_names_out()) > 0


def test_vectorize_target(test_target_data):
    train_target_lf = test_target_data.slice(0, 3)
    test_target_lf = test_target_data.slice(3, 2)

    y_train, y_test = vectorize_target(train_target_lf, test_target_lf)

    assert isinstance(y_train, np.ndarray)
    assert isinstance(y_test, np.ndarray)
    assert y_train.shape == (3,)
    assert y_test.shape == (2,)
    np.testing.assert_array_equal(y_train, [15, 30, 45])
    np.testing.assert_array_equal(y_test, [75, -10])


def test_train_model():
    X_train = csr_matrix([[1, 0], [0, 1], [1, 1]])
    y_train = np.array([1.0, 2.0, 3.0])
    model = LinearRegression()

    trained_model = train_model(model, X_train, y_train)

    assert trained_model is model
    assert hasattr(trained_model, "coef_")
    assert hasattr(trained_model, "intercept_")


def test_validate_model():
    X_test = csr_matrix([[1, 0], [0, 1]])
    y_test = np.array([1.0, 2.0])

    mock_model = Mock()
    mock_model.predict.return_value = np.array([1.1, 1.9])

    results = validate_model(mock_model, X_test, y_test)

    assert isinstance(results, dict)

    assert "test_mean_squared_error" in results
    assert "test_mean_absolute_error" in results
    assert "test_r2_score" in results
    assert "test_root_mean_squared_error" in results

    assert all(isinstance(v, float) for v in results.values())
    mock_model.predict.assert_called_once_with(X_test)


def test_save_model_and_vectorizer():
    from sklearn.feature_extraction import DictVectorizer

    model = LinearRegression()
    vectorizer = DictVectorizer()
    model_tuple = (model, vectorizer)

    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp_file:
        save_path = Path(tmp_file.name)

    try:
        save_model_and_vectorizer(model_tuple, save_path)
        assert save_path.exists()

        loaded_model_tuple = joblib.load(save_path)
        assert len(loaded_model_tuple) == 2
        assert isinstance(loaded_model_tuple[0], LinearRegression)
        assert isinstance(loaded_model_tuple[1], DictVectorizer)
    finally:
        if save_path.exists():
            save_path.unlink()
