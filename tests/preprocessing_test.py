from datetime import datetime

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from e2e_taxi_ride_duration_prediction.preprocessing import (
    basic_preprocessing,
    calculate_duration,
    cast_categorical_columns,
    create_pickup_dropoff_pairs,
    filter_by_date_range,
    filter_valid_durations,
)


@pytest.fixture
def test_data():
    return pl.LazyFrame(
        {
            "VendorID": [1, 2, 1, 2, 1],
            "tpep_pickup_datetime": [
                datetime(2025, 1, 1, 10, 0, 0),
                datetime(2025, 1, 1, 11, 30, 0),
                datetime(2024, 12, 31, 23, 45, 0),  # below date range
                datetime(2026, 1, 1, 0, 0, 0),  # above date range
                datetime(2025, 1, 2, 14, 15, 0),
            ],
            "tpep_dropoff_datetime": [
                datetime(2025, 1, 1, 10, 15, 0),  # 15 min duration
                datetime(2025, 1, 1, 12, 0, 0),  # 30 min duration
                datetime(2025, 1, 1, 0, 30, 0),  # 45 min duration
                datetime(2026, 1, 1, 1, 15, 0),  # 75 min duration (invalid)
                datetime(2025, 1, 2, 14, 5, 0),  # -10 min duration (invalid)
            ],
            "RatecodeID": [1, 1, 2, 1, 3],
            "store_and_fwd_flag": ["N", "N", "Y", "N", "Y"],
            "PULocationID": [100, 200, 150, 300, 100],
            "DOLocationID": [110, 250, 160, 310, 120],
            "payment_type": [1, 2, 1, 1, 2],
            "trip_distance": [2.5, 5.1, 3.2, 8.7, 1.8],
            "fare_amount": [12.5, 18.0, 15.5, 25.0, 8.5],
        }
    )


@pytest.fixture
def test_duration_data():
    return pl.LazyFrame({"duration": [15, 30, 45, 75, -10]})


@pytest.fixture
def test_date_range():
    return pl.datetime(2025, 1, 1), pl.datetime(2025, 2, 1)


def test_calculate_duration(test_data):
    expected = pl.LazyFrame({"duration": [15.0, 30.0, 45.0, 75.0, -10.0]})
    result = calculate_duration(test_data).select("duration")
    assert_frame_equal(result, expected)


def test_filter_by_date_range(test_data, test_date_range):
    start, end = test_date_range
    expected = pl.LazyFrame(
        {
            "tpep_pickup_datetime": [
                datetime(2025, 1, 1, 10, 0, 0),
                datetime(2025, 1, 1, 11, 30, 0),
                datetime(2025, 1, 2, 14, 15, 0),
            ]
        }
    )
    result = filter_by_date_range(test_data, start, end).select("tpep_pickup_datetime")
    assert_frame_equal(result, expected)


def test_filter_valid_durations(test_duration_data):
    expected = pl.LazyFrame({"duration": [15, 30, 45]})
    result = filter_valid_durations(test_duration_data)
    assert_frame_equal(result, expected)


def test_cast_categorical_columns(test_data):
    expected = pl.LazyFrame(
        {
            "VendorID": ["1", "2", "1", "2", "1"],
            "RatecodeID": ["1", "1", "2", "1", "3"],
            "store_and_fwd_flag": ["N", "N", "Y", "N", "Y"],
            "PULocationID": ["100", "200", "150", "300", "100"],
            "DOLocationID": ["110", "250", "160", "310", "120"],
            "payment_type": ["1", "2", "1", "1", "2"],
        },
        schema={
            "VendorID": pl.Categorical,
            "RatecodeID": pl.Categorical,
            "store_and_fwd_flag": pl.Categorical,
            "PULocationID": pl.Categorical,
            "DOLocationID": pl.Categorical,
            "payment_type": pl.Categorical,
        },
    )
    result = cast_categorical_columns(test_data).select(
        [
            "VendorID",
            "RatecodeID",
            "store_and_fwd_flag",
            "PULocationID",
            "DOLocationID",
            "payment_type",
        ]
    )
    assert_frame_equal(result, expected)


def test_create_pickup_dropoff_pairs(test_data):
    expected = pl.LazyFrame(
        {
            "pickup_dropoff_pair": [
                "100_110",
                "200_250",
                "150_160",
                "300_310",
                "100_120",
            ]
        },
        schema={"pickup_dropoff_pair": pl.Categorical},
    )
    result = create_pickup_dropoff_pairs(test_data).select("pickup_dropoff_pair")
    assert_frame_equal(result, expected)


def test_basic_preprocessing(test_data, test_date_range):
    start, end = test_date_range
    expected = pl.LazyFrame(
        {
            "VendorID": ["1", "2"],
            "tpep_pickup_datetime": [
                datetime(2025, 1, 1, 10, 0, 0),
                datetime(2025, 1, 1, 11, 30, 0),
            ],
            "tpep_dropoff_datetime": [
                datetime(2025, 1, 1, 10, 15, 0),
                datetime(2025, 1, 1, 12, 0, 0),
            ],
            "RatecodeID": ["1", "1"],
            "store_and_fwd_flag": ["N", "N"],
            "PULocationID": ["100", "200"],
            "DOLocationID": ["110", "250"],
            "payment_type": ["1", "2"],
            "trip_distance": [2.5, 5.1],
            "fare_amount": [12.5, 18.0],
            "duration": [15.0, 30.0],
            "pickup_dropoff_pair": ["100_110", "200_250"],
        },
        schema_overrides={
            "VendorID": pl.Categorical,
            "RatecodeID": pl.Categorical,
            "store_and_fwd_flag": pl.Categorical,
            "PULocationID": pl.Categorical,
            "DOLocationID": pl.Categorical,
            "payment_type": pl.Categorical,
            "pickup_dropoff_pair": pl.Categorical,
        },
    )
    result = basic_preprocessing(test_data, start, end)
    assert_frame_equal(result, expected)
