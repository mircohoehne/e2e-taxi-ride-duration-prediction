from datetime import datetime

import polars as pl
import pytest
from _pytest.logging import LogCaptureFixture
from loguru import logger
from prefect.testing.utilities import prefect_test_harness


@pytest.fixture()
def string_cache():
    with pl.StringCache():
        yield


@pytest.fixture
def test_data(string_cache) -> pl.LazyFrame:
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


@pytest.fixture(autouse=True, scope="session")
def prefect_test_fixture():
    with prefect_test_harness():
        yield


@pytest.fixture
def test_target_data():
    return pl.LazyFrame({"duration": [15, 30, 45, 75, -10]})


@pytest.fixture
def caplog(caplog: LogCaptureFixture):
    """Make loguru work with pytest. Source: https://loguru.readthedocs.io/en/stable/resources/migration.html#replacing-caplog-fixture-from-pytest-library"""
    handler_id = logger.add(
        caplog.handler,
        format="{message}",
        level=0,
        filter=lambda record: record["level"].no >= caplog.handler.level,
        enqueue=False,  # Set to 'True' if your test is spawning child processes.
    )
    yield caplog
    logger.remove(handler_id)
