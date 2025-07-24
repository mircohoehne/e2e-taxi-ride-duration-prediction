import logging
from pathlib import Path
from unittest.mock import Mock, patch

import polars as pl
import pytest
import requests

from e2e_taxi_ride_duration_prediction.ingestion import (
    concatenate_parquet_files,
    download_parquet_file,
    generate_year_month_tuples,
    get_data_path,
    get_nyc_taxi_data,
)


def test_generate_year_month_tuples_across_years():
    result = generate_year_month_tuples(start=(2019, 11), end=(2020, 3))
    expected = [
        (2019, 11),
        (2019, 12),
        (2020, 1),
        (2020, 2),
        (2020, 3),
    ]
    assert result == expected


def test_generate_year_month_tuples_single_month():
    expected = [(2023, 5)]
    result = generate_year_month_tuples(start=(2023, 5), end=(2023, 5))
    assert result == expected


def test_generate_year_month_tuples_single_year():
    result = generate_year_month_tuples(start=(2023, 1), end=(2023, 12))
    expected = [(2023, month) for month in range(1, 13)]
    assert result == expected


def test_get_data_path():
    root = Path("/test/path")
    start = (2023, 1)
    end = (2023, 3)

    expected = Path("/test/path/data/raw/yellow_tripdata_2023-01_2023-03.parquet")
    result = get_data_path(root, start, end)
    assert result == expected


def test_get_data_path_different_years():
    root = Path("/test/path")
    start = (2022, 12)
    end = (2023, 2)

    expected = Path("/test/path/data/raw/yellow_tripdata_2022-12_2023-02.parquet")
    result = get_data_path(root, start, end)
    assert result == expected


def test_download_parquet_file_exists(tmp_path):
    filepath = tmp_path / "test.parquet"
    filepath.write_text("existing data")

    result = download_parquet_file("http://example.com/test.parquet", filepath)
    assert result is True


def test_download_parquet_file_success(tmp_path):
    filepath = tmp_path / "test.parquet"

    mock_response = Mock()
    mock_response.ok = True
    mock_response.content = b"parquet data"

    with patch("requests.get", return_value=mock_response):
        result = download_parquet_file("http://example.com/test.parquet", filepath)

    assert result is True
    assert filepath.read_bytes() == b"parquet data"


def test_download_parquet_file_with_session(tmp_path):
    filepath = tmp_path / "test.parquet"

    mock_session = Mock()
    mock_response = Mock()
    mock_response.ok = True
    mock_response.content = b"parquet data"
    mock_session.get.return_value = mock_response

    result = download_parquet_file(
        "http://example.com/test.parquet", filepath, mock_session
    )

    assert result is True
    assert filepath.read_bytes() == b"parquet data"
    mock_session.get.assert_called_once_with(
        "http://example.com/test.parquet", stream=True
    )


def test_download_parquet_file_http_error(tmp_path, caplog):
    filepath = tmp_path / "test.parquet"

    mock_response = Mock()
    mock_response.ok = False
    mock_response.status_code = 404

    with patch("requests.get", return_value=mock_response):
        result = download_parquet_file("http://example.com/test.parquet", filepath)

    assert result is False
    assert not filepath.exists()
    assert "Failed to download" in caplog.text
    assert "Status code: 404" in caplog.text


def test_download_parquet_file_network_error(tmp_path, caplog):
    filepath = tmp_path / "test.parquet"

    with patch("requests.get", side_effect=requests.RequestException("Network error")):
        result = download_parquet_file("http://example.com/test.parquet", filepath)

    assert result is False
    assert not filepath.exists()
    assert "Network error downloading" in caplog.text


def test_concatenate_parquet_files_empty_list(tmp_path):
    output_path = tmp_path / "output.parquet"

    with pytest.raises(
        FileNotFoundError, match="No parquet files provided for concatenation"
    ):
        concatenate_parquet_files([], output_path)


def test_concatenate_parquet_files(tmp_path):
    file1 = tmp_path / "file1.parquet"
    file2 = tmp_path / "file2.parquet"
    output_path = tmp_path / "output.parquet"

    df1 = pl.DataFrame(
        {
            "tpep_pickup_datetime": ["2023-01-01 10:00:00", "2023-01-01 11:00:00"],
            "trip_distance": [1.0, 2.0],
        }
    ).with_columns(pl.col("tpep_pickup_datetime").str.to_datetime())

    df2 = pl.DataFrame(
        {
            "tpep_pickup_datetime": ["2023-01-01 09:00:00", "2023-01-01 12:00:00"],
            "trip_distance": [1.5, 2.5],
        }
    ).with_columns(pl.col("tpep_pickup_datetime").str.to_datetime())

    df1.write_parquet(file1)
    df2.write_parquet(file2)

    concatenate_parquet_files([file1, file2], output_path)

    assert output_path.exists()
    result_df = pl.read_parquet(output_path)
    assert len(result_df) == 4
    assert result_df["tpep_pickup_datetime"].is_sorted()


@patch("requests.Session")
@patch("e2e_taxi_ride_duration_prediction.ingestion.concatenate_parquet_files")
@patch("e2e_taxi_ride_duration_prediction.ingestion.download_parquet_file")
def test_get_nyc_taxi_data_download_and_concatenate(
    mock_download, mock_concat, mock_session, tmp_path
):
    def mock_download_side_effect(url, filepath, session=None):
        sample_df = pl.DataFrame(
            {"tpep_pickup_datetime": ["2023-01-01 10:00:00"], "trip_distance": [1.0]}
        ).with_columns(pl.col("tpep_pickup_datetime").str.to_datetime())

        filepath.parent.mkdir(parents=True, exist_ok=True)
        sample_df.write_parquet(filepath)
        return True

    mock_download.side_effect = mock_download_side_effect

    def mock_concat_side_effect(file_paths, output_path):
        sample_df = pl.DataFrame(
            {"tpep_pickup_datetime": ["2023-01-01 10:00:00"], "trip_distance": [1.0]}
        ).with_columns(pl.col("tpep_pickup_datetime").str.to_datetime())

        output_path.parent.mkdir(parents=True, exist_ok=True)
        sample_df.write_parquet(output_path)

    mock_concat.side_effect = mock_concat_side_effect

    result = get_nyc_taxi_data(tmp_path, start=(2023, 1), end=(2023, 2))

    assert mock_download.call_count == 2
    mock_concat.assert_called_once()

    assert isinstance(result, pl.LazyFrame)


def test_get_nyc_taxi_data_existing_file(tmp_path, caplog):
    caplog.set_level(logging.INFO)

    output_file = tmp_path / "data/raw/yellow_tripdata_2023-01_2023-02.parquet"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    sample_df = pl.DataFrame(
        {"tpep_pickup_datetime": ["2023-01-01 10:00:00"], "trip_distance": [1.0]}
    ).with_columns(pl.col("tpep_pickup_datetime").str.to_datetime())

    sample_df.write_parquet(output_file)

    result = get_nyc_taxi_data(tmp_path, start=(2023, 1), end=(2023, 2))

    assert "Found existing parquet file" in caplog.text
    assert isinstance(result, pl.LazyFrame)


@patch("requests.Session")
@patch("e2e_taxi_ride_duration_prediction.ingestion.download_parquet_file")
def test_get_nyc_taxi_data_no_files_downloaded(mock_download, mock_session, tmp_path):
    mock_download.return_value = False

    with pytest.raises(FileNotFoundError, match="No parquet files were downloaded"):
        get_nyc_taxi_data(tmp_path, start=(2023, 1), end=(2023, 2))


@patch("requests.Session")
def test_get_nyc_taxi_data_network_error(mock_session, tmp_path, caplog):
    mock_session.side_effect = requests.RequestException("Network error")

    with pytest.raises(requests.RequestException):
        get_nyc_taxi_data(tmp_path, start=(2023, 1), end=(2023, 2))

    assert "Network error occurred" in caplog.text
