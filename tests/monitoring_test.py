import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import polars as pl

from e2e_taxi_ride_duration_prediction.monitoring import (
    add_predictions_to_data,
    generate_monitoring_report,
)


def test_add_predictions_to_data():
    test_data = pl.LazyFrame(
        {
            "PULocationID": [1, 2],
            "DOLocationID": [3, 4],
            "trip_distance": [1.0, 2.0],
            "tpep_pickup_datetime": [
                datetime(2025, 1, 1, 10, 0),
                datetime(2025, 1, 1, 11, 0),
            ],
            "tpep_dropoff_datetime": [
                datetime(2025, 1, 1, 10, 15),
                datetime(2025, 1, 1, 11, 30),
            ],
        }
    )

    mock_model = Mock()
    mock_model.predict.return_value = [15.0, 30.0]
    mock_vectorizer = Mock()

    with (
        patch("builtins.open", mock=Mock()),
        patch("joblib.load", return_value=(mock_model, mock_vectorizer)),
    ):
        result_df = add_predictions_to_data(test_data, "dummy_model.pkl").collect()

        assert result_df["prediction"].to_list() == [15.0, 30.0]
        assert "duration" in result_df.columns


def test_generate_monitoring_report():
    test_data = pl.DataFrame(
        {
            "PULocationID": [1, 2],
            "DOLocationID": [3, 4],
            "trip_distance": [1.0, 2.0],
            "duration": [15.0, 30.0],
        }
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        ref_path = Path(temp_dir) / "ref.parquet"
        current_path = Path(temp_dir) / "current.parquet"
        report_path = str(Path(temp_dir) / "report.html")

        test_data.write_parquet(ref_path)
        test_data.write_parquet(current_path)

        mock_run = Mock()

        with (
            patch(
                "e2e_taxi_ride_duration_prediction.monitoring.add_predictions_to_data"
            ) as mock_add_pred,
            patch(
                "e2e_taxi_ride_duration_prediction.monitoring.Report"
            ) as mock_report_class,
        ):
            mock_report = Mock()
            mock_report.run.return_value = mock_run
            mock_report_class.return_value = mock_report

            mock_add_pred.return_value = test_data.lazy().with_columns(
                pl.lit(20.0).alias("prediction")
            )

            result = generate_monitoring_report(
                reference_data_path=ref_path,
                current_data_path=current_path,
                report_path=report_path,
                model_path="dummy_model.pkl",
            )

            assert result is mock_run
            mock_run.save_html.assert_called_once_with(report_path)
