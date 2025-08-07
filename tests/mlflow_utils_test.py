import tempfile
from pathlib import Path
from unittest.mock import patch

from e2e_taxi_ride_duration_prediction.mlflow_utils import setup_mlflow


def test_setup_mlflow():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / "test.db"

        with (
            patch("mlflow.set_tracking_uri") as mock_set_uri,
            patch("mlflow.set_experiment") as mock_set_exp,
            patch("mlflow.sklearn.autolog"),
            patch("mlflow.xgboost.autolog"),
        ):
            result = setup_mlflow(custom_tracking_uri=temp_path)

            assert result is True
            mock_set_uri.assert_called_once_with(f"sqlite:///{temp_path}")
            mock_set_exp.assert_called_once_with("taxi_ride_duration_prediction")


def test_setup_mlflow_exception():
    with patch("mlflow.set_tracking_uri", side_effect=Exception("Test error")):
        result = setup_mlflow()
        assert result is False
