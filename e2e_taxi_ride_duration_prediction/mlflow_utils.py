from pathlib import Path

import mlflow
from loguru import logger


def setup_mlflow(
    custom_tracking_uri: str | Path | None = None,
    custom_experiment_name: str | None = None,
    autologging: bool = True,
    autolog_sklearn_params: dict | None = None,
    autolog_xgboost_params: dict | None = None,
) -> bool:
    """Setup Mlflow tracking.

    Initializes the MLflow tracking URI (sqlite database) and experiment name.

    Args:
        custom_tracking_uri (str | Path | None): Custom path for the MLflow tracking URI.
        custom_experiment_name (str | None): Custom name for the MLflow experiment.
        autologging (bool): Whether to enable autologging for sklearn and xgboost
        autolog_sklearn_params (dict | None): Parameters for sklearn autologging
        autolog_xgboost_params (dict | None): Parameters for xgboost autologging

    Returns:
        bool: True if setup is successful, False otherwise.
    """
    try:
        db_path = (
            custom_tracking_uri or Path(__file__).parent.parent / "mlruns" / "mlflow.db"
        )
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        experiment_name = custom_experiment_name or "taxi_ride_duration_prediction"
        mlflow.set_tracking_uri(f"sqlite:///{db_path}")
        mlflow.set_experiment(experiment_name)

        if autologging:
            sklearn_params = autolog_sklearn_params or {
                "log_input_examples": True,
                "log_model_signatures": True,
                "log_models": True,
                "log_datasets": False,
            }
            xgboost_params = autolog_xgboost_params or {
                "log_input_examples": True,
                "log_model_signatures": True,
                "log_models": True,
                "log_datasets": False,
            }

            mlflow.sklearn.autolog(**sklearn_params)
            mlflow.xgboost.autolog(**xgboost_params)

        logger.success("MLflow tracking URI and experiment set up successfully.")
        return True
    except Exception as e:
        logger.error(f"Failed to set up MLflow: {e}")
        return False


if __name__ == "__main__":
    print(setup_mlflow())
