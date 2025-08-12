from pathlib import Path

import joblib
import polars as pl
from evidently import DataDefinition, Dataset, Regression, Report
from evidently.core.report import Snapshot
from evidently.presets import DataDriftPreset, RegressionPreset
from prefect import task

from e2e_taxi_ride_duration_prediction.preprocessing import calculate_duration

pl.Config.set_engine_affinity("streaming")


@task
def add_predictions_to_data(
    data: pl.LazyFrame,
    model_path: str | Path,
    feature_columns: list[str] = ["PULocationID", "DOLocationID", "trip_distance"],
) -> pl.LazyFrame:
    """Add prediction column to data using trained model."""
    with open(model_path, "rb") as f:
        model, dict_vectorizer = joblib.load(f)

    data = calculate_duration(data)
    X_dicts = data.select(feature_columns).collect().to_dicts()
    X_features = dict_vectorizer.transform(X_dicts)
    predictions = model.predict(X_features)

    return data.with_columns(pl.Series("prediction", predictions))


@task
def generate_monitoring_report(
    reference_data_path: str | Path,
    current_data_path: str | Path,
    report_path: str,
    model_path: str | Path | None = None,
    feature_columns: list[str] = ["PULocationID", "DOLocationID", "trip_distance"],
    target: str = "duration",
    data_drift: bool = True,
    regression: bool = True,
    include_tests: bool = True,
) -> Snapshot:
    """
    Generate a monitoring report comparing reference and current data.

    Args:
        reference_data_path (str | Path): Path to the reference data.
        current_data_path (str | Path): Path to the current data.
        report_path (str | Path): Path to save the generated report.
        model_path (str | Path | None): Path to model for predictions (required for regression preset).
        data_drift (bool): Include data drift preset.
        regression (bool): Include regression preset.

    Returns:
        Snapshot: The generated monitoring report snapshot.
    """
    if data_drift and regression:
        metrics = [DataDriftPreset(), RegressionPreset()]
    elif data_drift and not regression:
        metrics = [DataDriftPreset()]
    elif not data_drift and regression:
        metrics = [RegressionPreset()]
    else:
        raise ValueError("At least one of data_drift or regression must be True.")

    if regression and model_path is None:
        raise ValueError("model_path is required when regression=True")

    reference_data = pl.scan_parquet(reference_data_path)
    current_data = pl.scan_parquet(current_data_path)

    if regression and model_path:
        reference_data = add_predictions_to_data(
            reference_data, model_path, feature_columns
        )
        current_data = add_predictions_to_data(
            current_data,
            model_path,
            feature_columns,
        )

    reference_data = Dataset.from_pandas(
        reference_data.collect().to_pandas(),
        data_definition=DataDefinition(
            regression=[Regression(target=target, prediction="prediction")]
        ),
    )
    current_data = Dataset.from_pandas(
        current_data.collect().to_pandas(),
        data_definition=DataDefinition(
            regression=[Regression(target=target, prediction="prediction")]
        ),
    )

    report = Report(metrics=metrics, include_tests=include_tests)
    run = report.run(reference_data, current_data)

    run.save_html(report_path)

    return run
