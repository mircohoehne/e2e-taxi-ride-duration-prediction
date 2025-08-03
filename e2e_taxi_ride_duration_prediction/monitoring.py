from pathlib import Path

import polars as pl
from evidently import Report
from evidently.presets import DataDriftPreset, RegressionPreset
from prefect import task

pl.Config.set_engine_affinity("streaming")


@task
def generate_monitoring_report(
    reference_data_path: str | Path,
    current_data_path: str | Path,
    report_path: str,
    data_drift: bool = True,
    regression: bool = True,
) -> Report:
    """
    Generate a monitoring report comparing reference and current data.

    Args:
        reference_data_path (str | Path): Path to the reference data.
        current_data_path (str | Path): Path to the current data.
        report_path (str | Path): Path to save the generated report.

    Returns:
        Report: The generated monitoring report.
    """
    if data_drift and regression:
        metrics = [DataDriftPreset(), RegressionPreset()]
    elif data_drift and not regression:
        metrics = [DataDriftPreset()]
    elif not data_drift and regression:
        metrics = [RegressionPreset()]
    else:
        raise ValueError("At least one of data_drift or regression must be True.")

    reference_data = pl.scan_parquet(reference_data_path).collect().to_pandas()
    current_data = pl.scan_parquet(current_data_path).collect().to_pandas()

    report = Report(metrics=metrics)
    run = report.run(reference_data=reference_data, current_data=current_data)

    run.save_html(report_path)

    return report
