import tempfile
from pathlib import Path
from typing import List, Tuple

import polars as pl
import requests
from loguru import logger
from prefect import flow, task
from tqdm.auto import tqdm


@task
def generate_year_month_tuples(
    start: Tuple[int, int], end: Tuple[int, int]
) -> List[Tuple[int, int]]:
    """Generate year-month tuples for the given date range.

    Args:
        start: (year, month) tuple for start date
        end: (year, month) tuple for end date

    Returns:
        List of (year, month) tuples in the range
    """
    start_year, start_month = start
    end_year, end_month = end
    return [
        (year, month)
        for year in range(start_year, end_year + 1)
        for month in range(1, 13)
        if (year, month) >= (start_year, start_month)
        and (year, month) <= (end_year, end_month)
    ]


@task
def get_data_path(root: Path, start: Tuple[int, int], end: Tuple[int, int]) -> Path:
    """Get the path for combined parquet file.

    Args:
        root: Root directory path
        start: (year, month) tuple for start date
        end: (year, month) tuple for end date

    Returns:
        Path to the cached parquet file
    """
    return (
        root
        / f"data/raw/yellow_tripdata_{start[0]:04d}-{start[1]:02d}_{end[0]:04d}-{end[1]:02d}.parquet"
    )


@task
def download_parquet_file(
    url: str, filepath: Path, session: requests.Session | None = None
) -> bool:
    """Download a parquet file from URL to filepath.

    Args:
        url: URL to download from
        filepath: Local path to save file
        session: Optional requests session for connection pooling

    Returns:
        True if download successful, False otherwise
    """
    if filepath.exists():
        return True

    http_client = session or requests
    try:
        r = http_client.get(url, stream=True)
        if r.ok:
            with open(filepath, "wb") as out:
                out.write(r.content)
            return True
        else:
            logger.warning(f"Failed to download {url}. Status code: {r.status_code}")
            return False
    except requests.RequestException as e:
        logger.error(f"Network error downloading {url}: {str(e)}")
        return False


@task
def concatenate_parquet_files(file_paths: List[Path], output_path: Path) -> None:
    """Concatenate multiple parquet files into a single file.

    Args:
        file_paths: List of parquet file paths to concatenate
        output_path: Path for the output concatenated file

    Raises:
        FileNotFoundError: If no valid parquet files found
    """
    if not file_paths:
        raise FileNotFoundError("No parquet files provided for concatenation.")

    lfs = [
        pl.scan_parquet(x)
        for x in tqdm(
            file_paths,
            desc="Reading temp Parquet files.",
            total=len(file_paths),
        )
    ]

    logger.info("Concatenating and sorting the data.")
    lf = pl.concat(lfs, how="diagonal_relaxed", rechunk=True).sort(
        "tpep_pickup_datetime"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving concatenated parquet file to {output_path}")
    lf.sink_parquet(output_path.resolve(), engine="streaming")


@flow
def get_nyc_taxi_data(
    root: Path | None = None,
    start: Tuple[int, int] = (2022, 1),
    end: Tuple[int, int] = (2025, 5),
) -> pl.LazyFrame:
    if not root:
        root = Path(__file__).parents[1]
    try:
        output_file = get_data_path(root, start, end)

        if output_file.exists():
            logger.info(
                f"Found existing parquet file for NYC Taxi data from {start[0]}-{start[1]} to {end[0]}-{end[1]}. Loading it."
            )
        else:
            logger.info(
                f"Downloading NYC Taxi data from {start[0]}-{start[1]} to {end[0]}-{end[1]}."
            )

            # Generate date range and download files
            year_month_tuples = generate_year_month_tuples(start, end)

            with tempfile.TemporaryDirectory() as temp_dir:
                temp_folder_path = Path(temp_dir)
                base_url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{:04d}-{:02d}.parquet"

                with requests.Session() as session:
                    for year, month in tqdm(
                        year_month_tuples, desc="Downloading NYC Taxi Data"
                    ):
                        file_path = (
                            temp_folder_path
                            / f"yellow_tripdata_{year:04d}-{month:02d}.parquet"
                        )
                        url = base_url.format(year, month)
                        download_parquet_file(url, file_path, session)

                logger.info("Concatenating all downloaded parquet files.")
                parquet_files = [
                    f
                    for f in temp_folder_path.iterdir()
                    if f.is_file() and f.suffix == ".parquet"
                ]

                if not parquet_files:
                    raise FileNotFoundError(
                        "No parquet files were downloaded to the temporary directory."
                    )

                concatenate_parquet_files(parquet_files, output_file)

        return pl.scan_parquet(output_file)

    except requests.RequestException as e:
        logger.error(f"Network error occurred: {str(e)}")
        raise
    except Exception as e:
        logger.error(
            f"An error occurred while downloading or processing the NYC Taxi data: {str(e)}"
        )
        raise
