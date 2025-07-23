import logging
import tempfile
from pathlib import Path

import polars as pl
import requests
from tqdm.auto import tqdm


def get_nyc_taxi_data(root: Path, start=(2022, 1), end=(2025, 5)):
    try:
        # Check if the combined file already exists
        output_file = (
            root
            / f"data/raw/yellow_tripdata_{start[0]:04d}-{start[1]:02d}_{end[0]:04d}-{end[1]:02d}.parquet"
        )

        if output_file.exists():
            logging.info(
                f"Found existing parquet file for NYC Taxi data from {start[0]}-{start[1]} to {end[0]}-{end[1]}. Loading it."
            )
        else:
            logging.info(
                f"Downloading NYC Taxi data from {start[0]}-{start[1]} to {end[0]}-{end[1]}."
            )

            # Create year-month tuples for the date range
            start_year, start_month = start
            end_year, end_month = end
            year_month_tuples = [
                (year, month)
                for year in range(start_year, end_year + 1)
                for month in range(1, 13)
                if (year, month) >= (start_year, start_month)
                and (year, month) <= (end_year, end_month)
            ]

            with tempfile.TemporaryDirectory() as temp_dir:
                temp_folder_path = Path(temp_dir)

                base = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{:04d}-{:02d}.parquet"
                for year, month in tqdm(
                    year_month_tuples, desc="Downloading NYC Taxi Data"
                ):
                    file = (
                        temp_folder_path
                        / f"yellow_tripdata_{year:04d}-{month:02d}.parquet"
                    )
                    if not file.exists():
                        r = requests.get(base.format(year, month), stream=True)
                        if r.ok:
                            with open(file, "wb") as out:
                                out.write(r.content)
                        else:
                            logging.warning(
                                f"Failed to download data for {year}-{month}. Status code: {r.status_code}"
                            )

                logging.info("Concatenating all downloaded parquet files.")
                parquet_files = [
                    f
                    for f in temp_folder_path.iterdir()
                    if f.is_file() and f.suffix == ".parquet"
                ]

                if not parquet_files:
                    raise FileNotFoundError(
                        "No parquet files were downloaded to the temporary directory."
                    )

                lfs = [
                    pl.scan_parquet(x)
                    for x in tqdm(
                        parquet_files,
                        desc="Reading temp Parquet files.",
                        total=len(parquet_files),
                    )
                ]
                logging.info("Concatenating and sorting the data.")
                lf = pl.concat(lfs, how="diagonal_relaxed", rechunk=True).sort("tpep_pickup_datetime")

                output_file.parent.mkdir(parents=True, exist_ok=True)

                logging.info(f"Saving concatenated parquet file to {output_file}")
                lf.sink_parquet(output_file.resolve(), engine="streaming")

        return pl.scan_parquet(output_file)

    except requests.RequestException as e:
        logging.error(f"Network error occurred: {str(e)}")
        raise
    except Exception as e:
        logging.error(
            f"An error occurred while downloading or processing the NYC Taxi data: {str(e)}"
        )
        raise
