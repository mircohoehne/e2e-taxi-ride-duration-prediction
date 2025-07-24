import polars as pl


def calculate_duration(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.with_columns(
        (
            (
                pl.col("tpep_dropoff_datetime") - pl.col("tpep_pickup_datetime")
            ).dt.total_seconds()
            / 60
        ).alias("duration")
    )


def filter_by_date_range(
    lf: pl.LazyFrame, start: pl.Datetime, end: pl.Datetime
) -> pl.LazyFrame:
    return lf.filter(
        (pl.col("tpep_pickup_datetime") < end)
        & (pl.col("tpep_pickup_datetime") >= start)
    )


def filter_valid_durations(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.filter((pl.col("duration") > 0) & (pl.col("duration") <= 60))


def cast_categorical_columns(
    lf: pl.LazyFrame,
    categorical_columns: list[str] = [
        "VendorID",
        "RatecodeID",
        "store_and_fwd_flag",
        "PULocationID",
        "DOLocationID",
        "payment_type",
    ],
) -> pl.LazyFrame:
    return lf.with_columns(
        [pl.col(col).cast(pl.Utf8).cast(pl.Categorical) for col in categorical_columns]
    )


def create_pickup_dropoff_pairs(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.with_columns(
        pl.concat_str([pl.col("PULocationID"), pl.col("DOLocationID")], separator="_")
        .cast(pl.Categorical)
        .alias("pickup_dropoff_pair")
    )


def basic_preprocessing(
    lf: pl.LazyFrame, start: pl.Datetime, end: pl.Datetime
) -> pl.LazyFrame:
    """Basic preprocessing of the LazyFrame.

    The preprocessing includes the following steps:
        - Creating the duration column from pickup and dropoff times
        - Filter out dates outside of the given range
        - Filter out extreme and impossible durations (negative and longer than an hour)
        - Cast all categorical columns as pl.Categorical
        - Create new Column with pickup_dropoff LocationID pairs

    Args:
        lf: The LazyFrame that should be preprocessed.
        start: Datetime indicating the start of daterange.
        end: Datetime indicating the end of daterange
    Returns:
        The LazyFrame with preprocessing instructions.
        Keep in mind that the execution is lazy, i.e. the operations will be performed when .collect() is called.
    """
    return (
        lf.pipe(calculate_duration)
        .pipe(filter_by_date_range, start, end)
        .pipe(filter_valid_durations)
        .pipe(cast_categorical_columns)
        .pipe(create_pickup_dropoff_pairs)
    )
