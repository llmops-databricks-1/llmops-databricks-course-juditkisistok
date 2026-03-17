import kagglehub
import polars as pl
from kagglehub import KaggleDatasetAdapter
from loguru import logger
from pyspark.sql import DataFrame as SparkDataFrame

## CONSTANTS
TABLE_NAME = "eurovision_data"
TABLE_NAME_WIKI = "eurovision_wikipedia"
TABLE_NAME_ARXIV = "eurovision_arxiv"


## FUNCTIONS
def load_eurovision_data_from_kaggle(kaggle_data_type: str) -> pl.DataFrame:
    """Load a Eurovision dataset CSV from Kaggle as a Polars DataFrame.

    Args:
        kaggle_data_type: One of 'contest', 'country', or 'song'.

    Returns:
        Collected Polars DataFrame for the requested dataset type.
    """
    data = kagglehub.dataset_load(
        KaggleDatasetAdapter.POLARS,
        "diamondsnake/eurovision-song-contest-data",
        path=f"Kaggle Dataset/{kaggle_data_type}_data.csv",
        polars_kwargs={"encoding": "utf8-lossy", "ignore_errors": True},
    )
    return data.collect()


def prepare_eurovision_tabular_data(kaggle_dict: dict) -> pl.DataFrame:
    """Join contest, country, and song data into a single Eurovision dataset.

    Args:
        kaggle_dict: Dict with keys 'contest', 'country', 'song'
            mapping to Polars DataFrames.

    Returns:
        Joined Polars DataFrame with host region and participant region columns added.
    """
    contest_data, country_data, song_data = (
        kaggle_dict["contest"],
        kaggle_dict["country"],
        kaggle_dict["song"],
    )

    hosts = contest_data.join(country_data, left_on="host", right_on="country").rename(
        {"region": "host_region"}
    )

    countries = song_data.join(country_data, on="country").rename(
        {"region": "participant_region"}
    )

    eurovision_dataset = hosts.join(countries, on="year")
    return eurovision_dataset


def write_to_delta_table(
    df: SparkDataFrame, catalog: str, schema: str, table_name: str
) -> None:
    """Write a Spark DataFrame to a Delta table in Unity Catalog.

    Args:
        df: Spark DataFrame to write.
        catalog: Unity Catalog name.
        schema: Schema name.
        table_name: Target table name.
    """
    table_path = f"{catalog}.{schema}.{table_name}"

    df.write.format("delta").mode("overwrite").option("mergeSchema", "true").saveAsTable(
        table_path
    )

    logger.info(f"Created Delta table: {table_path}")
    logger.info(f"Records: {df.count()}")
