import kagglehub
import polars as pl
from kagglehub import KaggleDatasetAdapter
from loguru import logger


def load_eurovision_data_from_kaggle(kaggle_data_types: list) -> pl.DataFrame:
    data = kagglehub.dataset_load(
  KaggleDatasetAdapter.POLARS,
  "diamondsnake/eurovision-song-contest-data",
  path = f"Kaggle Dataset/{kaggle_data_types}_data.csv",
  polars_kwargs={"encoding": "utf8-lossy", "ignore_errors": True}
)
    return data.collect()

def prepare_eurovision_tabular_data(kaggle_dict: dict) -> pl.DataFrame:
    contest_data, country_data, song_data = (kaggle_dict["contest"], kaggle_dict["country"], 
                                            kaggle_dict["song"])

    hosts = (
        contest_data.join(country_data, left_on="host", right_on="country")
        .rename({"region": "host_region"})
    )

    countries = (
        song_data.join(country_data, on="country")
        .rename({"region": "participant_region"})
    )

    eurovision_dataset = (
        hosts.join(countries, on="year")
    )
    return eurovision_dataset

def write_to_delta_table(df: pl.DataFrame, catalog: str, schema: str, table_name: str) -> None:
    table_path = f"{catalog}.{schema}.{table_name}"

    df.write \
        .format("delta") \
        .mode("overwrite") \
        .option("mergeSchema", "true") \
        .saveAsTable(table_path)

    logger.info(f"Created Delta table: {table_path}")
    logger.info(f"Records: {df.count()}")

