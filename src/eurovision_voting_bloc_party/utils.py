from datetime import datetime

import arxiv
import kagglehub
import polars as pl
import wikipediaapi
from kagglehub import KaggleDatasetAdapter
from loguru import logger
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import SparkSession

TABLE_NAME = "eurovision_data"
TABLE_NAME_WIKI = "eurovision_wikipedia"
TABLE_NAME_ARXIV = "eurovision_arxiv"


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


def read_delta_table(
    spark: SparkSession, catalog: str, schema: str, table_name: str
) -> None:
    """Read a Delta table from Unity Catalog and log sample records.

    Args:
        spark: Active SparkSession.
        catalog: Unity Catalog name.
        schema: Schema name.
        table_name: Table name to read.
    """
    df = spark.table(f"{catalog}.{schema}.{table_name}")

    logger.info(f"Table: {catalog}.{schema}.{table_name}")
    logger.info(f"Total records: {df.count()}")
    logger.info("Schema:")
    df.printSchema()

    logger.info("Sample records:")
    df.show(5)


def fetch_wikipedia_page(wiki: wikipediaapi.Wikipedia, year: str) -> dict | None:
    """Fetch a Wikipedia page for a Eurovision Song Contest year.

    Args:
        wiki: Initialized Wikipedia API client.
        year: The contest year as a string, e.g. '2024'.

    Returns:
        Dict with year, title, text, and summary, or None if the page doesn't exist.
    """
    page = wiki.page(f"Eurovision_Song_Contest_{year}")
    if page.exists():
        return {
            "year": year,
            "title": page.title,
            "text": page.text,
            "summary": page.summary,
        }


def fetch_arxiv_data(query: str = "eurovision", max_results: int = 50) -> list[dict]:
    """Fetch papers from arXiv matching the given query.

    Args:
        query: Search query string.
        max_results: Maximum number of results to return.

    Returns:
        List of dicts with paper metadata including title, authors, summary, and PDF URL.
    """
    client = arxiv.Client(delay_seconds=3, num_retries=5)
    search = arxiv.Search(
        query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance
    )
    results = []
    for result in client.results(search):
        paper = {
            "arxiv_id": result.entry_id.split("/")[-1],
            "title": result.title,
            "authors": [author.name for author in result.authors],
            "summary": result.summary,
            "published": int(result.published.strftime("%Y%m%d%H%M")),
            "updated": result.updated.isoformat() if result.updated else None,
            "categories": ", ".join(result.categories),
            "pdf_url": result.pdf_url,
            "primary_category": result.primary_category,
            "ingestion_timestamp": datetime.now().isoformat(),
        }
        results.append(paper)
    logger.info(f"Fetched {len(results)} papers from arXiv for query '{query}'")
    return results


def create_sync_metadata_table(spark: SparkSession, catalog: str, schema: str) -> None:
    """Create a Delta table to track vector search index synchronization metadata.

    Args:
        spark: Active SparkSession.
        catalog: Unity Catalog name.
        schema: Schema name.
    """

    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {catalog}.{schema}.sync_metadata (
            source_table STRING NOT NULL,
    last_synced TIMESTAMP
        )
    """)

    spark.sql(f"""
          INSERT INTO {catalog}.{schema}.sync_metadata
          SELECT source, NULL AS last_synced
          FROM (VALUES ('eurovision'), ('kaggle')) AS t(source)
          WHERE NOT EXISTS (
              SELECT 1 FROM {catalog}.{schema}.sync_metadata
          )
          """)
    logger.info(f"Created sync metadata table: {catalog}.{schema}.sync_metadata")
