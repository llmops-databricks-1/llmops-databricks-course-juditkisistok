# Databricks notebook source
# COMMAND ----------
import polars as pl
import wikipediaapi
from loguru import logger
from pyspark.sql import SparkSession

from eurovision_voting_bloc_party.config import ProjectConfig, get_env, load_config
from eurovision_voting_bloc_party.data_processors import (
    DataProcessor,
    KaggleProcessor,
    WikipediaProcessor,
)

spark = SparkSession.builder.getOrCreate()

# Load config
env = get_env(spark)
cfg = load_config("project_config.yaml", env)

CATALOG = cfg.catalog
SCHEMA = cfg.schema

# Create schema if it doesn't exist
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
logger.info(f"Schema {CATALOG}.{SCHEMA} ready")
# COMMAND ----------
# Step 1: process ArXiv data
data_processor = DataProcessor(spark=spark, config=cfg)

logger.info(f"Catalog: {cfg.catalog}, Schema: {cfg.schema}, Volume: {cfg.volume}")

data_processor.process_and_save()

# COMMAND ----------
# Step 2: process Wikipedia data
wikipedia_data_processor = WikipediaProcessor(spark=spark, config=cfg)
wikipedia_data_processor.process_and_save()


# COMMAND ----------
class WikipediaProcessor:
    def __init__(self, spark: SparkSession, config: ProjectConfig) -> None:
        self.spark = spark
        self.cfg = config
        self.years = [str(year) for year in range(1956, 2026)]
        self.wiki = wikipediaapi.Wikipedia(
            user_agent="EurovisionVotingBlocParty/1.0", language="en"
        )

    def fetch_wikipedia_page(self, year: str) -> dict | None:
        """Fetch a Wikipedia page for a Eurovision Song Contest year.

        Args:
            wiki: Initialized Wikipedia API client.
            year: The contest year as a string, e.g. '2024'.

        Returns:
            Dict with year, title, text, and summary, or None if the page doesn't exist.
        """
        page = self.wiki.page(f"Eurovision_Song_Contest_{year}")
        if page.exists():
            return {
                "year": year,
                "title": page.title,
                "summary": page.summary,
                "sections": self._extract_sections(page.sections),
            }

    def _extract_sections(self, text: str) -> list[dict]:
        result = []
        for section in text:
            if section.text.strip():
                result.append(
                    {
                        "section_title": section.title,
                        "section_text": section.text,
                    }
                )
            result.extend(self._extract_sections(section.sections))
        return result

    def _flatten_page(self, page: dict) -> dict:
        flat_df = (
            pl.DataFrame(page["sections"])
            .with_columns(pl.lit(page["year"]).alias("year"))
            .with_columns(pl.lit(page["title"]).alias("title"))
            .with_columns(pl.lit(page["summary"]).alias("summary"))
            .with_columns(
                pl.concat_str(
                    [
                        pl.col("year"),
                        pl.col("section_title")
                        .str.replace_all(" ", "_")
                        .str.to_lowercase(),
                    ],
                    separator="_",
                ).alias("chunk_id")
            )
        )

        return flat_df

    def get_all_wikipedia_pages(self) -> list[dict]:
        wikipedia_data = [
            self._flatten_page(page)
            for year in self.years
            if (page := self.fetch_wikipedia_page(year)) is not None
        ]
        return pl.concat(wikipedia_data)

    def process_and_save(self) -> None:
        all_pages = self.get_all_wikipedia_pages()
        spark_df = self.spark.createDataFrame(all_pages.to_arrow())
        # TODO: check ids for duplicates and add suffix if needed
        # TODO: write to Delta, enable CDF
        return all_pages, spark_df


wikipedia_data_processor = WikipediaProcessor(spark=spark, config=cfg)
hi = wikipedia_data_processor.process_and_save()

# COMMAND ----------


# COMMAND ----------
# Step 2: process Kaggle data
kaggle_data_processor = KaggleProcessor(spark=spark, config=cfg)
kaggle_data_processor.process_and_save()
