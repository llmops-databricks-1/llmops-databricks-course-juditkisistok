import json
import os
import re
import time
import urllib.request

import arxiv
import polars as pl
import wikipediaapi
from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql import types as T
from pyspark.sql.functions import (
    col,
    concat_ws,
    current_timestamp,
    explode,
    udf,
)
from pyspark.sql.types import ArrayType, StringType, StructField, StructType

from eurovision_voting_bloc_party.config import ProjectConfig
from eurovision_voting_bloc_party.utils import (
    load_eurovision_data_from_kaggle,
    prepare_eurovision_tabular_data,
)


class DataProcessor:
    """
    DataProcessor handles the complete workflow of:
    - Downloading papers from arXiv
    - Storing paper metadata
    - Parsing PDFs with ai_parse_document
    - Extracting and cleaning text chunks
    - Saving chunks to Delta tables
    """

    def __init__(self, spark: SparkSession, config: ProjectConfig) -> None:
        """
        Initialize DataProcessor with Spark session and configuration.

        Args:
            spark: SparkSession instance
            config: ProjectConfig object with table configurations
        """
        self.spark = spark
        self.cfg = config
        self.catalog = config.catalog
        self.schema = config.schema
        self.volume = config.volume
        self.end = time.strftime("%Y%m%d%H%M", time.gmtime())
        self.pdf_dir = f"/Volumes/{self.catalog}/{self.schema}/{self.volume}/{self.end}"
        os.makedirs(self.pdf_dir, exist_ok=True)
        self.papers_table = f"{self.catalog}.{self.schema}.eurovision_arxiv_papers"
        self.parsed_table = f"{self.catalog}.{self.schema}.ai_parsed_docs_table"

    def download_and_store_papers(
        self,
    ) -> list[dict] | None:
        """
        Download papers from arxiv and store metadata
        in arxiv_papers table.

        Returns:
            List of paper metadata dictionaries if papers were downloaded,
            otherwise None
        """
        # check if paper table exists and load the set of already processed arxiv_ids
        # to avoid duplicates
        if self.spark.catalog.tableExists(self.papers_table):
            existing_ids = {
                row.arxiv_id
                for row in self.spark.table(self.papers_table)
                .select("arxiv_id")
                .collect()
            }
            logger.info(
                f"Found {len(existing_ids)} existing papers in {self.papers_table}"
            )
        else:
            existing_ids = set()
            logger.info(f"No existing papers table found at {self.papers_table}.")

        # Search for papers in arxiv
        client = arxiv.Client(delay_seconds=10, num_retries=3)
        search = arxiv.Search(query="eurovision")
        papers = client.results(search)

        # Download papers and collect metadata
        records = []

        for paper in papers:
            logger.info(f"Processing paper {paper.get_short_id()}")
            paper_id = paper.get_short_id().replace("/", "_")
            if paper_id in existing_ids:
                logger.info(f"Skipping paper {paper_id} as it already exists.")
                continue
            try:
                urllib.request.urlretrieve(
                    paper.pdf_url, f"{self.pdf_dir}/{paper_id}.pdf"
                )
                # Collect metadata
                records.append(
                    {
                        "arxiv_id": paper_id,
                        "title": paper.title,
                        "authors": [author.name for author in paper.authors],
                        "summary": paper.summary,
                        "pdf_url": paper.pdf_url,
                        "published": int(paper.published.strftime("%Y%m%d%H%M")),
                        "processed": int(self.end),
                        "volume_path": f"{self.pdf_dir}/{paper_id}.pdf",
                    }
                )
            except Exception as e:
                logger.warning(f"Paper {paper_id} was not successfully processed: {e}.")
            # Avoid hitting API rate limits
            time.sleep(3)

        # Only process if we have records
        if len(records) == 0:
            logger.info("No new papers found.")
            return None

        logger.info(f"Downloaded {len(records)} papers to {self.pdf_dir}")

        # Create DataFrame and save to arxiv_papers table
        schema = T.StructType(
            [
                T.StructField("arxiv_id", T.StringType(), False),
                T.StructField("title", T.StringType(), True),
                T.StructField("authors", T.ArrayType(T.StringType()), True),
                T.StructField("summary", T.StringType(), True),
                T.StructField("pdf_url", T.StringType(), True),
                T.StructField("published", T.LongType(), True),
                T.StructField("processed", T.LongType(), True),
                T.StructField("volume_path", T.StringType(), True),
            ]
        )

        metadata_df = self.spark.createDataFrame(records, schema=schema).withColumn(
            "ingest_ts", current_timestamp()
        )

        # Create table if it doesn't exist
        metadata_df.write.format("delta").mode("ignore").saveAsTable(self.papers_table)

        # MERGE to avoid duplicates based on arxiv_id
        metadata_df.createOrReplaceTempView("new_papers")
        self.spark.sql(f"""
            MERGE INTO {self.papers_table} target
            USING new_papers source
            ON target.arxiv_id = source.arxiv_id
            WHEN NOT MATCHED THEN INSERT (
                arxiv_id, title, authors, summary, pdf_url,
                published, processed, volume_path
            ) VALUES (
                source.arxiv_id, source.title, source.authors,
                source.summary, source.pdf_url, source.published,
                source.processed, source.volume_path
            )
        """)
        logger.info(f"Merged {len(records)} paper records into {self.papers_table}")
        return records

    def parse_pdfs_with_ai(self) -> None:
        """
        Parse PDFs using ai_parse_document and store in ai_parsed_docs table.

        """

        self.spark.sql(f"""
            CREATE TABLE IF NOT EXISTS {self.parsed_table} (
                path STRING,
                parsed_content STRING,
                processed LONG
            )
        """)

        self.spark.sql(f"""
            INSERT INTO {self.parsed_table}
            SELECT
                path,
                ai_parse_document(content) AS parsed_content,
                {self.end} AS processed
            FROM READ_FILES(
                "{self.pdf_dir}",
                format => 'binaryFile'
            )
        """)

        logger.info(f"Parsed PDFs from {self.pdf_dir} and saved to {self.parsed_table}")

    @staticmethod
    def _extract_chunks(parsed_content_json: str) -> list[tuple[str, str]]:
        """
        Extract chunks from parsed_content JSON.

        Args:
            parsed_content_json: JSON string containing
            parsed document structure

        Returns:
            List of tuples containing (chunk_id, content)
        """
        parsed_dict = json.loads(parsed_content_json)
        chunks = []

        for element in parsed_dict.get("document", {}).get("elements", []):
            if element.get("type") == "text":
                chunk_id = element.get("id", "")
                content = element.get("content", "")
                chunks.append((chunk_id, content))

        return chunks

    @staticmethod
    def _extract_paper_id(path: str) -> str:
        """
        Extract paper ID from file path.

        Args:
            path: File path (e.g., "/path/to/paper_id.pdf")

        Returns:
            Paper ID extracted from the path
        """
        return path.replace(".pdf", "").split("/")[-1]

    @staticmethod
    def _clean_chunk(text: str) -> str:
        """
        Clean and normalize chunk text
        Args:
            text: Raw text content

        Returns:
            Cleaned text content
        """
        # Fix hyphenation across line breaks:
        # "docu-\nments" => "documents"
        t = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)

        # Collapse internal newlines into spaces
        t = re.sub(r"\s*\n\s*", " ", t)

        # Collapse repeated whitespace
        t = re.sub(r"\s+", " ", t)

        return t.strip()

    def process_chunks(self) -> None:
        """
        Process parsed documents to extract and clean chunks.
        Reads from ai_parsed_docs table and saves to arxiv_chunks table.
        """
        logger.info(
            f"Processing parsed documents from "
            f"{self.parsed_table} for end date {self.end}"
        )

        df = self.spark.table(self.parsed_table).where(f"processed = {self.end}")

        # Define schema for the extracted chunks
        chunk_schema = ArrayType(
            StructType(
                [
                    StructField("chunk_id", StringType(), True),
                    StructField("content", StringType(), True),
                ]
            )
        )

        extract_chunks_udf = udf(self._extract_chunks, chunk_schema)
        extract_paper_id_udf = udf(self._extract_paper_id, StringType())
        clean_chunk_udf = udf(self._clean_chunk, StringType())

        metadata_df = self.spark.table(self.papers_table).select(
            col("arxiv_id"),
            col("title"),
            col("summary"),
            concat_ws(", ", col("authors")).alias("authors"),
            (col("published") / 100000000).cast("int").alias("year"),
            ((col("published") % 100000000) / 1000000).cast("int").alias("month"),
            ((col("published") % 1000000) / 10000).cast("int").alias("day"),
        )

        # Create the transformed dataframe
        chunks_df = (
            df.withColumn("arxiv_id", extract_paper_id_udf(col("path")))
            .withColumn("chunks", extract_chunks_udf(col("parsed_content")))
            .withColumn("chunk", explode(col("chunks")))
            .select(
                col("arxiv_id"),
                col("chunk.chunk_id").alias("chunk_id"),
                clean_chunk_udf(col("chunk.content")).alias("text"),
                concat_ws("_", col("arxiv_id"), col("chunk.chunk_id")).alias("id"),
            )
            .join(metadata_df, "arxiv_id", "left")
        )

        # Write to table
        arxiv_chunks_table = f"{self.catalog}.{self.schema}.arxiv_chunks_table"
        chunks_df.write.mode("append").saveAsTable(arxiv_chunks_table)
        logger.info(f"Saved chunks to {arxiv_chunks_table}")

        # Enable Change Data Feed
        self.spark.sql(f"""
            ALTER TABLE {arxiv_chunks_table}
            SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
        """)
        logger.info(f"Change Data Feed enabled for {arxiv_chunks_table}")

    def process_and_save(self) -> None:
        """
        Complete workflow: download papers, parse PDFs, and process chunks.
        """
        # Step 1: Download papers and store metadata
        records = self.download_and_store_papers()

        # Only continue if we have new papers
        if records is None:
            logger.info("No new papers to process. Exiting.")
            return False

        # Step 2: Parse PDFs with ai_parse_document
        self.parse_pdfs_with_ai()
        logger.info("Parsed documents.")

        # Step 3: Process chunks
        self.process_chunks()
        logger.info("Processing complete!")
        return True


class WikipediaProcessor:
    """Fetches Eurovision Wikipedia pages, splits them into chunks, and saves to Delta."""

    def __init__(self, spark: SparkSession, config: ProjectConfig) -> None:
        """Initialize WikipediaProcessor with Spark session and configuration.

        Args:
            spark: SparkSession instance
            config: ProjectConfig object with table configurations
        """
        self.spark = spark
        self.cfg = config
        self.catalog = config.catalog
        self.schema = config.schema
        self.volume = config.volume
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

    def _split_into_chunks(self, text: str, chunk_size: int = 1000) -> list[str]:
        """Split text into chunks, preferring natural boundaries over hard cuts.

        Tries paragraph breaks, then sentence boundaries, then word boundaries,
        falling back to hard character splits only as a last resort.

        Args:
            text: Text to split
            chunk_size: Maximum character length per chunk

        Returns:
            List of text chunks each within chunk_size characters
        """
        if len(text) <= chunk_size:
            return [text]

        for sep in ["\n\n", ". ", " "]:
            parts = text.split(sep)
            if len(parts) == 1:
                continue

            chunks, current_chunk = [], ""
            for part in parts:
                candidate_chunk = (
                    (current_chunk + sep + part).strip() if current_chunk else part
                )
                if len(candidate_chunk) <= chunk_size:
                    current_chunk = candidate_chunk
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = part
            if current_chunk:
                chunks.append(current_chunk)
            return chunks

        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    def _extract_sections(self, sections: list) -> list[dict]:
        """Recursively extract and chunk text from Wikipedia page sections.

        Args:
            sections: List of wikipediaapi section objects

        Returns:
            List of dicts with section_title, text, and sub_chunk_idx
        """
        result = []
        for section in sections:
            if section.text.strip():
                for idx, chunk in enumerate(self._split_into_chunks(section.text)):
                    result.append(
                        {
                            "section_title": section.title,
                            "text": chunk,
                            "sub_chunk_idx": idx,
                        }
                    )
            result.extend(self._extract_sections(section.sections))
        return result

    def _flatten_page(self, page: dict) -> pl.DataFrame:
        """Flatten a page dict into a Polars DataFrame with one row per chunk.

        Adds year, title, summary, and a unique chunk_id to each row.

        Args:
            page: Dict with keys year, title, summary, and sections

        Returns:
            Polars DataFrame with columns: section_title, text, sub_chunk_idx,
            year, title, summary, chunk_id
        """
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
                        pl.col("sub_chunk_idx").cast(pl.String),
                    ],
                    separator="_",
                ).alias("chunk_id")
            )
        )

        return flat_df

    def get_all_wikipedia_pages(self) -> pl.DataFrame:
        """Fetch and flatten all Eurovision Wikipedia pages for years 1956–2025.

        Returns:
            Concatenated Polars DataFrame of all pages and their chunks
        """
        wikipedia_data = [
            self._flatten_page(page)
            for year in self.years
            if (page := self.fetch_wikipedia_page(year)) is not None
        ]
        return pl.concat(wikipedia_data)

    def process_and_save(self) -> None:
        """Fetch all Wikipedia pages, write chunks to Delta, and enable CDF."""
        all_pages = self.get_all_wikipedia_pages()
        spark_df = self.spark.createDataFrame(all_pages.to_arrow())

        wiki_table = f"{self.catalog}.{self.schema}.eurovision_wikipedia_chunks"
        spark_df.write.format("delta").mode("overwrite").saveAsTable(wiki_table)
        logger.info(f"Saved Wikipedia data to {wiki_table}")

        self.spark.sql(f"""
            ALTER TABLE {wiki_table}
            SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
        """)
        logger.info(f"Change Data Feed enabled for {wiki_table}")


class KaggleProcessor:
    """Loads Eurovision Kaggle datasets and produces one text summary per country."""

    def __init__(self, spark: SparkSession, config: ProjectConfig) -> None:
        """Initialize KaggleProcessor with Spark session and configuration.

        Args:
            spark: SparkSession instance
            config: ProjectConfig object with table configurations
        """
        self.spark = spark
        self.cfg = config

    def _aggregate_data_by_country(self) -> pl.DataFrame:
        """Load and join Kaggle datasets, then aggregate statistics per country.

        Returns:
            Polars DataFrame with one row per country containing participation
            stats, win history, languages, and musical styles
        """
        kaggle_dict = {
            c: load_eurovision_data_from_kaggle(c) for c in ["contest", "country", "song"]
        }
        contest_df = prepare_eurovision_tabular_data(kaggle_dict)

        country_stats = (
            contest_df.sort("year")
            .group_by("country")
            .agg(
                [
                    pl.len().alias("participations"),
                    (pl.col("final_place") == 1).sum().alias("wins"),
                    pl.col("final_place").mean().round(1).alias("avg_place"),
                    pl.col("final_place").min().alias("best_place"),
                    pl.col("year").sort().alias("years"),
                    pl.col("participant_region").first().alias("region"),
                    pl.col("language").drop_nulls().unique().sort().alias("languages"),
                    pl.col("style").drop_nulls().unique().sort().alias("styles"),
                    pl.struct(["year", "song_name", "artist_name", "language", "style"])
                    .filter(pl.col("final_place") == 1)
                    .alias("winning_entries"),
                ]
            )
        )
        return country_stats

    def _format_summary(self, row: dict) -> str:
        """Serialize a country's aggregated stats into a natural language summary.

        Args:
            row: Dict with country stats from _aggregate_data_by_country

        Returns:
            Natural language string describing the country's Eurovision history
        """
        years = sorted(row["years"])
        year_range = f"{years[0]}–{years[-1]}"
        if row["wins"] > 0:
            wins_list = ", ".join(
                f"'{e['song_name']}' by {e['artist_name']} "
                f"({e['year']}, {e['language']}, {e['style']})"
                for e in row["winning_entries"]
            )
            wins_text = f" They have won {row['wins']} time(s): {wins_list}."
        else:
            wins_text = " They have never won."
        languages_text = f" They have performed in: {', '.join(row['languages'])}."
        styles_text = f" Their musical styles include: {', '.join(row['styles'])}."
        avg_place_text = (
            f" Their average placement is {row['avg_place']},"
            if row["avg_place"] is not None
            else ""
        )
        best_place_text = (
            f" with a best placement of {int(row['best_place'])}."
            if row["best_place"] is not None
            else ""
        )
        return (
            f"{row['country']} has participated in Eurovision"
            f"{row['participations']} times "
            f"({year_range}), representing {row['region']}."
            f"{wins_text}"
            f" {avg_place_text} "
            f"{best_place_text}"
            f"{languages_text}"
            f"{styles_text}"
        )

    def process_and_save(self) -> None:
        """Aggregate Kaggle data by country, write text summaries to Delta,
        and enable CDF."""
        country_stats = self._aggregate_data_by_country()
        country_stats = country_stats.with_columns(
            pl.struct(country_stats.columns)
            .map_elements(self._format_summary, return_dtype=pl.String)
            .alias("text"),
            pl.col("country")
            .str.replace_all(" ", "_")
            .str.to_lowercase()
            .alias("chunk_id"),
        ).select(["chunk_id", "text"])

        kaggle_table = f"{self.cfg.catalog}.{self.cfg.schema}.eurovision_kaggle_chunks"
        self.spark.createDataFrame(country_stats.to_arrow()).write.format("delta").mode(
            "overwrite"
        ).saveAsTable(kaggle_table)

        self.spark.sql(f"""
            ALTER TABLE {kaggle_table}
            SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
        """)
