import time

from databricks.sdk import WorkspaceClient
from databricks.vector_search.client import VectorSearchClient
from databricks.vector_search.index import VectorSearchIndex
from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from eurovision_voting_bloc_party.config import ProjectConfig


class VectorSearchManager:
    """Manages vector search endpoints and indexes for chunks."""

    def __init__(
        self,
        config: ProjectConfig,
        spark: SparkSession,
        endpoint_name: str | None = None,
        embedding_model: str | None = None,
        usage_policy_id: str | None = None,
    ) -> None:
        """Initialize VectorSearchManager.

        Args:
            config: ProjectConfig object
            spark: SparkSession instance
            endpoint_name: Name of the vector search endpoint (uses config if None)
            embedding_model: Name of the embedding model endpoint (uses config if None)
            usage_policy_id: ID of the usage policy for the endpoint (optional)
        """
        self.config = config
        self.spark = spark
        self.endpoint_name = endpoint_name or config.vector_search_endpoint
        self.embedding_model = embedding_model or config.embedding_endpoint
        self.catalog = config.catalog
        self.schema = config.schema
        self.usage_policy_id = usage_policy_id

        w = WorkspaceClient()
        self.client = VectorSearchClient(
            workspace_url=w.config.host,
            personal_access_token=w.tokens.create(lifetime_seconds=1200).token_value,
        )

    def create_endpoint_if_not_exists(self) -> None:
        """Create vector search endpoint if it doesn't exist."""
        endpoints_response = self.client.list_endpoints()
        endpoints = (
            endpoints_response.get("endpoints", [])
            if isinstance(endpoints_response, dict)
            else []
        )
        endpoint_exists = any(
            (ep.get("name") if isinstance(ep, dict) else getattr(ep, "name", None))
            == self.endpoint_name
            for ep in endpoints
        )

        if not endpoint_exists:
            logger.info(f"Creating vector search endpoint: {self.endpoint_name}")
            self.client.create_endpoint_and_wait(
                name=self.endpoint_name,
                endpoint_type="STANDARD",
                usage_policy_id=self.usage_policy_id,
            )
            logger.info(f"✓ Vector search endpoint created: {self.endpoint_name}")
        else:
            logger.info(f"✓ Vector search endpoint exists: {self.endpoint_name}")

    def create_unified_table(
        self,
        source_tables: dict[str, str],
        unified_table: str,
    ) -> None:
        """Union multiple chunk tables into a single Delta table for indexing.

        Each source table must have a `text` column. IDs are prefixed with the
        source name to avoid collisions (e.g. "arxiv_chunk_001").

        Args:
            source_tables: Dict mapping source name to fully qualified table name,
                e.g. {"arxiv": "catalog.schema.arxiv_chunks_table"}
            unified_table: Fully qualified name for the output unified table
        """
        dfs = []
        for source, table in source_tables.items():
            id_col = "id" if source == "arxiv" else "chunk_id"
            dfs.append(
                self.spark.table(table)
                .select(
                    F.concat_ws("_", F.lit(source), F.col(id_col)).alias("id"),
                    F.col("text"),
                )
                .withColumn("source", F.lit(source))
            )

        unified_df = dfs[0]
        for df in dfs[1:]:
            unified_df = unified_df.union(df)

        unified_df.write.format("delta").mode("overwrite").saveAsTable(unified_table)
        logger.info(f"Written unified chunks to {unified_table}")

        self.spark.sql(
            f"ALTER TABLE {unified_table} "
            f"SET TBLPROPERTIES (delta.enableChangeDataFeed = true)"
        )
        logger.info(f"CDF enabled for {unified_table}")

    def create_or_get_index(
        self,
        index_name: str,
        source_table: str,
        primary_key: str,
        embedding_source_column: str = "text",
    ) -> VectorSearchIndex:
        """Create a delta sync index if it doesn't exist, or return the existing one.

        Args:
            index_name: Fully qualified index name (catalog.schema.index)
            source_table: Fully qualified source Delta table name
            primary_key: Primary key column of the source table
            embedding_source_column: Column to embed (default: "text")

        Returns:
            VectorSearchIndex object
        """
        self.create_endpoint_if_not_exists()

        try:
            index = self.client.get_index(index_name=index_name)
            logger.info(f"✓ Vector search index exists: {index_name}")
            return index
        except Exception:
            logger.info(f"Index {index_name} not found, will create it")

        try:
            index = self.client.create_delta_sync_index(
                endpoint_name=self.endpoint_name,
                source_table_name=source_table,
                index_name=index_name,
                pipeline_type="TRIGGERED",
                primary_key=primary_key,
                embedding_source_column=embedding_source_column,
                embedding_model_endpoint_name=self.embedding_model,
                usage_policy_id=self.usage_policy_id,
            )
            logger.info(f"✓ Vector search index created: {index_name}")
            return index
        except Exception as e:
            if "RESOURCE_ALREADY_EXISTS" not in str(e):
                raise
            logger.info(f"✓ Vector search index exists: {index_name}")
            return self.client.get_index(index_name=index_name)

    def sync_index(self, index_name: str) -> None:
        """Trigger a sync on an existing vector search index.

        Retries up to 5 times with increasing backoff if the endpoint is not
        yet ready (common immediately after endpoint creation).

        Args:
            index_name: Fully qualified index name (catalog.schema.index)
        """
        index = self.client.get_index(index_name=index_name)
        logger.info(f"Syncing vector search index: {index_name}")
        for attempt in range(5):
            try:
                index.sync()
                logger.info("✓ Index sync triggered")
                return
            except Exception as e:
                if "not ready yet" not in str(e) or attempt == 4:
                    raise
                wait = 30 * (attempt + 1)
                logger.info(f"Endpoint not ready, retrying in {wait}s...")
                time.sleep(wait)

    def search(
        self,
        query: str,
        index_name: str,
        columns: list[str] | None = None,
        num_results: int = 5,
        filters: dict | None = None,
        query_type: str = "hybrid",
    ) -> dict:
        """Search a vector index using similarity search.

        Args:
            query: Search query text
            index_name: Fully qualified index name to search
            columns: Columns to return in results (default: ["id", "text"])
            num_results: Number of results to return
            filters: Optional metadata filters to apply
            query_type: Search type — "hybrid" (semantic + keyword) or
                "ann" (semantic only)

        Returns:
            Raw search results dictionary with manifest and result keys
        """
        index = self.client.get_index(index_name=index_name)
        return index.similarity_search(
            query_text=query,
            columns=columns or ["id", "text"],
            num_results=num_results,
            filters=filters,
            query_type=query_type,
        )

    def parse_results(self, results: dict) -> list[dict]:
        """Parse raw vector search results into a list of dictionaries.

        Args:
            results: Raw results dict from similarity_search()

        Returns:
            List of dicts with column names as keys
        """
        columns = [col["name"] for col in results.get("manifest", {}).get("columns", [])]
        data_array = results.get("result", {}).get("data_array", [])
        return [dict(zip(columns, row, strict=False)) for row in data_array]
