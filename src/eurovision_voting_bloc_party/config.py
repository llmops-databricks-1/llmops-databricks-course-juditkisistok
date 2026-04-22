from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession


class ProjectConfig(BaseModel):
    """Project configuration model."""

    catalog: str = Field(..., description="Unity Catalog name")
    db_schema: str = Field(..., description="Schema name", alias="schema")
    volume: str = Field(..., description="Volume name")
    llm_endpoint: str = Field(..., description="LLM endpoint name")
    system_prompt: str = Field(
        default="You are a helpful AI assistant that helps users learn about Eurovision.",
        description="System prompt for the agent",
    )
    vector_search_endpoint: str = Field(..., description="Vector search endpoint name")
    embedding_endpoint: str = Field(..., description="Embedding model endpoint name")
    experiment_name: str | None = None

    model_config = {"populate_by_name": True}

    @classmethod
    def from_yaml(cls, config_path: str, env: str = "dev") -> "ProjectConfig":
        """Load configuration from YAML file.

        Args:
            config_path: Path to the YAML configuration file
            env: Environment name (dev, acc, prd)

        Returns:
            ProjectConfig instance
        """
        if env not in ["prd", "acc", "dev"]:
            raise ValueError(
                f"Invalid environment: {env}. \
                             Expected 'prd', 'acc', or 'dev'"
            )

        with open(config_path) as f:
            config_data = yaml.safe_load(f)

        if env not in config_data:
            raise ValueError(f"Environment '{env}' not found in config file")

        return cls(**config_data[env])

    @property
    def schema(self) -> str:
        """Alias for db_schema for backward compatibility."""
        return self.db_schema

    @property
    def full_schema_name(self) -> str:
        """Get fully qualified schema name."""
        return f"{self.catalog}.{self.db_schema}"

    @property
    def full_volume_path(self) -> str:
        """Get fully qualified volume path."""
        return f"{self.catalog}.{self.schema}.{self.volume}"


def load_config(
    config_path: str = "project_config.yaml", env: str = "dev"
) -> ProjectConfig:
    """Load project configuration.

    Args:
        config_path: Path to configuration file
        env: Environment name

    Returns:
        ProjectConfig instance
    """
    # Handle relative paths from notebooks
    if not Path(config_path).is_absolute():
        # Try to find config in parent directories
        current = Path.cwd()
        for _ in range(3):  # Search up to 3 levels
            candidate = current / config_path
            if candidate.exists():
                config_path = str(candidate)
                break
            current = current.parent

    return ProjectConfig.from_yaml(config_path, env)


def get_env(spark: SparkSession) -> str:
    """Get current environment from dbutils widget, falling back to ENV variable or 'dev'.

    Returns:
        Environment name (dev, acc, or prd)
    """
    try:
        dbutils = DBUtils(spark)
        return dbutils.widgets.get("env")
    except Exception:
        return "dev"
