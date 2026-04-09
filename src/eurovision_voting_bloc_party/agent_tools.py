import json
from datetime import datetime

import pyspark.sql.functions as F
from pyspark.sql import SparkSession

from eurovision_voting_bloc_party.mcp import ToolInfo

# tool specs

predict_winner_spec = {
    "type": "function",
    "function": {
        "name": "predict_winner",
        "description": "Fetch historical Eurovision country stats to predict this year's "
        "winner. Use this when someone asks who will win, who should win, or "
        "wants a prediction.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

roast_country_spec = {
    "type": "function",
    "function": {
        "name": "roast_country",
        "description": "Fetch a country's Eurovision history to deliver a loving but "
        "devastating roast. Use this when someone mentions they're from a country, "
        "asks about a specific country, or explicitly asks for a roast.",
        "parameters": {
            "type": "object",
            "properties": {
                "country_name": {
                    "type": "string",
                    "description": "The country to roast, e.g. 'United Kingdom', "
                    "'Norway', 'France'",
                }
            },
            "required": ["country_name"],
        },
    },
}


# tool functions


def create_predict_winner_tool(
    spark: SparkSession, catalog: str, schema: str
) -> ToolInfo:
    """
    Create a ToolInfo for predicting this year's Eurovision winner.

    Args:
        spark: SparkSession instance
        catalog: Unity Catalog name
        schema: Schema name
    Returns:
        ToolInfo wrapping the predict_winner function
    """

    def predict_winner() -> str:
        """
        Pull historical Eurovision stats to inform winner prediction.

        Returns:
            JSON string with country stats relevant for prediction
        """

        kaggle_data = spark.table(f"{catalog}.{schema}.eurovision_kaggle_chunks")
        rows = kaggle_data.select("chunk_id", "text").collect()

        stats = [{"country": row.chunk_id, "stat": row.text} for row in rows]

        return json.dumps(
            {
                "year": datetime.now().year,
                "historical_data": stats,
            }
        )

    return ToolInfo(
        name="predict_winner", spec=predict_winner_spec, exec_fn=predict_winner
    )


def create_roast_country_tool(spark: SparkSession, catalog: str, schema: str) -> ToolInfo:
    """
      Create a ToolInfo for roasting a country based on its Eurovision history.

      Args:
        spark: SparkSession instance
        catalog: Unity Catalog name
        schema: Schema name

    Returns:
        ToolInfo wrapping the roast_country function

    """

    def roast_country(country_name: str) -> str:
        """
        Fetch Eurovision stats for a country to fuel a loving but devastating roast.
        Args:
            country_name: Name of the country to roast
        Returns:
            JSON string with the country's Eurovision history
        """

        kaggle_data = spark.table(f"{catalog}.{schema}.eurovision_kaggle_chunks")
        country_id = country_name.strip().lower().replace(" ", "_")

        rows = (
            kaggle_data.filter(F.col("chunk_id") == country_id).select("text").collect()
        )

        if not rows:
            return json.dumps(
                {
                    "error": f"""No data found for country: {country_name}.
                                    Maybe they never qualified?"""
                }
            )

        return json.dumps(
            {
                "country": country_name,
                "stats": rows[0].text,
            }
        )

    return ToolInfo(name="roast_country", spec=roast_country_spec, exec_fn=roast_country)
