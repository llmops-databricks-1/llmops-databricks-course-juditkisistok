<h1 align="center">
Eurovision Voting Bloc Party
</h1>

LLM-powered Q&A system for Eurovision Song Contest data using Databricks.

## Overview

Ingests Eurovision contest data from Kaggle and Wikipedia, stores it in Delta tables, and uses Databricks LLMs to answer questions about Eurovision history.

## Setup

**Requirements:**

- Python 3.12
- `uv` package manager
- Databricks workspace access

**Install dependencies:**

```bash
uv sync --extra dev

Configure Databricks:
Update databricks.yml with your workspace URL and run:
databricks bundle deploy

Project Structure

├── notebooks/          # Databricks notebooks
│   └── 01_preprocess.py
├── resources/          # Job definitions
├── src/                # Python package
│   └── eurovision_voting_bloc_party/
├── databricks.yml      # Bundle configuration
├── project_config.yaml # Environment config
└── pyproject.toml      # Dependencies

Usage

Run notebook on Databricks:
databricks bundle run data_preprocessing_job

Query with LLM:
The notebooks demonstrate how to query Delta tables and use LLMs to answer questions about Eurovision data.

Development

Uses uv run for all commands:
uv run pre-commit run --all-files
uv run pytest
```
