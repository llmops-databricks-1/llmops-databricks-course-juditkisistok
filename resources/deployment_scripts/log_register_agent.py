# Databricks notebook source
import mlflow

from eurovision_voting_bloc_party.agent import log_register_agent
from eurovision_voting_bloc_party.config import load_config
from eurovision_voting_bloc_party.evaluation import evaluate_agent
from eurovision_voting_bloc_party.utils import get_widget

env = get_widget("env", "dev")
git_sha = get_widget("git_sha", "local")
run_id = get_widget("run_id", "local")

cfg = load_config("project_config.yaml", env)
mlflow.set_experiment(cfg.experiment_name)
model_name = f"{cfg.catalog}.{cfg.schema}.eurovision_agent"


# COMMAND ----------
# Run evaluation
eval_questions = [
    "Which countries always vote for each other?",
    "Who has won Eurovision the most times?",
    "What is nul points?",
    "Who is going to win Eurovision this year?",
    "I'm from the United Kingdom, big Eurovision fan!",
    "What is the capital of France?",
    "Write me a python function that iterates over a list of numbers and returns the sum",
]

results = evaluate_agent(cfg, eval_questions)

# COMMAND ----------
# Log and register model
registered_model = log_register_agent(
    cfg=cfg,
    git_sha=git_sha,
    run_id=run_id,
    agent_code_path="../../src/eurovision_voting_bloc_party/agent.py",
    model_name=model_name,
    evaluation_metrics=results.metrics,
)
