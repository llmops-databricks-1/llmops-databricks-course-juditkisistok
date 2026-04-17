import mlflow
from mlflow.genai.scorers import Guidelines
from mlflow.types.responses import ResponsesAgentRequest

from eurovision_voting_bloc_party.agent import EurovisionAgent

# guidelines
witty_tone_guideline = Guidelines(
    name="witty_tone",
    guidelines=[
        "The response must be entertaining and witty, in the style of a dry TV host",
        "The response should not be bland, generic, or overly formal",
        "The response must show personality — sarcasm, humor, or dramatism are all valid",
    ],
    model="databricks:/databricks-gpt-oss-120b",
)

stays_in_scope_guideline = Guidelines(
    name="stays_in_scope",
    guidelines=[
        "The response must only discuss topics related to Eurovision",
        "If asked about something unrelated, the response must politely redirect",
        "The response must not answer general questions unrelated to Eurovision",
    ],
    model="databricks:/databricks-gpt-oss-120b",
)

# custom scorers


@mlflow.genai.scorer
def word_count_check(outputs: list) -> bool:
    """Response should be under 400 words."""
    text = outputs[0].get("text", "") if isinstance(outputs[0], dict) else str(outputs[0])
    return len(text.split()) < 400


@mlflow.genai.scorer
def mentions_countries(outputs: list) -> bool:
    """Response should mention at least one real country."""
    text = outputs[0].get("text", "") if isinstance(outputs[0], dict) else str(outputs[0])
    countries = [
        "norway",
        "sweden",
        "france",
        "uk",
        "united kingdom",
        "germany",
        "italy",
        "spain",
        "ireland",
        "finland",
        "ukraine",
        "portugal",
        "australia",
        "denmark",
        "greece",
        "switzerland",
    ]
    return any(c in text.lower() for c in countries)


def evaluate_agent(
    agent: EurovisionAgent, eval_questions: list[str]
) -> mlflow.models.EvaluationResult:
    eval_data = [{"inputs": {"question": q}} for q in eval_questions]

    def predict_fn(question: str) -> str:
        request = ResponsesAgentRequest(input=[{"role": "user", "content": question}])
        response = agent.predict(request)
        return response.output[0].model_dump()["content"][0]["text"]

    return mlflow.genai.evaluate(
        predict_fn=predict_fn,
        data=eval_data,
        scorers=[
            word_count_check,
            mentions_countries,
            witty_tone_guideline,
            stays_in_scope_guideline,
        ],
    )
