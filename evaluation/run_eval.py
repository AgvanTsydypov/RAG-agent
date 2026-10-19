import sys
import types

# --- Compatibility shim for a ragas upstream bug (issue #2745) ---------------
# ragas hard-imports ChatVertexAI from a path that was removed in modern
# langchain. We don't use Vertex AI, so we only need the import to resolve.
# This avoids downgrading langchain-core (which would break the agent).
_vertex_path = "langchain_community.chat_models.vertexai"
if _vertex_path not in sys.modules:
    try:
        from langchain_google_vertexai import ChatVertexAI  # use the real one if present
    except Exception:
        class ChatVertexAI:  # harmless stub; never instantiated in our eval
            pass
    _shim = types.ModuleType(_vertex_path)
    _shim.ChatVertexAI = ChatVertexAI
    sys.modules[_vertex_path] = _shim
# -----------------------------------------------------------------------------

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import EvaluationDataset, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)

from app.config import settings
from evaluation.collect import collect_agent, collect_baseline
from evaluation.dataset import EVAL_SET

_judge_llm = LangchainLLMWrapper(
    ChatOpenAI(
        model=settings.chat_model,
        api_key=settings.openai_api_key,
        reasoning_effort=settings.reasoning_effort,  # "none"
        # No use_responses_api here: RAGAS passes `n` (multiple completions),
        # which the Responses API rejects but Chat Completions accepts.
    )
)
_judge_emb = LangchainEmbeddingsWrapper(
    OpenAIEmbeddings(model=settings.embedding_model, api_key=settings.openai_api_key)
)

METRICS = [
    Faithfulness(llm=_judge_llm),
    ResponseRelevancy(llm=_judge_llm, embeddings=_judge_emb),
    LLMContextPrecisionWithReference(llm=_judge_llm),
    LLMContextRecall(llm=_judge_llm),
]


def _build_rows(collector) -> list[dict]:
    rows = []
    for item in EVAL_SET:
        if item.get("turns"):  # single-turn metrics; multi-turn lives in benchmark.py
            continue
        trace = collector(item["question"])
        rows.append(
            {
                "user_input": item["question"],
                "retrieved_contexts": trace.contexts or [""],  # RAGAS needs a non-empty list
                "response": trace.answer,
                "reference": item["reference"],
            }
        )
    return rows


def run(name: str, collector):
    print(f"\n=== Collecting answers: {name} ===")
    dataset = EvaluationDataset.from_list(_build_rows(collector))
    print(f"=== Scoring with the judge (many LLM calls): {name} ===")
    result = evaluate(dataset=dataset, metrics=METRICS)
    print(f"\n--- {name} ---\n{result}")
    return result


if __name__ == "__main__":
    run("BASELINE (Stage 2 linear pipeline)", collect_baseline)
    run("AGENT (Stage 4 LangGraph)", collect_agent)