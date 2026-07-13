from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from src.build_index import get_embedding_model
from src.config import OLLAMA_MODEL
from src.graph import build_rag_graph


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "evaluation" / "eval_dataset.csv"
DETAILED_PATH = PROJECT_ROOT / "evaluation" / "detailed_metrics.csv"
SUMMARY_PATH = PROJECT_ROOT / "evaluation" / "summary_metrics.csv"

JUDGE = ChatOllama(model=OLLAMA_MODEL, temperature=0)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)

    denominator = np.linalg.norm(left_array) * np.linalg.norm(right_array)
    if denominator == 0:
        return 0.0

    return float(np.dot(left_array, right_array) / denominator)


def answer_similarity(answer: str, reference_answer: str) -> float:   #Semantic similarity using the same local BAAI embedding model.
    embeddings = get_embedding_model()
    answer_vector = embeddings.embed_query(answer)
    reference_vector = embeddings.embed_query(reference_answer)
    return cosine_similarity(answer_vector, reference_vector)


def format_context(documents: list[tuple[Any, float]]) -> str:
    blocks = []

    for rank, (document, score) in enumerate(documents, start=1):
        chunk_id = document.metadata.get("chunk_id", "unknown")
        blocks.append(
            f"[Rank {rank} | chunk_id={chunk_id} | score={float(score):.4f}]\n"
            f"{document.page_content}"
        )

    return "\n\n".join(blocks)


def parse_json_object(text: str) -> dict[str, Any]:    # To extract a JSON object even when the local model adds extra text.
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in judge response: {text}")

    return json.loads(match.group(0))


def clamp_score(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def judge_answer(   #LLM-as-a-judge approach with local Ollama model
    question: str,
    reference_answer: str,
    answer: str,
    context: str,
) -> dict[str, Any]:
    prompt = ChatPromptTemplate.from_template(
        """
You are evaluating a financial RAG answer.

Score each metric from 0 to 1:
- faithfulness: Are all claims supported by the retrieved context?
- answer_relevance: Does the answer directly address the question?
- answer_correctness: Does the answer agree with the reference answer?
- context_relevance: Is the retrieved context useful for answering the question?

Return ONLY valid JSON using exactly this structure:
{{
  "faithfulness": 0.0,
  "answer_relevance": 0.0,
  "answer_correctness": 0.0,
  "context_relevance": 0.0,
  "reason": "one short explanation"
}}

Question:
{question}

Reference answer:
{reference_answer}

Generated answer:
{answer}

Retrieved context:
{context}
"""
    )

    response = (prompt | JUDGE).invoke(
        {
            "question": question,
            "reference_answer": reference_answer,
            "answer": answer,
            "context": context[:14000],
        }
    )

    try:
        parsed = parse_json_object(response.content)
    except (ValueError, json.JSONDecodeError):
        return {
            "faithfulness": 0.0,
            "answer_relevance": 0.0,
            "answer_correctness": 0.0,
            "context_relevance": 0.0,
            "judge_reason": "Judge returned invalid JSON.",
        }

    return {
        "faithfulness": clamp_score(parsed.get("faithfulness")),
        "answer_relevance": clamp_score(parsed.get("answer_relevance")),
        "answer_correctness": clamp_score(parsed.get("answer_correctness")),
        "context_relevance": clamp_score(parsed.get("context_relevance")),
        "judge_reason": str(parsed.get("reason", "")),
    }


def chunk_ids(results: list[tuple[Any, float]]) -> list[str]:
    return [
        document.metadata.get("chunk_id", "unknown")
        for document, _ in results
    ]


def reranking_metrics(
    retrieved: list[tuple[Any, float]],
    reranked: list[tuple[Any, float]],
) -> dict[str, float | int | bool]:
    retrieved_ids = chunk_ids(retrieved)
    reranked_ids = chunk_ids(reranked)

    top_five_before = retrieved_ids[:5]
    top_five_after = reranked_ids[:5]

    overlap = len(set(top_five_before) & set(top_five_after)) / max(
        1, len(top_five_after)
    )

    gains = []
    for new_rank, chunk_id in enumerate(top_five_after, start=1):
        if chunk_id in retrieved_ids:
            old_rank = retrieved_ids.index(chunk_id) + 1
            gains.append(old_rank - new_rank)

    return {
        "top1_changed": bool(
            retrieved_ids and reranked_ids and retrieved_ids[0] != reranked_ids[0]
        ),
        "top5_overlap": float(overlap),
        "mean_rank_gain": float(np.mean(gains)) if gains else 0.0,
    }


def answerability_check(answer: str, expected_answerable: bool) -> float:
    refusal_markers = (
        "does not contain enough information",
        "insufficient information",
        "cannot determine",
        "not provided in the context",
    )
    refused = any(marker in answer.lower() for marker in refusal_markers)

    if expected_answerable:
        return 0.0 if refused else 1.0

    return 1.0 if refused else 0.0


def run_evaluation(ticker: str = "JPM") -> pd.DataFrame:
    dataset = pd.read_csv(DATASET_PATH)
    graph = build_rag_graph()
    rows: list[dict[str, Any]] = []

    for record in dataset.to_dict(orient="records"):
        question = str(record["question"])
        reference_answer = str(record["reference_answer"])
        category = str(record["category"])
        expected_answerable = bool(record["expected_answerable"])

        for strategy in ("fixed", "semantic"):
            started = time.perf_counter()

            result = graph.invoke(
                {
                    "query": question,
                    "ticker": ticker,
                    "strategy": strategy,
                    "retrieved_docs": [],
                    "reranked_docs": [],
                    "answer": "",
                }
            )

            latency = time.perf_counter() - started
            retrieved = result["retrieved_docs"]
            reranked = result["reranked_docs"]
            answer = result["answer"]

            context = format_context(reranked)
            judge_scores = judge_answer(
                question=question,
                reference_answer=reference_answer,
                answer=answer,
                context=context,
            )
            rank_scores = reranking_metrics(retrieved, reranked)

            rows.append(
                {
                    "question": question,
                    "category": category,
                    "strategy": strategy,
                    "reference_answer": reference_answer,
                    "generated_answer": answer,
                    "retrieved_chunk_ids": chunk_ids(retrieved),
                    "reranked_chunk_ids": chunk_ids(reranked),
                    "answer_similarity": answer_similarity(
                        answer, reference_answer
                    ),
                    "faithfulness": judge_scores["faithfulness"],
                    "answer_relevance": judge_scores["answer_relevance"],
                    "answer_correctness": judge_scores["answer_correctness"],
                    "context_relevance": judge_scores["context_relevance"],
                    "answerability_score": answerability_check(
                        answer, expected_answerable
                    ),
                    "top1_changed": rank_scores["top1_changed"],
                    "top5_overlap": rank_scores["top5_overlap"],
                    "mean_rank_gain": rank_scores["mean_rank_gain"],
                    "latency_seconds": latency,
                    "judge_reason": judge_scores["judge_reason"],
                }
            )

    detailed = pd.DataFrame(rows)
    detailed.to_csv(DETAILED_PATH, index=False)

    metric_columns = [
        "answer_similarity",
        "faithfulness",
        "answer_relevance",
        "answer_correctness",
        "context_relevance",
        "answerability_score",
        "top5_overlap",
        "mean_rank_gain",
        "latency_seconds",
    ]

    summary = (
        detailed.groupby("strategy", as_index=False)[metric_columns]
        .mean(numeric_only=True)
    )

    top1_change_rate = (
        detailed.groupby("strategy")["top1_changed"]
        .mean()
        .rename("top1_change_rate")
        .reset_index()
    )
    summary = summary.merge(top1_change_rate, on="strategy", how="left")
    summary.to_csv(SUMMARY_PATH, index=False)

    print("\nEvaluation summary:\n")
    print(summary.to_string(index=False))
    print(f"\nDetailed results: {DETAILED_PATH}")
    print(f"Summary results:  {SUMMARY_PATH}")

    return detailed


if __name__ == "__main__":
    run_evaluation("JPM")
