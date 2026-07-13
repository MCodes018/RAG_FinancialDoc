
#Validates the evaluation dataset before expensive RAG/LLM evaluation runs.

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.build_index import get_embedding_model


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "evaluation" / "eval_dataset.csv"
REPORT_PATH = PROJECT_ROOT / "evaluation" / "test_quality_report.csv"

REQUIRED_COLUMNS = {
    "question",
    "reference_answer",
    "category",
    "expected_answerable",
}


def cosine_matrix(vectors: list[list[float]]) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=float)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normalized = matrix / norms
    return normalized @ normalized.T


def audit_dataset() -> pd.DataFrame:
    dataset = pd.read_csv(DATASET_PATH)
    findings: list[dict[str, object]] = []

    missing_columns = REQUIRED_COLUMNS - set(dataset.columns)
    findings.append(
        {
            "check": "required_columns",
            "status": "pass" if not missing_columns else "fail",
            "details": (
                "All required columns are present."
                if not missing_columns
                else f"Missing: {sorted(missing_columns)}"
            ),
        }
    )

    missing_cells = int(
        dataset[list(REQUIRED_COLUMNS & set(dataset.columns))]
        .isna()
        .sum()
        .sum()
    )
    findings.append(
        {
            "check": "missing_values",
            "status": "pass" if missing_cells == 0 else "fail",
            "details": f"{missing_cells} missing required values.",
        }
    )

    duplicate_count = int(dataset["question"].duplicated().sum())
    findings.append(
        {
            "check": "exact_duplicates",
            "status": "pass" if duplicate_count == 0 else "fail",
            "details": f"{duplicate_count} exact duplicate questions.",
        }
    )

    categories = sorted(dataset["category"].dropna().unique().tolist())
    findings.append(
        {
            "check": "category_coverage",
            "status": "pass" if len(categories) >= 5 else "warn",
            "details": f"{len(categories)} categories: {categories}",
        }
    )

    answerability_values = dataset["expected_answerable"].astype(str).str.lower()
    has_answerable = answerability_values.isin(["true", "1"]).any()
    has_unanswerable = answerability_values.isin(["false", "0"]).any()
    findings.append(
        {
            "check": "answerability_coverage",
            "status": "pass" if has_answerable and has_unanswerable else "warn",
            "details": (
                "Includes both answerable and unanswerable questions."
                if has_answerable and has_unanswerable
                else "Add at least one genuinely unanswerable control question."
            ),
        }
    )

    short_references = int(
        dataset["reference_answer"].fillna("").str.split().str.len().lt(8).sum()
    )
    findings.append(
        {
            "check": "reference_answer_detail",
            "status": "pass" if short_references == 0 else "warn",
            "details": f"{short_references} reference answers contain fewer than 8 words.",
        }
    )

    embeddings = get_embedding_model()
    question_vectors = embeddings.embed_documents(dataset["question"].tolist())
    similarities = cosine_matrix(question_vectors)

    near_duplicate_pairs = []
    for left in range(len(dataset)):
        for right in range(left + 1, len(dataset)):
            if similarities[left, right] >= 0.92:
                near_duplicate_pairs.append(
                    (
                        left,
                        right,
                        round(float(similarities[left, right]), 3),
                    )
                )

    findings.append(
        {
            "check": "near_duplicates",
            "status": "pass" if not near_duplicate_pairs else "warn",
            "details": (
                "No near-duplicate questions above 0.92 cosine similarity."
                if not near_duplicate_pairs
                else f"Potential near duplicates: {near_duplicate_pairs}"
            ),
        }
    )

    report = pd.DataFrame(findings)
    report.to_csv(REPORT_PATH, index=False)
    print(report.to_string(index=False))
    print(f"\nSaved: {REPORT_PATH}")
    return report


if __name__ == "__main__":
    audit_dataset()
