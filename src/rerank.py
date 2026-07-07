
from typing import List, Tuple

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder


RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER = CrossEncoder(RERANKER_MODEL)

def get_reranker():
    return RERANKER


def rerank_chunks(
    query: str,
    retrieved_docs: List[Tuple[Document, float]],
    top_n: int = 5,
) -> List[Tuple[Document, float]]:

    reranker = get_reranker()

    pairs = [(query, doc.page_content) for doc, _ in retrieved_docs]
    scores = reranker.predict(pairs)

    reranked = [
        (doc, float(score))
        for (doc, _), score in zip(retrieved_docs, scores)
    ]

    reranked.sort(key=lambda x: x[1], reverse=True)

    return reranked[:top_n]