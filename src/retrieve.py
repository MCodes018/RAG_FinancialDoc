
import logging
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_chroma import Chroma

from src.config import VECTOR_DB_DIR, TOP_K
from src.build_index import get_embedding_model


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_vectorstore(ticker: str, strategy: str) -> Chroma:
    persist_dir = VECTOR_DB_DIR / f"{ticker}_{strategy}"

    return Chroma(
        persist_directory=str(persist_dir),
        embedding_function=get_embedding_model(),
        collection_name=f"{ticker}_{strategy}_collection",
    )


def retrieve_chunks(
    query: str,
    ticker: str,
    strategy: str,
    top_k: int = TOP_K,
) -> List[Tuple[Document, float]]:
    # Return top-k chunks with similarity scores.

    vectorstore = load_vectorstore(ticker, strategy)

    results = vectorstore.similarity_search_with_score(
        query=query,
        k=top_k,
    )

    return results


if __name__ == "__main__":
    query = "What are JPMorgan's major risk factors?"

    for strategy in ["fixed", "semantic"]:
        print(f"\n--- {strategy.upper()} RETRIEVAL ---")
        results = retrieve_chunks(query, "JPM", strategy)

        for doc, score in results:
            print("Score:", score)
            print("Chunk ID:", doc.metadata.get("chunk_id"))
            print(doc.page_content[:500])
            print("-" * 80)