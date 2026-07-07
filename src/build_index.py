
import logging
from typing import List

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from src.config import VECTOR_DB_DIR, EMBEDDING_MODEL
from src.chunking import load_chunks


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


EMBEDDINGS = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

def get_embedding_model():
    return EMBEDDINGS

def build_chroma_index(ticker: str, strategy: str) -> Chroma:

    chunks: List[Document] = load_chunks(ticker=ticker, strategy=strategy)

    persist_dir = VECTOR_DB_DIR / f"{ticker}_{strategy}"

    embeddings = get_embedding_model()

    logger.info("Building Chroma index for %s / %s", ticker, strategy)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_dir),
        collection_name=f"{ticker}_{strategy}_collection",
    )

    logger.info("Index saved to %s", persist_dir)

    return vectorstore


def build_all_indexes(ticker: str) -> None:
    build_chroma_index(ticker, "fixed")
    build_chroma_index(ticker, "semantic")


if __name__ == "__main__":
    build_all_indexes("JPM")