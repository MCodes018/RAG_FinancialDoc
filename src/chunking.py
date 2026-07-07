import json
import logging
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    CHUNK_DIR,
    EMBEDDING_MODEL,
    FIXED_CHUNK_SIZE,
    FIXED_CHUNK_OVERLAP,
)
from src.ingest import load_processed_document


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_fixed_chunks(document: Document) -> List[Document]:
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=FIXED_CHUNK_SIZE,
        chunk_overlap=FIXED_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents([document])

    for idx, chunk in enumerate(chunks):
        chunk.metadata.update(
            {
                "chunk_id": f"{document.metadata['ticker']}_fixed_{idx}",
                "chunk_strategy": "fixed",
                "chunk_index": idx,
            }
        )

    logger.info("Created %s fixed-size chunks", len(chunks))

    return chunks


def create_semantic_chunks(document: Document) -> List[Document]:
    max_chars = 4500
    paragraphs = [
        p.strip()
        for p in document.page_content.split("\n")
        if len(p.strip()) > 50
    ]

    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) <= max_chars:
            current_chunk += "\n" + paragraph
        else:
            chunks.append(current_chunk.strip())
            current_chunk = paragraph

    if current_chunk:
        chunks.append(current_chunk.strip())

    documents = []

    for idx, chunk_text in enumerate(chunks):
        documents.append(
            Document(
                page_content=chunk_text,
                metadata={
                    **document.metadata,
                    "chunk_id": f"{document.metadata['ticker']}_semantic_{idx}",
                    "chunk_strategy": "semantic",
                    "chunk_index": idx,
                },
            )
        )

    logger.info("Created %s semantic chunks", len(documents))

    return documents


def save_chunks(chunks: List[Document], ticker: str, strategy: str) -> Path:

    CHUNK_DIR.mkdir(parents=True, exist_ok=True)

    output_path = CHUNK_DIR / f"{ticker}_{strategy}_chunks.jsonl"

    with output_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            record = {
                "page_content": chunk.page_content,
                "metadata": chunk.metadata,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("Saved %s chunks to %s", strategy, output_path)

    return output_path


def load_chunks(ticker: str, strategy: str) -> List[Document]:

    input_path = CHUNK_DIR / f"{ticker}_{strategy}_chunks.jsonl"

    if not input_path.exists():
        raise FileNotFoundError(f"Chunk file not found: {input_path}")

    chunks = []

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            chunks.append(
                Document(
                    page_content=record["page_content"],
                    metadata=record["metadata"],
                )
            )

    return chunks


def chunk_company(ticker: str) -> None:

#Saving both chunks (tie-up process)
    document = load_processed_document(ticker)

    fixed_chunks = create_fixed_chunks(document)
    save_chunks(fixed_chunks, ticker=ticker, strategy="fixed")

    semantic_chunks = create_semantic_chunks(document)
    save_chunks(semantic_chunks, ticker=ticker, strategy="semantic")


if __name__ == "__main__":
    chunk_company("JPM")