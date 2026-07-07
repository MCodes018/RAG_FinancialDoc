# Financial RAG Pipeline using LangChain, LangGraph and Ollama

## Project Overview

The goal of this project was to build an end-to-end Retrieval Augmented Generation (RAG) pipeline capable of answering questions from financial documents. For this implementation, I used a SEC 10-K filing of JPMorgan Chase downloaded from the SEC EDGAR database.

Rather than simply building a chatbot, the objective was to understand how different retrieval decisions affect the final generated answer. In particular, I wanted to compare:

- Fixed-size chunking vs Semantic chunking
- Retrieval before and after reranking
- The effect of better retrieval on the final LLM response

The project was implemented using **LangChain** for the RAG components, **LangGraph** for orchestrating the workflow, **ChromaDB** as the vector database, and **Ollama** to run the language model locally.

---

# Motivation

Large Language Models are powerful, but they cannot reliably answer questions about documents they have never seen. RAG solves this problem by retrieving relevant document chunks first and then providing them as context to the LLM.

However, retrieval quality depends heavily on decisions made before generation, such as:

- how the document is chunked
- which embedding model is used
- whether reranking is performed

Instead of assuming one approach is better, this project compares multiple approaches and evaluates the generated responses.

---

# Objectives

The objectives of this project were:

- Build an end-to-end financial RAG pipeline.
- Download and preprocess SEC filings.
- Compare fixed-size and semantic chunking strategies.
- Evaluate the effect of reranking.
- Generate grounded responses using a local LLM.
- Compare the quality of answers produced by different retrieval pipelines.

---

# System Architecture

```
SEC Filing
      │
      ▼
Download from EDGAR
      │
      ▼
Clean & Preprocess
      │
      ▼
LangChain Document
      │
 ┌────┴─────┐
 ▼          ▼
Fixed    Semantic
Chunking Chunking
 │          │
 ▼          ▼
Embeddings (BAAI/bge-base-en-v1.5)
 │
 ▼
Chroma Vector Database
 │
 ▼
Retriever
 │
 ▼
Cross-Encoder Reranker
 │
 ▼
Ollama (Llama 3.2)
 │
 ▼
Generated Answer
 │
 ▼
Evaluation
```

---

# Technologies Used

| Component | Technology |
|-----------|------------|
| Framework | LangChain |
| Workflow | LangGraph |
| Vector Database | ChromaDB |
| Embedding Model | BAAI/bge-base-en-v1.5 |
| Reranker | cross-encoder/ms-marco-MiniLM-L6-v2 |
| LLM | Ollama (Llama 3.2) |
| Data Source | SEC EDGAR |
| Language | Python |

---

# Design Decisions

## Why SEC EDGAR?

SEC filings are publicly available, structured financial documents that contain information such as business operations, financial risks, governance, and market outlook. They are commonly used in financial analysis, making them a suitable dataset for evaluating RAG systems.

---

## Why LangChain?

LangChain provides modular components for building RAG systems, including document processing, embeddings, retrievers, vector stores, and prompt pipelines. Using LangChain allowed me to focus on experimenting with retrieval strategies rather than implementing every component from scratch.

---

## Why LangGraph?

Instead of writing one long retrieval function, I wanted the pipeline to be organised into clear stages.

The workflow consists of:

```
Retrieve
    ↓
Rerank
    ↓
Generate
```

This makes it easier to understand how data flows through the system and allows additional steps to be added later if required.

---

## Why ChromaDB?

I chose ChromaDB because it is lightweight, easy to integrate with LangChain, stores vectors locally, and is sufficient for experimentation without requiring any external services.

---

## Why BAAI/bge-base-en-v1.5?

The embedding model converts document chunks and user queries into vector representations.

I chose BAAI/bge-base-en-v1.5 because it performs well on semantic retrieval benchmarks while still being free and runnable on a local machine.

---

## Why Two Chunking Strategies?

One of the objectives of this project was to compare retrieval quality.

### Fixed-size Chunking

Uses RecursiveCharacterTextSplitter to split documents into equally sized chunks with overlap.

Advantages:

- Simple
- Fast
- Easy to reproduce

Disadvantages:

- May split related concepts across chunk boundaries.

### Semantic Chunking

Instead of splitting purely by character count, this approach groups related paragraphs together to preserve context.

Advantages:

- Better context preservation
- More meaningful retrieval

Disadvantages:

- Slightly slower preprocessing
- Larger chunk sizes

---

## Why Reranking?

Vector search retrieves the most similar chunks, but similarity does not always mean relevance.

To improve retrieval quality, I used a cross-encoder reranker that scores each retrieved chunk based on both the query and the document text.

Only the highest ranked chunks are then passed to the language model.

---

## Why Ollama?

Instead of relying on paid APIs, I wanted the project to run completely offline.

Ollama allows local execution of language models while integrating smoothly with LangChain, making it suitable for experimentation without API costs.

---

# Evaluation

To compare the retrieval strategies fairly, I created a common set of financial questions stored in `evaluation/test_queries.csv`.

Each query is processed using:

- Fixed chunking pipeline
- Semantic chunking pipeline

For every run, the project records:

- Retrieved chunks
- Reranked chunks
- Generated answer

The outputs are stored in `evaluation/results.csv` for comparison.

---

# Observations

After evaluating both approaches, I observed that:

- Fixed-size chunking generally retrieved relevant information quickly but occasionally split related financial concepts across chunks.
- Semantic chunking usually produced more complete and better organised answers because related paragraphs remained together.
- Cross-encoder reranking improved retrieval by promoting more relevant chunks before answer generation.
- The overall quality of the generated responses depended more on retrieval quality than on the language model itself.

These observations reinforced the importance of retrieval design when building RAG systems.

---

# Installation

## 1. Clone the repository

```bash
git clone <repository-url>
cd financial-rag-project
```

## 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it.

**Windows (PowerShell)**

```powershell
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
source .venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Configure SEC EDGAR

Create a `.env` file in the project root.

```env
SEC_USER_NAME=Your Name
SEC_USER_EMAIL=your_email@example.com
```

## 5. Install Ollama

Download Ollama from:

https://ollama.com/download

Pull the model:

```bash
ollama pull llama3.2:3b
```

# Running the Project

The project is organised into modular stages to make experimentation with different RAG components easier.

The typical execution flow is:

```
Document Ingestion
        ↓
Document Chunking
        ↓
Vector Index Creation
        ↓
Retrieval
        ↓
Reranking
        ↓
Answer Generation
        ↓
Evaluation
```

The individual modules can be executed independently depending on the stage of the pipeline being tested. During development, this made it easier to debug and compare different retrieval strategies without rerunning the entire workflow.

The evaluation pipeline uses the predefined financial questions stored in:

```
evaluation/test_queries.csv
```

The generated outputs are written to:

```
evaluation/results.csv
```

which was used to compare the performance of fixed-size and semantic chunking strategies.

# Notes

- The first execution may take longer because the embedding model, reranker, and Ollama model are downloaded locally.
- All components used in this project are free and run locally.
- The current implementation uses the JPMorgan Chase SEC 10-K filing as the evaluation document.

## Future Improvements

During this project, several extensions were identified that could further improve the pipeline:

- Compare against section-aware chunking in addition to fixed and paragraph-aware chunking.
- Support multiple SEC filings and multiple companies.
- Introduce hybrid retrieval combining dense and sparse search.
- Add automatic evaluation metrics such as Context Precision and Answer Faithfulness using RAGAS.
- Deploy the pipeline behind a simple web interface for interactive querying.