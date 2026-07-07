
from typing import List, Tuple, TypedDict

from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END

from src.retrieve import retrieve_chunks
from src.rerank import rerank_chunks
from src.generate import generate_answer


class RAGState(TypedDict):
    query: str
    ticker: str
    strategy: str
    retrieved_docs: List[Tuple[Document, float]]
    reranked_docs: List[Tuple[Document, float]]
    answer: str


def retrieve_node(state: RAGState) -> RAGState:
    retrieved = retrieve_chunks(
        query=state["query"],
        ticker=state["ticker"],
        strategy=state["strategy"],
        top_k=10,
    )

    return {
        **state,
        "retrieved_docs": retrieved,
    }


def rerank_node(state: RAGState) -> RAGState:
    reranked = rerank_chunks(
        query=state["query"],
        retrieved_docs=state["retrieved_docs"],
        top_n=5,
    )

    return {
        **state,
        "reranked_docs": reranked,
    }


def generate_node(state: RAGState) -> RAGState:
    answer = generate_answer(
        query=state["query"],
        reranked_docs=state["reranked_docs"],
    )

    return {
        **state,
        "answer": answer,
    }


def build_rag_graph():
    graph = StateGraph(RAGState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("generate", generate_node)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


if __name__ == "__main__":
    rag_graph = build_rag_graph()

    query = input("Enter your question: ").strip()

    for strategy in ["fixed", "semantic"]:
        print(f"\n\n{strategy.upper()} CHUNKING\n")

        result = rag_graph.invoke(
            {
                "query": query,
                "ticker": "JPM",
                "strategy": strategy,
                "retrieved_docs": [],
                "reranked_docs": [],
                "answer": "",
            }
        )

        print(result["answer"])