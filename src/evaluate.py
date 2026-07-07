
import pandas as pd

from src.graph import build_rag_graph

def chunk_ids(results):
    return [
        doc.metadata.get("chunk_id", "unknown")
        for doc, _ in results
    ]


def run_evaluation(ticker: str = "JPM") -> pd.DataFrame:
    queries_df=pd.read_csv("evaluation/queries.csv")
    graph = build_rag_graph()
    rows = []

    for query in queries_df["query"]:
        for strategy in ["fixed", "semantic"]:
            result = graph.invoke(
                {
                    "query": query,
                    "ticker": ticker,
                    "strategy": strategy,
                    "retrieved_docs": [],
                    "reranked_docs": [],
                    "answer": "",
                }
            )

            rows.append(
                {
                    "query": query,
                    "strategy": strategy,
                    "retrieved_top_10": chunk_ids(result["retrieved_docs"]),
                    "reranked_top_5": chunk_ids(result["reranked_docs"]),
                    "answer": result["answer"],
                    "manual_relevance_score_before": "",
                    "manual_relevance_score_after": "",
                    "notes": "",
                }
            )

    df = pd.DataFrame(rows)
    output_path = "evaluation/results.csv"
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    df = run_evaluation("JPM")
    print(df)