"""Hybrid search module combining vector search and BM25 keyword search using Reciprocal Rank Fusion (RRF)."""

import sys
from pathlib import Path
from typing import Any, Dict, List

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from src.search_vector import search as vector_search
    from src.search_keyword import search as keyword_search
except ImportError:
    from search_vector import search as vector_search
    from search_keyword import search as keyword_search


def hybrid_search(query: str, top_k: int = 5, pool: int = 10) -> List[Dict[str, Any]]:
    """Perform hybrid retrieval using vector search and BM25 keyword search merged via RRF.

    Score is calculated using Reciprocal Rank Fusion:
    score = sum(1 / (60 + rank)) for each rank list (rank starting at 1).
    Deduplicates results by chunk id and tracks the retrievers that found each document.
    Includes original vector score as 'vector_score' (None if found only by keyword search).
    """
    vector_results = vector_search(query, top_k=pool)
    keyword_results = keyword_search(query, top_k=pool)

    fused: Dict[str, Dict[str, Any]] = {}

    # Score vector search results
    for rank, item in enumerate(vector_results, start=1):
        chunk_id = item["id"]
        rrf_score = 1.0 / (60.0 + rank)
        if chunk_id not in fused:
            fused[chunk_id] = {
                "id": chunk_id,
                "source": item.get("source", ""),
                "title": item.get("title", ""),
                "text": item.get("text", ""),
                "score": rrf_score,
                "vector_score": item.get("score"),
                "found_by": ["vector"],
            }
        else:
            fused[chunk_id]["score"] += rrf_score
            fused[chunk_id]["vector_score"] = item.get("score")
            if "vector" not in fused[chunk_id]["found_by"]:
                fused[chunk_id]["found_by"].append("vector")

    # Score keyword search results
    for rank, item in enumerate(keyword_results, start=1):
        chunk_id = item["id"]
        rrf_score = 1.0 / (60.0 + rank)
        if chunk_id not in fused:
            fused[chunk_id] = {
                "id": chunk_id,
                "source": item.get("source", ""),
                "title": item.get("title", ""),
                "text": item.get("text", ""),
                "score": rrf_score,
                "vector_score": None,
                "found_by": ["keyword"],
            }
        else:
            fused[chunk_id]["score"] += rrf_score
            if "keyword" not in fused[chunk_id]["found_by"]:
                fused[chunk_id]["found_by"].append("keyword")

    results = list(fused.values())
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def main():
    if len(sys.argv) < 2:
        print("Usage: python src/hybrid.py \"<query>\"")
        sys.exit(1)

    query = sys.argv[1].strip()
    results = hybrid_search(query, top_k=3)

    if not results:
        print("No matching documents found.")
        return

    for r in results:
        preview = r["text"][:150].replace("\n", " ")
        print(f"Source:   {r['source']}")
        print(f"Score:    {r['score']:.4f}")
        print(f"Found By: {r['found_by']}")
        print(f"Text:     {preview}")
        print("-" * 60)


if __name__ == "__main__":
    main()
