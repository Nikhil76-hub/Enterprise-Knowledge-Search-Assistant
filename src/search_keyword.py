"""Keyword search module using BM25."""

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rank_bm25 import BM25Okapi

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from src.config import CHUNKS_PATH
except Exception:
    CHUNKS_PATH = BASE_DIR / "data" / "chunks.jsonl"

_CHUNKS: Optional[List[Dict[str, Any]]] = None
_BM25_INDEX: Optional[BM25Okapi] = None


def tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric words using regex."""
    return re.findall(r"\w+", text.lower())


def _load_index() -> Tuple[List[Dict[str, Any]], BM25Okapi]:
    """Load chunks and build BM25 index in memory (cached at module level)."""
    global _CHUNKS, _BM25_INDEX
    if _CHUNKS is None or _BM25_INDEX is None:
        chunks_file = Path(CHUNKS_PATH)
        if not chunks_file.exists():
            raise FileNotFoundError(f"Chunks file not found at {chunks_file}. Run ingest.py chunk first.")

        chunks: List[Dict[str, Any]] = []
        with open(chunks_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))

        corpus = [tokenize(c.get("text", "")) for c in chunks]
        _CHUNKS = chunks
        _BM25_INDEX = BM25Okapi(corpus)

    return _CHUNKS, _BM25_INDEX


def search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Search chunks using BM25, returning top_k results with score > 0 sorted best first."""
    chunks, bm25 = _load_index()
    if not chunks:
        return []

    tokens = tokenize(query)
    if not tokens:
        return []

    scores = bm25.get_scores(tokens)

    results: List[Dict[str, Any]] = []
    for idx, score in enumerate(scores):
        if score > 0:
            c = chunks[idx]
            results.append({
                "id": c["id"],
                "source": c.get("source", ""),
                "title": c.get("title", ""),
                "text": c.get("text", ""),
                "score": float(score),
            })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def main():
    if len(sys.argv) < 2:
        print("Usage: python src/search_keyword.py \"<query>\"")
        sys.exit(1)

    query = sys.argv[1].strip()
    results = search(query, top_k=3)

    if not results:
        print("No matching documents found.")
        return

    for r in results:
        preview = r["text"][:150].replace("\n", " ")
        print(f"Source: {r['source']}")
        print(f"Score:  {r['score']:.4f}")
        print(f"Text:   {preview}")
        print("-" * 60)


if __name__ == "__main__":
    main()
