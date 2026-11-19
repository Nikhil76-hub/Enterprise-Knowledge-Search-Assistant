"""Vector similarity search module using Chroma and SentenceTransformers."""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from src.config import CHROMA_DIR, CHUNKS_PATH, EMBED_MODEL
except Exception:
    CHUNKS_PATH = BASE_DIR / "data" / "chunks.jsonl"
    CHROMA_DIR = BASE_DIR / "data" / "chroma"
    EMBED_MODEL = "all-MiniLM-L6-v2"

# Module-level model and client cache
_MODEL = None
_CHROMA_CLIENT = None


def get_embedding_model(model_name: str = EMBED_MODEL):
    """Load and cache the sentence-transformers embedding model."""
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer(model_name)
    return _MODEL


def get_chroma_client(chroma_dir: Path = CHROMA_DIR):
    """Get or create persistent Chroma client."""
    global _CHROMA_CLIENT
    if _CHROMA_CLIENT is None:
        import chromadb
        _CHROMA_CLIENT = chromadb.PersistentClient(path=str(chroma_dir))
    return _CHROMA_CLIENT


def build_index(chunks_path: Path = CHUNKS_PATH, chroma_dir: Path = CHROMA_DIR) -> int:
    """Read chunks from CHUNKS_PATH, generate normalized embeddings,
    and store in a persistent Chroma collection 'kb_chunks' using cosine distance.
    Rebuilds cleanly by deleting and recreating the collection.
    """
    if not Path(chunks_path).exists():
        raise FileNotFoundError(f"Chunks file not found at {chunks_path}. Run ingest.py chunk first.")

    chunks: List[Dict[str, Any]] = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    if not chunks:
        print("No chunks found in chunks file.")
        return 0

    client = get_chroma_client(chroma_dir)

    # Rebuild cleanly: delete existing collection if it exists
    try:
        client.delete_collection(name="kb_chunks")
    except Exception:
        pass

    collection = client.create_collection(
        name="kb_chunks",
        metadata={"hnsw:space": "cosine"},
    )

    ids = [chunk["id"] for chunk in chunks]
    texts = [chunk["text"] for chunk in chunks]
    metadatas = [
        {
            "source": chunk.get("source", ""),
            "title": chunk.get("title", ""),
            "text": chunk.get("text", ""),
        }
        for chunk in chunks
    ]

    model = get_embedding_model()
    embeddings = model.encode(texts, normalize_embeddings=True).tolist()

    collection.add(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=texts,
    )

    return len(chunks)


def search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Embed query, retrieve top_k chunks from Chroma, and return ranked results.
    Score is calculated as 1 - cosine distance, sorted best first.
    """
    client = get_chroma_client()
    try:
        collection = client.get_collection(name="kb_chunks")
    except Exception as e:
        raise RuntimeError("Collection 'kb_chunks' does not exist. Please run build_index() first.") from e

    count = collection.count()
    if count == 0:
        return []

    actual_k = min(top_k, count)
    model = get_embedding_model()
    query_embedding = model.encode([query], normalize_embeddings=True).tolist()

    query_results = collection.query(
        query_embeddings=query_embedding,
        n_results=actual_k,
        include=["metadatas", "documents", "distances"],
    )

    results: List[Dict[str, Any]] = []
    if not query_results or not query_results.get("ids") or not query_results["ids"][0]:
        return results

    ids = query_results["ids"][0]
    metadatas = query_results["metadatas"][0] if query_results.get("metadatas") else [{}] * len(ids)
    documents = query_results["documents"][0] if query_results.get("documents") else [""] * len(ids)
    distances = query_results["distances"][0] if query_results.get("distances") else [0.0] * len(ids)

    for i in range(len(ids)):
        meta = metadatas[i] or {}
        chunk_text_val = meta.get("text") or documents[i] or ""
        distance = distances[i]
        score = 1.0 - float(distance)

        results.append({
            "id": ids[i],
            "source": meta.get("source", ""),
            "title": meta.get("title", ""),
            "text": chunk_text_val,
            "score": score,
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python src/search_vector.py build")
        print("  python src/search_vector.py \"<query>\"")
        sys.exit(1)

    command = sys.argv[1].strip()

    if command == "build":
        count = build_index()
        print(f"Indexed {count} chunks into Chroma collection 'kb_chunks'.")
    else:
        results = search(command, top_k=3)
        for r in results[:3]:
            preview = r["text"][:150].replace("\n", " ")
            print(f"Source: {r['source']}")
            print(f"Score:  {r['score']:.4f}")
            print(f"Text:   {preview}")
            print("-" * 60)


if __name__ == "__main__":
    main()
