"""Grounded answer generation module using retrieved context and Gemini."""

import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from src.config import GEMINI_API_KEY, GEMINI_MODEL
except Exception:
    from dotenv import load_dotenv
    load_dotenv()
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

GEMINI_FALLBACK_MODEL: Optional[str] = os.getenv("GEMINI_FALLBACK_MODEL") or None

try:
    from src.hybrid import hybrid_search
except ImportError:
    from hybrid import hybrid_search


def _is_retryable(exc: Exception) -> bool:
    """Return True only for 503 UNAVAILABLE or 429 rate-limit errors."""
    msg = str(exc)
    return "503" in msg or "UNAVAILABLE" in msg or "429" in msg or "RESOURCE_EXHAUSTED" in msg


def _call_gemini(client: Any, model: str, prompt: str) -> str:
    """Call the Gemini API and return the response text."""
    from google.genai import types
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.2),
    )
    return response.text.strip() if response.text else ""


def answer_question(question: str) -> Dict[str, Any]:
    """Answer a user question grounded in retrieved enterprise knowledge documents."""
    chunks = hybrid_search(question, top_k=5)

    # Refusal guard BEFORE calling the LLM
    if not chunks:
        return {
            "answer": "I couldn't find this in the knowledge base.",
            "sources": [],
            "grounded": False,
        }

    vector_scores = [c["vector_score"] for c in chunks if c.get("vector_score") is not None]
    best_vector_score = max(vector_scores) if vector_scores else 0.0

    if best_vector_score < 0.25:
        return {
            "answer": "I couldn't find this in the knowledge base.",
            "sources": [],
            "grounded": False,
        }

    # Build prompt with numbered chunks
    context_blocks = []
    for idx, c in enumerate(chunks, start=1):
        source_name = c.get("source", "unknown")
        chunk_text = c.get("text", "")
        context_blocks.append(f"[{idx}] Source: {source_name}\n{chunk_text}")

    context_str = "\n\n".join(context_blocks)

    prompt = f"""Context:
{context_str}

Instructions:
You are an enterprise knowledge assistant. Answer the user's question using ONLY the provided context above.
Follow these rules strictly:
1. Answer ONLY from the provided context. Do not use outside knowledge or assumptions.
2. Cite sources inline using bracketed numbers corresponding to the context, like [1] or [2].
3. If the context does not contain the answer, say exactly: "I couldn't find this in the knowledge base."
4. Give a complete, helpful answer that covers everything relevant in the context, not just the minimum.
   Start with a direct answer in one sentence, then add supporting details: steps, limits, deadlines,
   exceptions, approval requirements, and contact emails or phone numbers when the context has them.
5. If the question has several parts, answer each part separately.
   Use short bullet points for lists of rules, limits, or steps. Use plain sentences otherwise.
6. Keep inline citations like [1], [2] after the facts they support.

Question: {question}
Answer:"""

    try:
        from google import genai

        client = genai.Client(api_key=GEMINI_API_KEY)

        # Attempt primary model with up to 2 retries for 503/429 only
        _retry_delays = [3, 8]
        last_exc: Optional[Exception] = None
        answer_text: Optional[str] = None

        for attempt in range(3):  # attempt 0, 1, 2
            try:
                answer_text = _call_gemini(client, GEMINI_MODEL, prompt)
                break  # success
            except Exception as exc:
                if _is_retryable(exc) and attempt < 2:
                    last_exc = exc
                    time.sleep(_retry_delays[attempt])
                else:
                    last_exc = exc
                    break  # non-retryable or exhausted retries

        # Fallback model if primary failed and fallback is configured
        if answer_text is None and GEMINI_FALLBACK_MODEL:
            try:
                answer_text = _call_gemini(client, GEMINI_FALLBACK_MODEL, prompt)
                last_exc = None
            except Exception as exc:
                last_exc = exc

        if answer_text is None:
            fallback_sources = [
                {"source": c.get("source", ""), "title": c.get("title", ""), "text": c.get("text", "")}
                for c in chunks
            ]
            return {
                "answer": "",
                "sources": fallback_sources,
                "grounded": True,
                "error": "The model is busy right now. Please try again in a minute.",
            }

    except Exception:
        fallback_sources = [
            {"source": c.get("source", ""), "title": c.get("title", ""), "text": c.get("text", "")}
            for c in chunks
        ]
        return {
            "answer": "",
            "sources": fallback_sources,
            "grounded": True,
            "error": "The model is busy right now. Please try again in a minute.",
        }

    if not answer_text or "I couldn't find this in the knowledge base." in answer_text:
        return {
            "answer": "I couldn't find this in the knowledge base.",
            "sources": [],
            "grounded": False,
        }

    # Extract cited chunks
    cited_nums = set(int(m) for m in re.findall(r"\[(\d+)\]", answer_text))
    cited_sources = []
    for idx in sorted(cited_nums):
        if 1 <= idx <= len(chunks):
            c = chunks[idx - 1]
            cited_sources.append({
                "source": c.get("source", ""),
                "title": c.get("title", ""),
                "text": c.get("text", ""),
            })

    if not cited_sources:
        cited_sources = [
            {
                "source": c.get("source", ""),
                "title": c.get("title", ""),
                "text": c.get("text", ""),
            }
            for c in chunks
        ]

    return {
        "answer": answer_text,
        "sources": cited_sources,
        "grounded": True,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python src/answer.py \"<question>\"")
        sys.exit(1)

    question = sys.argv[1].strip()
    result = answer_question(question)
    print(result["answer"])

    unique_sources = []
    seen = set()
    for s in result.get("sources", []):
        src_name = s.get("source")
        if src_name and src_name not in seen:
            seen.add(src_name)
            unique_sources.append(src_name)

    if unique_sources:
        print("\nSources:")
        for src in unique_sources:
            print(f"- {src}")


if __name__ == "__main__":
    main()
