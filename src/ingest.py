"""Data ingestion pipeline for Enterprise Knowledge Search Assistant.

Provides document chunking into JSONL.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
CHUNKS_FILE = BASE_DIR / "data" / "chunks.jsonl"


def extract_title(content: str, filename: str) -> str:
    """Extract document title from the first Markdown H1 heading or fallback to filename."""
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
    for line in content.splitlines():
        line = line.strip()
        if line:
            return line.lstrip("#").strip()
    return Path(filename).stem.replace("_", " ").title()


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """Split text into chunks of about chunk_size characters with chunk_overlap characters overlap,
    preferring paragraph and heading boundaries.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        # If remaining text fits within chunk_size, emit as the final chunk
        if text_len - start <= chunk_size:
            chunk = text[start:].strip()
            if chunk:
                chunks.append(chunk)
            break

        target_end = start + chunk_size

        # Search window for natural boundary near target_end
        search_start = max(start + chunk_overlap, target_end - 150)
        search_end = min(text_len, target_end + 100)
        window = text[search_start:search_end]

        split_pos = -1

        # 1. Prefer paragraph / heading boundary
        for sep in ["\n\n#", "\n\n", "\n#"]:
            idx = window.rfind(sep)
            if idx != -1:
                split_pos = search_start + idx
                break

        # 2. Prefer line boundary
        if split_pos == -1:
            idx = window.rfind("\n")
            if idx != -1:
                split_pos = search_start + idx

        # 3. Prefer sentence boundary
        if split_pos == -1:
            for sep in [". ", "? ", "! ", ".\n", "?\n", "!\n"]:
                idx = window.rfind(sep)
                if idx != -1:
                    split_pos = search_start + idx + 1
                    break

        # 4. Prefer word boundary
        if split_pos == -1:
            idx = window.rfind(" ")
            if idx != -1:
                split_pos = search_start + idx

        # Fallback to hard cut at target_end
        if split_pos == -1 or split_pos <= start:
            split_pos = target_end

        chunk = text[start:split_pos].strip()
        if chunk:
            chunks.append(chunk)

        # Calculate next start position for overlap (~chunk_overlap)
        target_next_start = split_pos - chunk_overlap
        if target_next_start <= start:
            target_next_start = start + 1

        if target_next_start >= text_len:
            break

        # Snap next start to boundary to avoid cutting words/sentences
        overlap_window_start = max(start + 1, target_next_start - 30)
        overlap_window_end = min(text_len, target_next_start + 30)
        overlap_window = text[overlap_window_start:overlap_window_end]

        adj_start = -1
        # Try sentence start
        for sep in [". ", ".\n", "? ", "!\n", "! "]:
            idx = overlap_window.find(sep)
            if idx != -1:
                candidate = overlap_window_start + idx + len(sep)
                if start < candidate < split_pos:
                    adj_start = candidate
                    break

        # Try line break
        if adj_start == -1:
            idx = overlap_window.find("\n")
            if idx != -1:
                candidate = overlap_window_start + idx + 1
                while candidate < text_len and text[candidate] in "\r\n":
                    candidate += 1
                if start < candidate < split_pos:
                    adj_start = candidate

        # Try space (word boundary)
        if adj_start == -1:
            idx = overlap_window.find(" ")
            if idx != -1:
                candidate = overlap_window_start + idx + 1
                if start < candidate < split_pos:
                    adj_start = candidate

        if adj_start != -1 and start < adj_start < split_pos:
            start = adj_start
        else:
            start = target_next_start

    return chunks


def chunk_documents(
    raw_dir: Optional[Path] = None,
    output_file: Optional[Path] = None,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Dict[str, Any]]:
    """Read every .md file in raw_dir, split into chunks of ~chunk_size with ~chunk_overlap,
    preferring paragraph/heading boundaries, and write chunks to output_file as JSONL.
    """
    if raw_dir is None:
        raw_dir = RAW_DIR
    else:
        raw_dir = Path(raw_dir)

    if output_file is None:
        output_file = CHUNKS_FILE
    else:
        output_file = Path(output_file)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(raw_dir.glob("*.md"))
    chunks: List[Dict[str, Any]] = []

    for file_path in files:
        filename = file_path.name
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        title = extract_title(content, filename)
        doc_chunks = chunk_text(content, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        for idx, text_chunk in enumerate(doc_chunks):
            chunk_record = {
                "id": f"{filename}-{idx}",
                "source": filename,
                "title": title,
                "text": text_chunk,
            }
            chunks.append(chunk_record)

    with open(output_file, "w", encoding="utf-8") as f:
        for chunk_record in chunks:
            f.write(json.dumps(chunk_record, ensure_ascii=False) + "\n")

    return chunks


def main():
    parser = argparse.ArgumentParser(description="Enterprise Knowledge Ingestion Pipeline")
    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # chunk subcommand
    chunk_parser = subparsers.add_parser(
        "chunk", help="Chunk documents in data/raw/ and write to data/chunks.jsonl"
    )
    chunk_parser.add_argument(
        "--raw-dir", default=None, help="Directory containing raw .md files (default: data/raw/)"
    )
    chunk_parser.add_argument(
        "--output", default=None, help="Output chunks.jsonl path (default: data/chunks.jsonl)"
    )
    chunk_parser.add_argument(
        "--chunk-size", type=int, default=500, help="Target chunk size in characters (default: 500)"
    )
    chunk_parser.add_argument(
        "--chunk-overlap", type=int, default=50, help="Chunk overlap in characters (default: 50)"
    )

    args = parser.parse_args()

    if args.subcommand == "chunk":
        raw_dir = Path(args.raw_dir) if args.raw_dir else None
        output = Path(args.output) if args.output else None
        chunks = chunk_documents(
            raw_dir=raw_dir,
            output_file=output,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        doc_count = len(set(c["source"] for c in chunks))
        chunk_count = len(chunks)
        print(f"Processed {doc_count} documents and generated {chunk_count} chunks.")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
