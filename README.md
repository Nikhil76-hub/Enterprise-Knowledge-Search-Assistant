# Enterprise Knowledge Search Assistant

A fully local, free, grounded Retrieval-Augmented Generation (RAG) application with zero Google Cloud dependencies. It indexes enterprise knowledge documents and provides grounded answers with citations using hybrid search and configurable LLM backends (Gemini or Ollama).

## Architecture

The assistant employs a multi-stage local retrieval and generation pipeline:

1. **Document Ingestion & Chunking (`src/ingest.py`)**:
   - Parses enterprise Markdown documents from `data/raw/`.
   - Splits content into ~500-character chunks with 50-character overlap, prioritizing paragraph and heading boundaries.
   - Outputs structured chunks to `data/chunks.jsonl`.

2. **BM25 Lexical Keyword Search (`src/search_keyword.py`)**:
   - Uses `rank-bm25` for exact keyword, acronym, policy limit, SLA number, and entity matching.

3. **Chroma Dense Vector Search (`src/search_vector.py`)**:
   - Generates local embeddings using `sentence-transformers` (`all-MiniLM-L6-v2`).
   - Stores and queries dense embeddings locally in `chromadb` (`data/chroma/`) for semantic retrieval.

4. **Hybrid Search Fusion**:
   - Combines lexical (BM25) and semantic (Chroma) search results using rank fusion to maximize retrieval recall and precision.

5. **Grounded Answer Generation (`src/answer.py`)**:
   - Synthesizes answers strictly grounded in retrieved document chunks with clear source citations.
   - Supports local `ollama` (100% offline and free) or `gemini` (via free Gemini API key).

## Project Structure

```text
├── data/
│   ├── raw/               # Raw enterprise markdown source files
│   ├── chunks.jsonl       # Chunked document records
│   └── chroma/            # Persistent local Chroma vector database
├── src/
│   ├── config.py          # Environment configuration & path constants
│   ├── ingest.py          # Document chunking pipeline
│   ├── search_keyword.py  # BM25 keyword retrieval
│   ├── search_vector.py   # Chroma vector similarity search
│   └── answer.py          # Grounded response synthesis
├── app.py                 # Streamlit UI entrypoint
├── requirements.txt       # Local Python dependencies
├── .env.example           # Sample environment variables
└── README.md
```

## Setup Instructions

### 1. Set Up Virtual Environment

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Configure your LLM backend:
- `LLM_BACKEND`: `gemini` (default) or `ollama`
- `GEMINI_API_KEY`: Required only when `LLM_BACKEND=gemini` (obtain a free key at [Google AI Studio](https://aistudio.google.com/))

### 4. Ingest & Chunk Documents

Chunk the sample enterprise knowledge documents:

```bash
python src/ingest.py chunk
```

### 5. Run the Streamlit Application

```bash
streamlit run app.py
```
