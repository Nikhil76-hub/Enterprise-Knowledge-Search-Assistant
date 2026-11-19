"""Configuration loader for Enterprise Knowledge Search Assistant."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# Constants
CHUNKS_PATH = BASE_DIR / "data" / "chunks.jsonl"
CHROMA_DIR = BASE_DIR / "data" / "chroma"
EMBED_MODEL: str = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")

# LLM Backend configuration (allowed values: gemini, ollama)
ALLOWED_BACKENDS = {"gemini", "ollama"}
LLM_BACKEND: str = os.getenv("LLM_BACKEND", "gemini").lower()

if LLM_BACKEND not in ALLOWED_BACKENDS:
    raise ValueError(
        f"Invalid LLM_BACKEND '{LLM_BACKEND}'. Allowed values are: {', '.join(sorted(ALLOWED_BACKENDS))}."
    )

GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL: str | None = os.getenv("GEMINI_MODEL")

if LLM_BACKEND == "gemini":
    missing = []
    if not GEMINI_API_KEY:
        missing.append("GEMINI_API_KEY")
    if not GEMINI_MODEL:
        missing.append("GEMINI_MODEL")
    if missing:
        raise EnvironmentError(
            f"Missing required environment variable(s) when LLM_BACKEND='gemini': {', '.join(missing)}. "
            "Please define them in your environment or in a .env file."
        )
