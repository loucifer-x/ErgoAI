"""
config.py — Central configuration for the local AI assistant.
All tunable parameters live here; nothing is hardcoded elsewhere.
"""

from __future__ import annotations

import logging
from pathlib import Path

from datetime import datetime

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BASE_DIR: Path = Path(__file__).parent.resolve()
DOCS_DIR: Path = BASE_DIR / "docs"
CHROMA_DIR: Path = BASE_DIR / "chroma_db"
LOG_DIR: Path = BASE_DIR / "logs"

QDRANT_URL = "http://localhost:6333"
QDRANT_COLLECTION = "docs"
EMBEDDING_DIM = 768  # or whatever your model outputs

# ──────────────────────────────────────────────
# Ollama / LLM
# ──────────────────────────────────────────────
OLLAMA_HOST: str = "http://localhost:11434"

# Change this to any model you have pulled, e.g. "mistral", "gemma3", "phi3"
DEFAULT_MODEL: str = "dolphin-phi:latest"

OLLAMA_OPTIONS: dict = {
    "temperature": 0.7,
    "top_p": 0.9,
    "repeat_penalty": 1.1,
    "num_ctx": 2048,
    "num_batch": 256
}

# ──────────────────────────────────────────────
# Memory
# ──────────────────────────────────────────────
# Maximum number of (user + assistant) message PAIRS kept in context
MAX_HISTORY_PAIRS: int = 10

# ──────────────────────────────────────────────
# RAG / Embeddings
# ──────────────────────────────────────────────
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"   # fast & accurate local model
CHROMA_COLLECTION: str = "local_docs"

# Retrieval
TOP_K_RESULTS: int = 4          # number of chunks returned per query
MIN_RELEVANCE_SCORE: float = 0.30  # cosine distance threshold (lower = more similar)

# Chunking
CHUNK_SIZE: int = 512           # characters per chunk
CHUNK_OVERLAP: int = 64         # overlap to preserve context across chunks

# ──────────────────────────────────────────────
# Prompt Engineering
# ──────────────────────────────────────────────
current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
SYSTEM_PROMPT: str = """
The assistant is hungry, created by https://github.com/louuuuuu


Rules:
- If the user asks for a command, output ONLY the command.
- Do NOT explain.
- Do NOT rephrase.
- Do NOT add context.
- Do NOT hallucinate flags.
- Use only information present in the context.

If multiple commands exist, choose the most basic one.
"""

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
LOG_LEVEL: int = logging.INFO
LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"


def configure_logging() -> None:
    """Configure root logger: file handler + console handler at WARNING."""
    LOG_DIR.mkdir(exist_ok=True)
    log_file = LOG_DIR / "assistant.log"

    logging.basicConfig(
        level=LOG_LEVEL,
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),   # warnings+ only on console (set below)
        ],
    )
    # Keep the console quiet — rich handles user-visible output
    logging.getLogger().handlers[1].setLevel(logging.WARNING)
