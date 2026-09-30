"""
Configuration module for the University of Moratuwa AI Chatbot.
Loads settings and environment variables safely from .env.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

# Base project paths
BASE_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: Path = BASE_DIR / "data" / "knowledge"
DB_PATH: Path = BASE_DIR / "data" / "knowledge_base.db"
ENV_FILE: Path = BASE_DIR / ".env"


def _load_env_file() -> None:
    """Loads .env variables using python-dotenv or a pure-Python fallback."""
    try:
        from dotenv import load_dotenv

        load_dotenv(dotenv_path=ENV_FILE)
    except ImportError:
        # Built-in fallback parser if python-dotenv is not yet installed
        if ENV_FILE.exists():
            try:
                with open(ENV_FILE, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if "=" in line:
                            k, v = line.split("=", 1)
                            k = k.strip()
                            v = v.strip().strip("'\"")
                            if k and k not in os.environ:
                                os.environ[k] = v
            except Exception:
                pass


_load_env_file()


# Gemini API Configuration
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
GEMINI_EMBEDDING_MODEL: str = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001").strip()

# RAG & Chunking Configuration
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "600"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))
TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "4"))
SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.35"))

# Conversation Memory Configuration
MAX_HISTORY_MESSAGES: int = int(os.getenv("MAX_HISTORY_MESSAGES", "20"))


def validate_api_key(api_key: str | None = None) -> Tuple[bool, str]:
    """
    Validates whether the Gemini API key is configured properly.

    Returns:
        Tuple[bool, str]: (is_valid, error_or_success_message)
    """
    key = api_key if api_key is not None else GEMINI_API_KEY
    if not key:
        return False, (
            "GEMINI_API_KEY is not set. Please create a .env file based on .env.example "
            "and set your valid Google Gemini API key."
        )
    if key in ("your_api_key_here", "YOUR_GEMINI_API_KEY", "your_gemini_api_key_here"):
        return False, (
            "GEMINI_API_KEY contains the default placeholder. "
            "Please update your .env file with your actual Gemini API key from Google AI Studio."
        )
    if len(key) < 10:
        return False, (
            "GEMINI_API_KEY appears to be invalid or too short. "
            "Please verify your API key in Google AI Studio."
        )
    return True, "API key is valid."
