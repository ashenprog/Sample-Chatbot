"""
Knowledge base storage module for University of Moratuwa AI Chatbot.
Provides local SQLite storage for document chunks and vector embeddings,
with vector similarity search.
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from document_loader import DocumentChunk

logger = logging.getLogger(__name__)


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """
    Computes cosine similarity between two numeric vectors.
    Uses numpy if available; falls back to pure Python math.
    """
    try:
        import numpy as np

        a = np.array(vec_a, dtype=np.float32)
        b = np.array(vec_b, dtype=np.float32)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
    except ImportError:
        dot_product = sum(x * y for x, y in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(x * x for x in vec_a))
        norm_b = math.sqrt(sum(y * y for y in vec_b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot_product / (norm_a * norm_b)


class KnowledgeBase:
    """Manages SQLite storage and vector retrieval for document chunks."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Creates and returns a connection to the SQLite database."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self) -> None:
        """Initializes database tables if they do not exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    metadata_json TEXT DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS indexed_files (
                    source TEXT PRIMARY KEY,
                    chunk_count INTEGER NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_source ON chunks (source);")
            conn.commit()

    def clear(self) -> None:
        """Removes all stored chunks and file records from the knowledge base."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM chunks;")
            cursor.execute("DELETE FROM indexed_files;")
            conn.commit()
        logger.info("Knowledge base cleared.")

    def add_chunks(
        self,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]],
    ) -> int:
        """
        Inserts or replaces document chunks with their vector embeddings into SQLite.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunks count ({len(chunks)}) does not match embeddings count ({len(embeddings)})."
            )

        if not chunks:
            return 0

        files_chunk_count: Dict[str, int] = {}
        for chunk in chunks:
            files_chunk_count[chunk.source] = files_chunk_count.get(chunk.source, 0) + 1

        with self._get_connection() as conn:
            cursor = conn.cursor()
            for chunk, emb in zip(chunks, embeddings):
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO chunks
                    (chunk_id, source, chunk_index, content, embedding_json, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?);
                    """,
                    (
                        chunk.chunk_id,
                        chunk.source,
                        chunk.chunk_index,
                        chunk.content,
                        json.dumps(emb),
                        json.dumps(chunk.metadata),
                    ),
                )

            for source, count in files_chunk_count.items():
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO indexed_files (source, chunk_count)
                    VALUES (?, ?);
                    """,
                    (source, count),
                )

            conn.commit()

        logger.info("Saved %d chunks into knowledge base.", len(chunks))
        return len(chunks)

    def count_chunks(self) -> int:
        """Returns the total number of chunks stored in the knowledge base."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM chunks;")
            row = cursor.fetchone()
            return row[0] if row else 0

    def get_indexed_files(self) -> List[Dict[str, Any]]:
        """Returns a list of all indexed files and their chunk counts."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT source, chunk_count, updated_at FROM indexed_files ORDER BY source;")
            rows = cursor.fetchall()
            return [
                {
                    "source": row["source"],
                    "chunk_count": row["chunk_count"],
                    "updated_at": row["updated_at"],
                }
                for row in rows
            ]

    def get_all_chunks_with_embeddings(self) -> List[Tuple[DocumentChunk, List[float]]]:
        """Loads all chunks and deserializes their vector embeddings from SQLite."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT chunk_id, source, chunk_index, content, embedding_json, metadata_json FROM chunks;")
            rows = cursor.fetchall()
            results: List[Tuple[DocumentChunk, List[float]]] = []
            for row in rows:
                chunk = DocumentChunk(
                    chunk_id=row["chunk_id"],
                    source=row["source"],
                    chunk_index=row["chunk_index"],
                    content=row["content"],
                    metadata=json.loads(row["metadata_json"]) if row["metadata_json"] else {},
                )
                emb = json.loads(row["embedding_json"])
                results.append((chunk, emb))
            return results

    def search_by_embedding(
        self,
        query_embedding: List[float],
        top_k: int = 4,
        similarity_threshold: float = 0.35,
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Performs semantic vector search against all stored chunks.
        Returns a sorted list of (DocumentChunk, similarity_score) tuples.
        """
        all_records = self.get_all_chunks_with_embeddings()
        if not all_records:
            return []

        scored_chunks: List[Tuple[DocumentChunk, float]] = []
        for chunk, emb in all_records:
            score = cosine_similarity(query_embedding, emb)
            if score >= similarity_threshold:
                scored_chunks.append((chunk, score))

        # Sort descending by similarity score
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return scored_chunks[:top_k]
