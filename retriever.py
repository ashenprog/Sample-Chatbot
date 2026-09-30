"""
Retriever module for semantic search over University of Moratuwa knowledge chunks.
Performs query contextualization from conversation history, generates embeddings,
and retrieves the most relevant chunks using vector similarity.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import config
from document_loader import DocumentChunk
from knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Represents a retrieved chunk with its similarity score and source."""
    chunk: DocumentChunk
    score: float
    source: str


class Retriever:
    """
    Semantic retriever that queries the KnowledgeBase using vector embeddings.
    Supports query contextualization across multi-turn conversations.
    """

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        gemini_client: Any,
        top_k: int = config.TOP_K_RESULTS,
        similarity_threshold: float = config.SIMILARITY_THRESHOLD,
    ):
        self.kb = knowledge_base
        self.gemini_client = gemini_client
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

    def contextualize_query(self, query: str, history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Enriches short or pronoun-heavy follow-up questions with context from
        recent turns in the conversation (e.g. 'What departments does it have?'
        following 'Tell me about the Faculty of Engineering').
        """
        if not history:
            return query

        query_clean = query.strip()
        words = re.findall(r"\b\w+\b", query_clean.lower())

        pronouns = {"it", "its", "they", "them", "their", "this", "that", "these", "those", "there"}
        has_pronoun = any(w in pronouns for w in words)
        is_short = len(words) <= 5

        if has_pronoun or is_short:
            # Look back through recent user queries for university topics/entities
            recent_user_turns = [turn["content"] for turn in history if turn.get("role") == "user"]
            if recent_user_turns:
                last_turn = recent_user_turns[-1]
                # Combine last question context with current query
                contextualized = f"{last_turn} {query_clean}"
                logger.debug("Contextualized query '%s' to '%s'", query_clean, contextualized)
                return contextualized

        return query_clean

    def retrieve(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]] = None,
        top_k: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """
        Retrieves top relevant chunks for a user question.

        Args:
            query: The user's query string.
            history: Optional conversation history for context resolution.
            top_k: Optional override for the number of chunks to retrieve.

        Returns:
            List of RetrievalResult objects sorted by relevance.
        """
        effective_top_k = top_k or self.top_k
        search_query = self.contextualize_query(query, history=history)

        try:
            # Generate query embedding
            query_emb = self.gemini_client.get_embedding(search_query)
            if not query_emb:
                logger.warning("Empty query embedding generated for '%s'", search_query)
                return []

            # Perform vector search in knowledge base
            matches = self.kb.search_by_embedding(
                query_embedding=query_emb,
                top_k=effective_top_k,
                similarity_threshold=self.similarity_threshold,
            )

            results: List[RetrievalResult] = []
            for chunk, score in matches:
                results.append(
                    RetrievalResult(
                        chunk=chunk,
                        score=score,
                        source=chunk.source,
                    )
                )

            return results
        except Exception as e:
            logger.error("Error during retrieval: %s", e)
            return []
