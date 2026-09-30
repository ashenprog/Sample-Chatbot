"""
Gemini Client module for the University of Moratuwa AI Chatbot.
Uses the official Google GenAI SDK (`google-genai`) with retry mechanisms,
robust error handling, text generation, and embeddings generation.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import config

logger = logging.getLogger(__name__)
# Suppress Google GenAI internal AFC warning messages on generate_content
logging.getLogger("google.genai.models").setLevel(logging.ERROR)



class GeminiClientError(Exception):
    """Base exception for Gemini client errors with user-friendly messages."""
    pass


class GeminiAPIKeyError(GeminiClientError):
    """Raised when the API key is missing or invalid."""
    pass


class GeminiRateLimitError(GeminiClientError):
    """Raised when API rate limits or quota are exceeded."""
    pass


class GeminiNetworkError(GeminiClientError):
    """Raised when network connectivity fails."""
    pass


class GeminiClient:
    """
    Wrapper around the official Google GenAI SDK (`google-genai`).
    Provides text generation and embeddings with automatic retries and user-friendly errors.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        embedding_model: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        self.api_key = api_key.strip() if api_key is not None else config.GEMINI_API_KEY
        self.model = (model or config.GEMINI_MODEL).strip()
        self.embedding_model = (embedding_model or config.GEMINI_EMBEDDING_MODEL).strip()
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._client: Any = None

    def _get_client(self) -> Any:
        """Lazily initializes and returns the Google GenAI client."""
        if self._client is not None:
            return self._client

        is_valid, msg = config.validate_api_key(self.api_key)
        if not is_valid:
            raise GeminiAPIKeyError(msg)

        try:
            from google import genai

            self._client = genai.Client(api_key=self.api_key)
            return self._client
        except ImportError:
            raise GeminiClientError(
                "The 'google-genai' package is not installed. "
                "Please run: pip install google-genai"
            )
        except Exception as e:
            raise GeminiClientError(f"Failed to initialize Gemini Client: {self._sanitize_error(e)}")

    def _sanitize_error(self, error: Exception) -> str:
        """Strips sensitive credentials and formats clean error messages."""
        msg = str(error)
        if self.api_key and self.api_key in msg:
            msg = msg.replace(self.api_key, "[REDACTED_API_KEY]")
        return msg

    def _handle_api_exception(self, e: Exception) -> None:
        """Translates low-level SDK and HTTP exceptions into clean, user-friendly exceptions."""
        error_str = self._sanitize_error(e).lower()

        if "api_key" in error_str or "unauthenticated" in error_str or "403" in error_str:
            raise GeminiAPIKeyError(
                "Invalid or unauthorized Gemini API key. Please check your GEMINI_API_KEY in .env."
            ) from e

        if "quota" in error_str or "resource_exhausted" in error_str or "429" in error_str:
            raise GeminiRateLimitError(
                "Gemini API rate limit or quota exceeded. Please wait a moment and try again."
            ) from e

        if (
            "connection" in error_str
            or "timeout" in error_str
            or "name resolution" in error_str
            or "network" in error_str
        ):
            raise GeminiNetworkError(
                "Network connection failure. Please check your internet connection and try again."
            ) from e

        raise GeminiClientError(f"Gemini API error: {self._sanitize_error(e)}") from e

    def generate_response(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Generates text using the Gemini API with exponential backoff on transient errors.

        Args:
            prompt: The current user prompt (including context if RAG-augmented).
            system_instruction: High-level system instructions.
            history: Optional conversation turns [{'role': 'user'|'assistant', 'content': '...'}].

        Returns:
            The generated response string.
        """
        client = self._get_client()

        # Build contents incorporating conversation history
        contents: List[Any] = []
        if history:
            for turn in history:
                role = "user" if turn["role"] == "user" else "model"
                contents.append({"role": role, "parts": [{"text": turn["content"]}]})

        # Append current prompt as the latest user turn
        contents.append({"role": "user", "parts": [{"text": prompt}]})

        # Set up generation config and disable automatic function calling (AFC) to avoid SDK warning
        config_kwargs: Dict[str, Any] = {}
        try:
            from google.genai import types

            afc_config = None
            if hasattr(types, "AutomaticFunctionCallingConfig"):
                try:
                    afc_config = types.AutomaticFunctionCallingConfig(disable=True)
                except Exception:
                    pass

            config_kwargs["config"] = types.GenerateContentConfig(
                system_instruction=system_instruction,
                automatic_function_calling=afc_config,
            )
        except Exception:
            if system_instruction:
                config_kwargs["config"] = {"system_instruction": system_instruction}

        # Build candidate models list: primary model first, followed by fallbacks if 503 occurs
        candidate_models = [self.model]
        fallback_candidates = ["gemini-2.0-flash", "gemini-2.5-flash-lite", "gemini-1.5-flash"]
        for fb in fallback_candidates:
            if fb != self.model and fb not in candidate_models:
                candidate_models.append(fb)

        last_exception: Optional[Exception] = None

        for model_to_try in candidate_models:
            attempt = 0
            while attempt < self.max_retries:
                attempt += 1
                try:
                    response = client.models.generate_content(
                        model=model_to_try,
                        contents=contents,
                        **config_kwargs,
                    )
                    if hasattr(response, "text") and response.text:
                        return response.text.strip()
                    elif hasattr(response, "output_text") and response.output_text:
                        return response.output_text.strip()
                    else:
                        return "I was unable to generate an answer. Please try rephrasing your question."
                except Exception as e:
                    last_exception = e
                    err_text = str(e).lower()
                    is_503 = "503" in err_text or "unavailable" in err_text or "high demand" in err_text
                    is_transient = is_503 or any(
                        token in err_text
                        for token in ["500", "deadline", "timed out", "retry"]
                    )

                    if is_transient and attempt < self.max_retries:
                        sleep_time = self.retry_delay * (2 ** (attempt - 1))
                        logger.warning(
                            "Transient error from Gemini API using '%s' (attempt %d/%d). Retrying in %.1fs...",
                            model_to_try,
                            attempt,
                            self.max_retries,
                            sleep_time,
                        )
                        time.sleep(sleep_time)
                        continue

                    # If this model specifically failed with 503 (high demand) and we have fallback models
                    if is_503 and model_to_try != candidate_models[-1]:
                        next_model = candidate_models[candidate_models.index(model_to_try) + 1]
                        print(
                            f"\n[Notice] '{model_to_try}' is currently experiencing high demand on Google's servers. "
                            f"Automatically switching to '{next_model}'..."
                        )
                        break  # Break retry loop to try next model in candidate_models

                    # For non-transient or non-503 errors, raise immediately
                    if not is_503:
                        self._handle_api_exception(e)

        if last_exception:
            self._handle_api_exception(last_exception)

        raise GeminiClientError("Maximum retries exceeded. Please try again later.")


    def get_embedding(self, text: str) -> List[float]:
        """
        Generates a semantic vector embedding for a single text chunk or query.
        """
        if not text or not text.strip():
            return []

        embeddings = self.get_embeddings_batch([text])
        return embeddings[0] if embeddings else []

    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generates semantic vector embeddings for a batch of texts.
        """
        if not texts:
            return []

        client = self._get_client()
        clean_texts = [t.strip() for t in texts]

        # To respect rate limits and payload sizes, batch in chunks of 20
        batch_size = 20
        all_embeddings: List[List[float]] = []

        for i in range(0, len(clean_texts), batch_size):
            chunk_batch = clean_texts[i : i + batch_size]
            attempt = 0
            while attempt < self.max_retries:
                attempt += 1
                try:
                    result = client.models.embed_content(
                        model=self.embedding_model,
                        contents=chunk_batch,
                    )

                    # Result parsing: response has .embeddings list
                    if hasattr(result, "embeddings") and result.embeddings:
                        for emb in result.embeddings:
                            if hasattr(emb, "values"):
                                all_embeddings.append(list(emb.values))
                            elif isinstance(emb, dict) and "values" in emb:
                                all_embeddings.append(list(emb["values"]))
                            elif isinstance(emb, (list, tuple)):
                                all_embeddings.append(list(emb))
                            else:
                                raise ValueError(f"Unrecognized embedding response item: {type(emb)}")
                        break
                    else:
                        raise GeminiClientError("Embed content returned an empty or invalid result.")
                except Exception as e:
                    err_text = str(e).lower()
                    is_transient = any(
                        token in err_text
                        for token in ["503", "500", "unavailable", "deadline", "timed out", "retry", "429"]
                    )
                    if is_transient and attempt < self.max_retries:
                        sleep_time = self.retry_delay * (2 ** (attempt - 1))
                        logger.warning(
                            "Transient embedding error (attempt %d/%d). Retrying in %.1fs...",
                            attempt,
                            self.max_retries,
                            sleep_time,
                        )
                        time.sleep(sleep_time)
                        continue

                    self._handle_api_exception(e)

        return all_embeddings
