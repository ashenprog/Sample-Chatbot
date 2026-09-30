"""
Unit tests for the University of Moratuwa AI Chatbot.
Covers configuration loading, document chunking, knowledge base CRUD,
retrieval, query contextualization, conversation history, command parsing,
and API key validation. All tests run offline without live API calls
and work out-of-the-box with both `pytest` and `python3 -m unittest`.
"""

from __future__ import annotations

import io
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import config
from chatbot import UoMChatbot
from document_loader import (
    Document,
    DocumentChunk,
    chunk_document,
    load_text_file,
    recursive_split_text,
)
from gemini_client import GeminiAPIKeyError, GeminiClient
from knowledge_base import KnowledgeBase, cosine_similarity
from prompts import format_context_prompt
from retriever import Retriever


class TestUoMChatbot(unittest.TestCase):
    """Test suite for University of Moratuwa AI Assistant."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.tmp_path = Path(self.test_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    # ==========================================
    # 1. Configuration & API Key Validation Tests
    # ==========================================
    def test_config_defaults(self):
        self.assertGreater(config.CHUNK_SIZE, 0)
        self.assertGreaterEqual(config.CHUNK_OVERLAP, 0)
        self.assertGreater(config.TOP_K_RESULTS, 0)
        self.assertGreater(config.MAX_HISTORY_MESSAGES, 0)
        self.assertIsInstance(config.GEMINI_MODEL, str)
        self.assertIsInstance(config.GEMINI_EMBEDDING_MODEL, str)

    def test_validate_api_key_empty(self):
        is_valid, msg = config.validate_api_key("")
        self.assertFalse(is_valid)
        self.assertIn("not set", msg.lower())

    def test_validate_api_key_placeholder(self):
        is_valid, msg = config.validate_api_key("your_api_key_here")
        self.assertFalse(is_valid)
        self.assertIn("placeholder", msg.lower())

    def test_validate_api_key_too_short(self):
        is_valid, msg = config.validate_api_key("abc123")
        self.assertFalse(is_valid)
        self.assertIn("too short", msg.lower())

    def test_validate_api_key_valid(self):
        is_valid, msg = config.validate_api_key("AIzaSyB_valid_mock_gemini_key_12345")
        self.assertTrue(is_valid)
        self.assertIn("valid", msg.lower())

    def test_gemini_client_missing_key_error(self):
        client = GeminiClient(api_key="")
        with self.assertRaises(GeminiAPIKeyError):
            client._get_client()

    # ==========================================
    # 2. Document Loading & Chunking Tests
    # ==========================================
    def test_load_text_file(self):
        test_file = self.tmp_path / "sample.md"
        test_file.write_text("# Test Title\nThis is a sample document for UoM.", encoding="utf-8")

        doc = load_text_file(test_file)
        self.assertEqual(doc.source, "sample.md")
        self.assertIn("This is a sample document for UoM.", doc.content)

    def test_recursive_split_text(self):
        sample_text = (
            "Paragraph 1 contains some introduction to University of Moratuwa.\n\n"
            "Paragraph 2 discusses the Engineering Faculty and departments.\n\n"
            "Paragraph 3 details the admissions criteria and GPA requirements."
        )
        chunks = recursive_split_text(sample_text, chunk_size=100, chunk_overlap=20)
        self.assertGreaterEqual(len(chunks), 3)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 140)

    def test_chunk_document(self):
        doc = Document(
            content="This is chunk one.\n\nThis is chunk two.\n\nThis is chunk three.",
            source="handbook.md",
        )
        chunks = chunk_document(doc, chunk_size=30, chunk_overlap=5)
        self.assertGreater(len(chunks), 0)
        self.assertEqual(chunks[0].source, "handbook.md")
        self.assertEqual(chunks[0].chunk_index, 0)
        self.assertTrue(chunks[0].chunk_id.startswith("handbook.md#chunk_"))

    def test_load_documents_from_sample_directory(self):
        from document_loader import load_documents_from_directory
        docs = load_documents_from_directory(config.DATA_DIR)
        self.assertGreaterEqual(len(docs), 5)
        sources = [d.source for d in docs]
        self.assertIn("01_faculties_and_departments.md", sources)
        self.assertIn("05_academic_calendar_and_contacts.txt", sources)


    # ==========================================
    # 3. Knowledge Base & Vector Search Tests
    # ==========================================
    def test_cosine_similarity(self):
        vec_a = [1.0, 0.0, 0.0]
        vec_b = [1.0, 0.0, 0.0]
        vec_c = [0.0, 1.0, 0.0]
        self.assertAlmostEqual(cosine_similarity(vec_a, vec_b), 1.0, places=3)
        self.assertAlmostEqual(cosine_similarity(vec_a, vec_c), 0.0, places=3)

    def test_knowledge_base_crud(self):
        db_file = self.tmp_path / "test_kb.db"
        kb = KnowledgeBase(db_file)
        self.assertEqual(kb.count_chunks(), 0)

        chunk1 = DocumentChunk(
            chunk_id="doc1#0001",
            content="Faculty of Engineering at University of Moratuwa.",
            source="doc1.md",
            chunk_index=0,
        )
        chunk2 = DocumentChunk(
            chunk_id="doc2#0001",
            content="Hostel facilities and student life at Katubedda.",
            source="doc2.md",
            chunk_index=0,
        )

        emb1 = [0.9, 0.1, 0.0]
        emb2 = [0.0, 0.1, 0.9]

        saved = kb.add_chunks([chunk1, chunk2], [emb1, emb2])
        self.assertEqual(saved, 2)
        self.assertEqual(kb.count_chunks(), 2)

        indexed = kb.get_indexed_files()
        self.assertEqual(len(indexed), 2)
        sources = [f["source"] for f in indexed]
        self.assertIn("doc1.md", sources)
        self.assertIn("doc2.md", sources)

        # Search with query close to emb1
        query_emb = [0.85, 0.15, 0.0]
        results = kb.search_by_embedding(query_emb, top_k=2, similarity_threshold=0.5)
        self.assertGreaterEqual(len(results), 1)
        best_chunk, best_score = results[0]
        self.assertEqual(best_chunk.source, "doc1.md")
        self.assertGreater(best_score, 0.9)

        # Clear KB
        kb.clear()
        self.assertEqual(kb.count_chunks(), 0)

    # ==========================================
    # 4. Retrieval & Query Contextualization Tests
    # ==========================================
    def test_query_contextualization(self):
        mock_kb = MagicMock()
        mock_client = MagicMock()
        retriever = Retriever(knowledge_base=mock_kb, gemini_client=mock_client)

        q1 = "Tell me about the Faculty of Engineering"
        self.assertEqual(retriever.contextualize_query(q1, history=[]), q1)

        history = [{"role": "user", "content": "Tell me about the Faculty of Engineering"}]
        follow_up = "What departments does it have?"
        contextualized = retriever.contextualize_query(follow_up, history=history)
        self.assertIn("Faculty of Engineering", contextualized)
        self.assertIn("departments", contextualized)

    def test_retriever_offline_mock(self):
        db_file = self.tmp_path / "retriever_test.db"
        kb = KnowledgeBase(db_file)
        chunk = DocumentChunk(
            chunk_id="it#0001",
            content="Faculty of Information Technology offers BSc in Artificial Intelligence.",
            source="it_faculty.md",
            chunk_index=0,
        )
        kb.add_chunks([chunk], [[1.0, 0.0, 0.0]])

        mock_client = MagicMock()
        mock_client.get_embedding.return_value = [1.0, 0.0, 0.0]

        retriever = Retriever(knowledge_base=kb, gemini_client=mock_client)
        results = retriever.retrieve("What courses does FIT offer?")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].source, "it_faculty.md")
        self.assertAlmostEqual(results[0].score, 1.0, places=2)

    # ==========================================
    # 5. Conversation History & Command Parsing
    # ==========================================
    def test_conversation_history_limit(self):
        orig_db = config.DB_PATH
        orig_max = config.MAX_HISTORY_MESSAGES
        try:
            config.DB_PATH = self.tmp_path / "mock.db"
            config.MAX_HISTORY_MESSAGES = 4

            chatbot = UoMChatbot()
            for i in range(6):
                chatbot.add_to_history("user", f"Question {i}")
                chatbot.add_to_history("assistant", f"Answer {i}")

            self.assertEqual(len(chatbot.history), 4)
            self.assertEqual(chatbot.history[-1]["content"], "Answer 5")
            self.assertEqual(chatbot.history[-2]["content"], "Question 5")
        finally:
            config.DB_PATH = orig_db
            config.MAX_HISTORY_MESSAGES = orig_max

    def test_chatbot_commands(self):
        orig_db = config.DB_PATH
        try:
            config.DB_PATH = self.tmp_path / "mock.db"
            chatbot = UoMChatbot()

            # Capture stdout
            captured = io.StringIO()
            old_stdout = sys.stdout
            sys.stdout = captured

            # /help
            self.assertTrue(chatbot.handle_command("/help"))
            self.assertIn("Available Commands:", captured.getvalue())

            # /clear
            chatbot.add_to_history("user", "Hello")
            self.assertTrue(chatbot.handle_command("/clear"))
            self.assertEqual(len(chatbot.history), 0)

            # /about
            self.assertTrue(chatbot.handle_command("/about"))
            self.assertIn("About University of Moratuwa", captured.getvalue())

            # unknown slash command
            self.assertTrue(chatbot.handle_command("/unknown_cmd"))
            self.assertIn("Unknown command", captured.getvalue())

            # normal question should return False to allow RAG handling
            self.assertFalse(chatbot.handle_command("What faculties exist at UoM?"))
        finally:
            sys.stdout = old_stdout
            config.DB_PATH = orig_db

    # ==========================================
    # 6. Prompt & Context Formatting Tests
    # ==========================================
    def test_format_context_prompt_with_chunks(self):
        chunks = [
            DocumentChunk(
                chunk_id="chunk_01",
                content="Katubedda campus is situated near Bolgoda Lake.",
                source="campus_guide.md",
                chunk_index=0,
            )
        ]
        prompt = format_context_prompt("Where is the campus located?", chunks)
        self.assertIn("campus_guide.md", prompt)
        self.assertIn("Bolgoda Lake", prompt)
        self.assertIn("Where is the campus located?", prompt)

    def test_format_context_prompt_empty_chunks(self):
        prompt = format_context_prompt("Random question", [])
        self.assertIn("No matching documents were found", prompt)
        self.assertIn("Random question", prompt)


if __name__ == "__main__":
    unittest.main()
