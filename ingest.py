"""
Knowledge Base Ingestion CLI for University of Moratuwa AI Chatbot.
Scans data/knowledge/, parses documents (.txt, .md, .pdf, .docx),
generates vector embeddings with Gemini, and saves them to the local SQLite database.

Usage:
    python ingest.py
    python ingest.py --rebuild
    python ingest.py --data-dir path/to/docs
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

import config
from document_loader import DocumentChunk, chunk_document, load_documents_from_directory
from gemini_client import GeminiAPIKeyError, GeminiClient, GeminiClientError
from knowledge_base import KnowledgeBase


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest University of Moratuwa documents into local vector knowledge base."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(config.DATA_DIR),
        help=f"Directory containing source documents (default: {config.DATA_DIR})",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=str(config.DB_PATH),
        help=f"Path to SQLite database file (default: {config.DB_PATH})",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Clear existing knowledge base before ingesting new documents.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=config.CHUNK_SIZE,
        help=f"Character size per chunk (default: {config.CHUNK_SIZE})",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=config.CHUNK_OVERLAP,
        help=f"Overlap size between chunks (default: {config.CHUNK_OVERLAP})",
    )
    return parser.parse_args()


def run_ingestion(
    data_dir: Path,
    db_path: Path,
    rebuild: bool = False,
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
) -> bool:
    print("=" * 60)
    print(" University of Moratuwa - Knowledge Base Builder")
    print("=" * 60)

    # 1. Check API Key
    is_valid, msg = config.validate_api_key()
    if not is_valid:
        print(f"\n[!] Configuration Error: {msg}")
        print("\nPlease follow these steps:")
        print("  1. Copy .env.example to .env:  cp .env.example .env")
        print("  2. Open .env and add your valid GEMINI_API_KEY")
        print("  3. Run 'python ingest.py' again.\n")
        return False

    # 2. Check source documents
    data_dir = Path(data_dir)
    if not data_dir.exists():
        print(f"\n[!] Creating data directory: {data_dir}")
        data_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[*] Scanning documents from: {data_dir}")
    documents = load_documents_from_directory(data_dir)
    if not documents:
        print(f"\n[!] No supported documents (.txt, .md, .pdf, .docx) found in {data_dir}.")
        print("    Please place University of Moratuwa documents inside that folder and rerun.")
        return False

    print(f"[+] Found {len(documents)} document(s):")
    for doc in documents:
        print(f"    - {doc.source} ({len(doc.content)} characters)")

    # 3. Chunk documents
    print(f"\n[*] Chunking documents (size={chunk_size}, overlap={chunk_overlap})...")
    all_chunks: List[DocumentChunk] = []
    for doc in documents:
        chunks = chunk_document(doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        all_chunks.extend(chunks)

    print(f"[+] Generated {len(all_chunks)} total text chunk(s).")
    if not all_chunks:
        print("[!] No text could be extracted to chunk. Aborting.")
        return False

    # 4. Generate embeddings with Gemini
    print(f"\n[*] Generating embeddings using Gemini model: {config.GEMINI_EMBEDDING_MODEL}...")
    try:
        client = GeminiClient()
        texts_to_embed = [chunk.content for chunk in all_chunks]
        start_time = time.time()
        embeddings = client.get_embeddings_batch(texts_to_embed)
        elapsed = time.time() - start_time
        print(f"[+] Successfully generated {len(embeddings)} embeddings in {elapsed:.2f}s.")
    except GeminiAPIKeyError as e:
        print(f"\n[!] Authentication Error: {e}")
        return False
    except GeminiClientError as e:
        print(f"\n[!] Embedding Generation Failed: {e}")
        return False
    except Exception as e:
        print(f"\n[!] Unexpected Error during embedding generation: {e}")
        return False

    # 5. Store in local SQLite Database
    print(f"\n[*] Storing chunks and embeddings into SQLite: {db_path}...")
    kb = KnowledgeBase(db_path)
    if rebuild:
        print("    [--rebuild flag set] Clearing existing knowledge base data...")
        kb.clear()

    saved_count = kb.add_chunks(all_chunks, embeddings)
    total_stored = kb.count_chunks()

    print("\n" + "=" * 60)
    print(" INGESTION COMPLETE")
    print("=" * 60)
    print(f" Indexed files : {len(documents)}")
    print(f" New chunks    : {saved_count}")
    print(f" Total in DB   : {total_stored}")
    print(f" Database file : {db_path}")
    print("\nYou can now start the chatbot by running:")
    print("    python chatbot.py\n")
    return True


def main() -> None:
    args = parse_args()
    success = run_ingestion(
        data_dir=Path(args.data_dir),
        db_path=Path(args.db_path),
        rebuild=args.rebuild,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
