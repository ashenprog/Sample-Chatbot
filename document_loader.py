"""
Document loader and chunking module for the UoM Knowledge Base.
Supports .txt, .md, .pdf, and .docx formats with recursive text chunking.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Represents an ingested raw document."""
    content: str
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentChunk:
    """Represents a chunk extracted from a document for embedding and retrieval."""
    chunk_id: str
    content: str
    source: str
    chunk_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)


def load_text_file(file_path: Path) -> Document:
    """Loads a plain text or markdown file with encoding fallback."""
    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
    for enc in encodings:
        try:
            with open(file_path, "r", encoding=enc) as f:
                content = f.read().strip()
                return Document(
                    content=content,
                    source=file_path.name,
                    metadata={"path": str(file_path), "format": file_path.suffix.lower()},
                )
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode text file {file_path.name} with standard encodings.")


def load_pdf_file(file_path: Path) -> Document:
    """Loads text from a PDF file using pypdf if available."""
    try:
        import pypdf
    except ImportError:
        raise ImportError(
            f"Cannot read PDF '{file_path.name}'. Please install pypdf: pip install pypdf"
        )

    try:
        reader = pypdf.PdfReader(str(file_path))
        text_parts: List[str] = []
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text.strip())

        content = "\n\n".join(text_parts).strip()
        if not content:
            logger.warning("PDF '%s' produced empty text (may be scanned images).", file_path.name)

        return Document(
            content=content,
            source=file_path.name,
            metadata={
                "path": str(file_path),
                "format": ".pdf",
                "page_count": len(reader.pages),
            },
        )
    except Exception as e:
        raise RuntimeError(f"Error reading PDF '{file_path.name}': {e}") from e


def load_docx_file(file_path: Path) -> Document:
    """Loads text from a DOCX file using python-docx if available."""
    try:
        import docx
    except ImportError:
        raise ImportError(
            f"Cannot read DOCX '{file_path.name}'. Please install python-docx: pip install python-docx"
        )

    try:
        doc = docx.Document(str(file_path))
        text_parts: List[str] = []

        # Extract paragraphs
        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                text_parts.append(text)

        # Extract tables
        for table in doc.tables:
            for row in table.rows:
                row_cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if row_cells:
                    text_parts.append(" | ".join(row_cells))

        content = "\n\n".join(text_parts).strip()
        return Document(
            content=content,
            source=file_path.name,
            metadata={"path": str(file_path), "format": ".docx"},
        )
    except Exception as e:
        raise RuntimeError(f"Error reading DOCX '{file_path.name}': {e}") from e


def load_file(file_path: Path) -> Document:
    """Loads a single document based on its file extension."""
    suffix = file_path.suffix.lower()
    if suffix in [".txt", ".md"]:
        return load_text_file(file_path)
    elif suffix == ".pdf":
        return load_pdf_file(file_path)
    elif suffix == ".docx":
        return load_docx_file(file_path)
    else:
        raise ValueError(f"Unsupported file format '{suffix}' for file '{file_path.name}'.")


def load_documents_from_directory(dir_path: Path) -> List[Document]:
    """
    Recursively scans and loads all supported documents (.txt, .md, .pdf, .docx)
    from the specified directory.
    """
    if not dir_path.exists():
        logger.warning("Directory '%s' does not exist.", dir_path)
        return []

    supported_suffixes = {".txt", ".md", ".pdf", ".docx"}
    documents: List[Document] = []

    for file_path in sorted(dir_path.glob("**/*")):
        if file_path.is_file() and file_path.suffix.lower() in supported_suffixes:
            if file_path.name.startswith("."):
                continue  # skip hidden/temporary files
            try:
                doc = load_file(file_path)
                if doc.content:
                    documents.append(doc)
                else:
                    logger.warning("Skipping empty document: %s", file_path.name)
            except Exception as e:
                logger.error("Failed to load '%s': %s", file_path.name, e)

    return documents


def recursive_split_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Splits text recursively using delimiters (paragraphs, lines, sentences, words)
    to respect natural language boundaries while enforcing chunk size and overlap.
    """
    if not text.strip():
        return []

    if len(text) <= chunk_size:
        return [text.strip()]

    # Ordered separators from most semantic to least
    separators = ["\n\n", "\n", ". ", "; ", ", ", " "]

    def _split(text_to_split: str, sep_index: int) -> List[str]:
        if len(text_to_split) <= chunk_size:
            return [text_to_split.strip()] if text_to_split.strip() else []

        if sep_index >= len(separators):
            # Hard character split as last resort
            chunks = []
            start = 0
            while start < len(text_to_split):
                end = min(start + chunk_size, len(text_to_split))
                chunk = text_to_split[start:end].strip()
                if chunk:
                    chunks.append(chunk)
                start += max(1, chunk_size - chunk_overlap)
            return chunks

        sep = separators[sep_index]
        splits = text_to_split.split(sep)
        chunks: List[str] = []
        current_chunk: List[str] = []
        current_len = 0

        for part in splits:
            part_len = len(part) + (len(sep) if current_chunk else 0)
            if current_len + part_len <= chunk_size:
                current_chunk.append(part)
                current_len += part_len
            else:
                if current_chunk:
                    merged = sep.join(current_chunk).strip()
                    if merged:
                        chunks.append(merged)
                    # Prepare overlap from the tail of current_chunk
                    overlap_acc: List[str] = []
                    overlap_len = 0
                    for prev_part in reversed(current_chunk):
                        if overlap_len + len(prev_part) <= chunk_overlap:
                            overlap_acc.insert(0, prev_part)
                            overlap_len += len(prev_part) + len(sep)
                        else:
                            break
                    current_chunk = overlap_acc
                    current_len = sum(len(p) for p in current_chunk) + len(sep) * max(0, len(current_chunk) - 1)

                if len(part) > chunk_size:
                    # If this single part exceeds chunk_size, split with next separator
                    sub_chunks = _split(part, sep_index + 1)
                    chunks.extend(sub_chunks)
                    current_chunk = []
                    current_len = 0
                else:
                    current_chunk.append(part)
                    current_len += len(part)

        if current_chunk:
            merged = sep.join(current_chunk).strip()
            if merged:
                chunks.append(merged)

        return chunks

    return _split(text, 0)


def chunk_document(doc: Document, chunk_size: int = 600, chunk_overlap: int = 100) -> List[DocumentChunk]:
    """Splits a document into indexed chunks with unique IDs."""
    text_chunks = recursive_split_text(doc.content, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    document_chunks: List[DocumentChunk] = []

    for index, chunk_text in enumerate(text_chunks):
        chunk_id = f"{doc.source}#chunk_{index:04d}"
        document_chunks.append(
            DocumentChunk(
                chunk_id=chunk_id,
                content=chunk_text,
                source=doc.source,
                chunk_index=index,
                metadata=doc.metadata,
            )
        )

    return document_chunks
