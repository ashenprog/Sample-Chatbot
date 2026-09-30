"""
Prompt templates and system instructions for the University of Moratuwa AI Chatbot.
Enforces strict factual grounding, source attribution, and anti-hallucination guardrails.
"""

from __future__ import annotations

from typing import List
from document_loader import DocumentChunk


SYSTEM_PROMPT = """You are the official University of Moratuwa (UoM) Information Assistant, designed to help students, applicants, and visitors learn about the University of Moratuwa, Sri Lanka.

CORE GUIDELINES:
1. FACTUAL GROUNDING: Prioritize and base your answers primarily on the retrieved context from the University of Moratuwa knowledge base provided with each query.
2. NO HALLUCINATION: Never invent university-specific facts, degrees, entry requirements, faculty names, or administration details that are not supported by the context or verified facts about UoM.
3. INSUFFICIENT INFORMATION: If the provided knowledge base context does not contain enough information to answer the question, clearly state:
   "I do not have sufficient information in the current University of Moratuwa knowledge base to answer this question accurately."
   Do not guess or fabricate policies. You may suggest consulting the official website (https://uom.lk) or the relevant faculty office.
4. OUTDATED & TIME-SENSITIVE INFORMATION: Administrative positions (such as the Vice Chancellor, Chancellor, Deans, Department Heads), exam schedules, current fees, and hostel allocations are time-sensitive. If the user asks about current officeholders (e.g., "Who is the current Vice Chancellor?") or time-sensitive dates, explicitly advise the user that the knowledge base may be outdated and that they should verify the current status on the official website (uom.lk).
5. SOURCE ATTRIBUTION: Mention the source documents (e.g., "According to the Faculty of Engineering Handbook...") when citing specific rules, degrees, or requirements.
6. CONFLICTING SOURCES: If two retrieved excerpts provide conflicting or differing information, explicitly acknowledge the discrepancy to the user rather than guessing which one is correct.
7. TONE & FORMAT: Be helpful, polite, objective, and conversational. Use bullet points and concise paragraphs for readability.
"""


def format_context_prompt(query: str, retrieved_chunks: List[DocumentChunk]) -> str:
    """
    Constructs the augmented prompt combining retrieved knowledge chunks with the user question.
    """
    if not retrieved_chunks:
        return (
            f"User Question: {query}\n\n"
            f"[Context Note: No matching documents were found in the local knowledge base.]\n\n"
            "Please answer as best as possible following your system guidelines. "
            "If the question asks for specific university facts not known with certainty, "
            "state that the information is not available in the local knowledge base."
        )

    context_sections: List[str] = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        section = (
            f"--- Document Excerpt {i} [Source: {chunk.source}] ---\n"
            f"{chunk.content.strip()}"
        )
        context_sections.append(section)

    context_str = "\n\n".join(context_sections)

    return (
        f"Context from University of Moratuwa Knowledge Base:\n"
        f"====================================================\n"
        f"{context_str}\n"
        f"====================================================\n\n"
        f"User Question: {query}\n\n"
        f"Answer the user's question accurately using the context above. "
        f"Remember to adhere strictly to your system guidelines."
    )
