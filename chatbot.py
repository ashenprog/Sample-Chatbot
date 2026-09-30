"""
Terminal-based AI Chatbot for University of Moratuwa (UoM), Sri Lanka.
Powered by Google Gemini API and a local RAG knowledge base.

Usage:
    python chatbot.py
"""

from __future__ import annotations

import sys
from typing import Dict, List, Optional

# Enable readline support for arrow keys and terminal input history if available
try:
    import readline  # noqa: F401
except ImportError:
    pass

import config
from gemini_client import (
    GeminiAPIKeyError,
    GeminiClient,
    GeminiClientError,
    GeminiNetworkError,
    GeminiRateLimitError,
)
from knowledge_base import KnowledgeBase
from prompts import SYSTEM_PROMPT, format_context_prompt
from retriever import RetrievalResult, Retriever

CHATBOT_VERSION = "1.0.0"


class UoMChatbot:
    """Core chatbot controller managing conversation loop, commands, RAG, and memory."""

    def __init__(self):
        self.kb = KnowledgeBase(config.DB_PATH)
        self.gemini_client = GeminiClient()
        self.retriever = Retriever(
            knowledge_base=self.kb,
            gemini_client=self.gemini_client,
            top_k=config.TOP_K_RESULTS,
            similarity_threshold=config.SIMILARITY_THRESHOLD,
        )
        self.history: List[Dict[str, str]] = []
        self.last_retrieved_sources: List[RetrievalResult] = []

    def print_banner(self) -> None:
        """Displays the startup terminal welcome banner."""
        banner = """
==================================================
 University of Moratuwa AI Assistant
==================================================
Ask me anything about the University of Moratuwa.

Type '/help' for commands.
Type '/exit' or 'exit' to leave.
"""
        print(banner)

    def check_environment(self) -> bool:
        """Checks API key and knowledge base status before starting."""
        is_valid, msg = config.validate_api_key()
        if not is_valid:
            print(f"[!] Configuration Notice: {msg}")
            print("    Please set a valid GEMINI_API_KEY in your .env file.")
            print("    The chatbot requires an API key to communicate with Gemini.\n")
            return False

        chunk_count = self.kb.count_chunks()
        if chunk_count == 0:
            print("[!] Notice: The local knowledge base is currently empty.")
            print("    To enable RAG search on UoM documents, run:")
            print("        python ingest.py")
            print("    The assistant will attempt to answer using general knowledge until ingested.\n")
        else:
            print(f"[*] Knowledge Base loaded: {chunk_count} chunk(s) indexed.\n")

        return True

    def handle_command(self, user_input: str) -> bool:
        """
        Executes terminal slash commands.
        Returns True if the input was a recognized command, False otherwise.
        """
        cmd = user_input.strip().lower()

        # Exit commands
        if cmd in ["/exit", "/quit", "exit", "quit"]:
            print("\nGoodbye! Thank you for using University of Moratuwa AI Assistant.\n")
            sys.exit(0)

        elif cmd == "/help":
            self._cmd_help()
            return True

        elif cmd == "/sources":
            self._cmd_sources()
            return True

        elif cmd == "/clear":
            self._cmd_clear()
            return True

        elif cmd == "/history":
            self._cmd_history()
            return True

        elif cmd == "/about":
            self._cmd_about()
            return True

        elif cmd.startswith("/"):
            print(f"Unknown command '{user_input}'. Type '/help' to view available commands.\n")
            return True

        return False

    def _cmd_help(self) -> None:
        print("\nAvailable Commands:")
        print("  /help     - Show this help message")
        print("  /sources  - View the document sources used for the most recent answer")
        print("  /history  - View recent conversation messages in this session")
        print("  /clear    - Clear conversation history memory")
        print("  /about    - View chatbot architecture and system information")
        print("  /exit     - Exit the application (or type 'exit')\n")

    def _cmd_sources(self) -> None:
        if not self.last_retrieved_sources:
            print("\nNo local knowledge base sources were retrieved for the last response.\n")
            return

        print("\nSources used for the most recent answer:")
        for i, res in enumerate(self.last_retrieved_sources, 1):
            print(f"  {i}. {res.source} (similarity score: {res.score:.3f})")
            # Show a brief excerpt
            snippet = res.chunk.content.replace("\n", " ")[:120]
            print(f"     Excerpt: \"{snippet}...\"")
        print()

    def _cmd_clear(self) -> None:
        self.history.clear()
        self.last_retrieved_sources.clear()
        print("\n[+] Conversation history has been cleared.\n")

    def _cmd_history(self) -> None:
        if not self.history:
            print("\nConversation history is currently empty.\n")
            return

        print(f"\n--- Current Session History ({len(self.history)} messages) ---")
        for turn in self.history:
            speaker = "You" if turn["role"] == "user" else "Assistant"
            print(f"\n{speaker}:")
            print(f"  {turn['content']}")
        print("\n--------------------------------------------------------------\n")

    def _cmd_about(self) -> None:
        chunk_count = self.kb.count_chunks()
        indexed_files = self.kb.get_indexed_files()
        print("\nAbout University of Moratuwa AI Assistant:")
        print(f"  Version             : {CHATBOT_VERSION}")
        print(f"  Gemini Model        : {config.GEMINI_MODEL}")
        print(f"  Embedding Model     : {config.GEMINI_EMBEDDING_MODEL}")
        print(f"  Knowledge Base DB   : {config.DB_PATH}")
        print(f"  Indexed Chunks      : {chunk_count}")
        print(f"  Memory Limit        : {config.MAX_HISTORY_MESSAGES} messages")
        if indexed_files:
            print("  Indexed Documents   :")
            for f in indexed_files:
                print(f"    - {f['source']} ({f['chunk_count']} chunks)")
        print()

    def add_to_history(self, role: str, content: str) -> None:
        """Appends a turn to conversation memory and enforces MAX_HISTORY_MESSAGES limit."""
        self.history.append({"role": role, "content": content})
        if len(self.history) > config.MAX_HISTORY_MESSAGES:
            # Drop oldest turns while keeping memory within limit
            self.history = self.history[-config.MAX_HISTORY_MESSAGES:]

    def ask(self, question: str) -> None:
        """Processes a user question via RAG and streams/prints the response."""
        # 1. Retrieve relevant knowledge base chunks
        retrieved_results = self.retriever.retrieve(question, history=self.history)
        self.last_retrieved_sources = retrieved_results
        retrieved_chunks = [res.chunk for res in retrieved_results]

        # 2. Format context prompt
        context_prompt = format_context_prompt(question, retrieved_chunks)

        # 3. Call Gemini
        try:
            print("\nAssistant:")
            answer = self.gemini_client.generate_response(
                prompt=context_prompt,
                system_instruction=SYSTEM_PROMPT,
                history=self.history,
            )
            print(f"{answer}\n")

            # 4. Display source citations if local sources were retrieved
            if retrieved_chunks:
                unique_sources = sorted(list({res.source for res in retrieved_results}))
                print("Sources:")
                for src in unique_sources:
                    print(f"- {src}")
                print()

            # 5. Save to conversation history
            self.add_to_history("user", question)
            self.add_to_history("assistant", answer)

        except GeminiAPIKeyError as e:
            print(f"Error: {e}\n")
        except GeminiRateLimitError as e:
            print(f"Error: {e}\n")
        except GeminiNetworkError as e:
            print(f"Error: {e}\n")
        except GeminiClientError as e:
            print(f"Error: Unable to generate response. {e}\n")
        except Exception as e:
            print(f"Error: An unexpected issue occurred. Please try again. ({e})\n")

    def run(self) -> None:
        """Main interactive terminal loop."""
        self.print_banner()

        if not self.check_environment():
            # Still allow user to inspect /help or /about, but warn them
            print("Chatbot running in limited mode. Please configure your .env file to enable full chat.\n")

        while True:
            try:
                user_input = input("You: ").strip()

                if not user_input:
                    continue

                # Check if user entered a command
                if self.handle_command(user_input):
                    continue

                # Process natural language question
                self.ask(user_input)

            except (KeyboardInterrupt, EOFError):
                print("\n\nSession terminated. Goodbye!\n")
                break


def main() -> None:
    chatbot = UoMChatbot()
    chatbot.run()


if __name__ == "__main__":
    main()
