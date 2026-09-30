# University of Moratuwa AI Chatbot

A terminal-based AI assistant specifically designed to answer questions about the **University of Moratuwa (UoM)**, Sri Lanka. Built using Python, the official **Google GenAI SDK** (`google-genai`), and a local **Retrieval-Augmented Generation (RAG)** architecture.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture & How RAG Works](#architecture--how-rag-works)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation Guide](#installation-guide)
  - [1. Clone / Navigate to Project](#1-clone--navigate-to-project)
  - [2. Set Up Virtual Environment](#2-set-up-virtual-environment)
  - [3. Install Dependencies](#3-install-dependencies)
  - [4. Configure Environment Variables](#4-configure-environment-variables)
- [How to Obtain a Gemini API Key](#how-to-obtain-a-gemini-api-key)
- [Knowledge Base & Ingestion](#knowledge-base--ingestion)
  - [Sample Knowledge Base](#sample-knowledge-base)
  - [Adding Your Own Documents](#adding-your-own-documents)
  - [Building / Rebuilding the Knowledge Base](#building--rebuilding-the-knowledge-base)
- [Running the Chatbot](#running-the-chatbot)
- [Terminal Commands](#terminal-commands)
- [Accuracy & Anti-Hallucination Guardrails](#accuracy--anti-hallucination-guardrails)
- [Running Unit Tests](#running-unit-tests)
- [Troubleshooting Common Issues](#troubleshooting-common-issues)
- [Future Extensibility](#future-extensibility)

---

## Overview

The University of Moratuwa AI Chatbot operates entirely within your terminal—no web browser, desktop GUI, or mobile app is required. It allows prospective students, undergraduates, researchers, and visitors to ask conversational questions about:

- Faculties, departments, and degree courses
- Undergraduate admissions via UGC and aptitude tests
- GPA grading scales and degree classification criteria
- Hostels, boarding facilities, and distance allocation rules
- Student clubs, societies (Rotaract, IEEE, E-Club, Mora Hiking Club), and sports facilities
- Postgraduate degrees (MSc, MBA, MPhil, PhD)
- Academic calendar structure and official contact directories

---

## Key Features

- **Terminal-Native CLI**: Clean conversation loop with history recall, banner, and commands.
- **RAG Architecture**: Answers are grounded in local knowledge chunks rather than generic hallucinated text.
- **Multi-Format Ingestion**: Supports `.md`, `.txt`, `.pdf` (via `pypdf`), and `.docx` (via `python-docx`).
- **Local Vector Store**: Lightweight SQLite-backed storage storing embeddings and chunk metadata—zero cloud database dependencies or heavy C++ compilation headaches.
- **Conversational Memory**: Maintains multi-turn context (e.g. asking *"Tell me about the Faculty of Engineering"* followed by *"What departments does it have?"* correctly resolves *"it"*).
- **Source Citations**: Displays verifiable local document citations (e.g. `Sources: 01_faculties_and_departments.md`) with a `/sources` command to inspect chunk similarity and excerpts.
- **Anti-Hallucination & Outdated Information Warning**: A strict system prompt ensures the model acknowledges missing information and cautions users that administrative roles (e.g., Vice Chancellor, Deans) are subject to periodic change.

---

## Architecture & How RAG Works

The chatbot uses Retrieval-Augmented Generation (RAG) to ensure accuracy:

```text
 User Question: "What are the requirements for Faculty of Engineering?"
                      │
                      ▼
         [Query Contextualizer]
  Enriches pronoun references using recent conversation history
                      │
                      ▼
            [Gemini Embeddings]
  Generates query embedding vector via gemini-embedding-001
                      │
                      ▼
       [Local SQLite Vector Search]
  Calculates cosine similarity across stored chunk embeddings
                      │
                      ▼
          [Context Prompt Builder]
  Assembles top-k retrieved excerpts + strict system instructions
                      │
                      ▼
            [Google Gemini API]
  Generates grounded, conversational answer using gemini-2.5-flash
                      │
                      ▼
        [Terminal Output + Citations]
```

### Document Ingestion Flow (`python ingest.py`):
1. **Document Loading**: Scans `data/knowledge/` for `.md`, `.txt`, `.pdf`, `.docx`.
2. **Recursive Text Splitting**: Chunks documents into ~600-character segments with 100-character overlap while preserving paragraph and sentence boundaries.
3. **Embedding Generation**: Batches chunk texts to Gemini's embedding API.
4. **SQLite Persistence**: Stores chunks, metadata, and embedding vectors into `data/knowledge_base.db`.

---

## Project Structure

```text
ChatBot/
├── chatbot.py               # Main interactive terminal loop & slash commands
├── ingest.py                # CLI script to build/rebuild the local vector knowledge base
├── config.py                # Environment variable loader & settings validation
├── gemini_client.py         # Official Google GenAI SDK wrapper (generation, embeddings, retries)
├── retriever.py             # Semantic vector retriever & query contextualizer
├── knowledge_base.py        # SQLite vector storage, metadata manager, and cosine similarity
├── document_loader.py       # Multi-format document parser (.txt, .md, .pdf, .docx) & chunker
├── prompts.py               # System prompts, anti-hallucination rules, and context templates
├── requirements.txt         # Project dependencies
├── .env.example             # Template for API keys and configuration
├── .gitignore               # Ensures secrets and database files are never committed
├── README.md                # Documentation and setup guide
│
├── data/
│   ├── knowledge/           # Raw UoM source documents (.md, .txt, .pdf, .docx)
│   │   ├── 01_faculties_and_departments.md
│   │   ├── 02_undergraduate_admissions_and_gpa.md
│   │   ├── 03_facilities_hostels_and_student_life.md
│   │   ├── 04_postgraduate_and_administration.md
│   │   └── 05_academic_calendar_and_contacts.txt
│   └── knowledge_base.db    # SQLite database (generated by ingest.py)
│
└── tests/
    ├── __init__.py
    └── test_chatbot.py      # Unit tests (18 tests, runs 100% offline)
```

---

## Prerequisites

- **Python**: Version 3.9, 3.10, 3.11, or 3.12+
- **Google Gemini API Key**: Free tier or paid key from Google AI Studio.
- **Operating System**: macOS, Linux, or Windows (WSL or Command Prompt / PowerShell).

---

## Installation Guide

### 1. Clone / Navigate to Project

```bash
cd /path/to/ChatBot
```

### 2. Set Up Virtual Environment

On **macOS / Linux**:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

On **Windows (PowerShell)**:
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

On **Windows (Command Prompt)**:
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

*(Note: Dependencies include `google-genai`, `python-dotenv`, `numpy`, `pypdf`, `python-docx`, and `pytest`.)*

### 4. Configure Environment Variables

Create your local `.env` file from `.env.example`:

On **macOS / Linux**:
```bash
cp .env.example .env
```

On **Windows**:
```cmd
copy .env.example .env
```

Open `.env` in any text editor and paste your Gemini API key:

```env
GEMINI_API_KEY=AIzaSy...your_actual_api_key_here
GEMINI_MODEL=gemini-2.5-flash
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
MAX_HISTORY_MESSAGES=20
```

> **Security Note:** Never commit your `.env` file or share your API key. `.env` is already included in `.gitignore`.

---

## How to Obtain a Gemini API Key

1. Go to [Google AI Studio](https://aistudio.google.com/).
2. Sign in with your Google account.
3. Click on **"Get API key"** in the sidebar.
4. Click **"Create API key"** (select a new or existing Google Cloud project).
5. Copy the generated key and paste it as `GEMINI_API_KEY` in your `.env` file.

---

## Knowledge Base & Ingestion

### Sample Knowledge Base
The repository includes a curated set of authentic documents under `data/knowledge/`:
- `01_faculties_and_departments.md`: All 5 undergraduate faculties, graduate studies, and departments.
- `02_undergraduate_admissions_and_gpa.md`: UGC Z-Score criteria, aptitude tests, and 4.00 GPA calculation.
- `03_facilities_hostels_and_student_life.md`: Hostel distance rules, library, sports complex, and student societies.
- `04_postgraduate_and_administration.md`: MSc, MBA, PhD programs, governance structure, and notice regarding officeholders.
- `05_academic_calendar_and_contacts.txt`: Semester structure, telephone directory, mailing address, and transit routes.

### Adding Your Own Documents
To add new documents (handbooks, prospectuses, syllabi):
1. Place any `.txt`, `.md`, `.pdf`, or `.docx` file into `data/knowledge/`.
2. Run `python ingest.py` to index the new files.

### Building / Rebuilding the Knowledge Base

Run the ingestion tool:

```bash
python ingest.py
```

To wipe the existing database and do a fresh re-index:

```bash
python ingest.py --rebuild
```

Example output:
```text
============================================================
 University of Moratuwa - Knowledge Base Builder
============================================================

[*] Scanning documents from: /path/to/ChatBot/data/knowledge
[+] Found 5 document(s):
    - 01_faculties_and_departments.md (4530 characters)
    - 02_undergraduate_admissions_and_gpa.md (3669 characters)
    - 03_facilities_hostels_and_student_life.md (3814 characters)
    - 04_postgraduate_and_administration.md (3802 characters)
    - 05_academic_calendar_and_contacts.txt (2233 characters)

[*] Chunking documents (size=600, overlap=100)...
[+] Generated 43 total text chunk(s).

[*] Generating embeddings using Gemini model: gemini-embedding-001...
[+] Successfully generated 43 embeddings in 1.45s.

[*] Storing chunks and embeddings into SQLite: data/knowledge_base.db...

============================================================
 INGESTION COMPLETE
============================================================
 Indexed files : 5
 New chunks    : 43
 Total in DB   : 43
 Database file : /path/to/ChatBot/data/knowledge_base.db
```

---

## Running the Chatbot

Start the interactive terminal chatbot:

```bash
python chatbot.py
```

### Example Session

```text
==================================================
 University of Moratuwa AI Assistant
==================================================
Ask me anything about the University of Moratuwa.

Type '/help' for commands.
Type '/exit' or 'exit' to leave.

[*] Knowledge Base loaded: 43 chunk(s) indexed.

You: What faculties are available at University of Moratuwa?

Assistant:
The University of Moratuwa has five primary undergraduate degree-awarding faculties, along with a dedicated graduate faculty:

1. Faculty of Engineering: The oldest and largest engineering faculty in Sri Lanka, offering Washington Accord-accredited programs across departments such as Computer Science & Engineering, Electronic & Telecommunication Engineering, Civil, Electrical, and Mechanical Engineering.
2. Faculty of Information Technology (FIT): The first and only dedicated IT faculty in Sri Lanka's state university system, offering degrees in IT, IT & Management, and Artificial Intelligence.
3. Faculty of Architecture: Renowned for built-environment education, offering RIBA- and RICS-accredited degrees in Architecture, Quantity Surveying, Facilities Management, Town & Country Planning, and Integrated Design.
4. Faculty of Business: Specializing in technology-driven management, offering honours degrees in Business Analytics, Financial Analytics, and Business Process Management.
5. Faculty of Medicine: The newest faculty (established in 2020), offering the MBBS degree program.
6. Faculty of Graduate Studies (FGS): Coordinates postgraduate research and degrees (MSc, MPhil, PhD) across faculties.

Sources:
- 01_faculties_and_departments.md

You: What about the GPA requirement?

Assistant:
According to the University of Moratuwa's credit and grading system, academic performance is evaluated on a 4.00 Grade Point Average (GPA) scale:

- A / A+ = 4.00 Grade Points (Excellent / Outstanding)
- A- = 3.70
- B+ = 3.30
- B = 3.00 (Good)
- B- = 2.70
- C+ = 2.30
- C = 2.00 (Satisfactory / Minimum pass grade)
- F = 0.00 (Fail)

Graduation & Honours Requirements:
- Minimum GPA to Pass / Graduate: An Overall GPA of at least 2.00 and no unresolved 'F' grades.
- First Class Honours: Overall GPA of 3.70 or higher.
- Second Class Honours (Upper Division): Overall GPA of 3.30 – 3.69.
- Second Class Honours (Lower Division): Overall GPA of 3.00 – 3.29.

Sources:
- 02_undergraduate_admissions_and_gpa.md

You: /sources

Sources used for the most recent answer:
  1. 02_undergraduate_admissions_and_gpa.md (similarity score: 0.812)
     Excerpt: "## Credit and Grading System (GPA) The University of Moratuwa operates on a modular credit-based semester system..."

You: /exit

Goodbye! Thank you for using University of Moratuwa AI Assistant.
```

---

## Terminal Commands

The chatbot supports slash commands that are handled directly by the CLI and never sent to Gemini:

| Command | Description |
|---|---|
| `/help` | Displays available slash commands and usage tips. |
| `/sources` | Shows document sources and similarity scores used for the most recent answer. |
| `/history` | Displays turn-by-turn dialogue history for the current session. |
| `/clear` | Clears conversation memory. |
| `/about` | Displays system info: model, embedding model, chunk count, and database path. |
| `/exit` | Gracefully terminates the session (aliases: `/quit`, `exit`, `quit`). |

---

## Accuracy & Anti-Hallucination Guardrails

The chatbot uses strict prompting and RAG grounding:
1. **Knowledge Base Grounding**: Answers are derived from indexed documents.
2. **Missing Information Handling**: If a topic is not covered in the local documents, the assistant explicitly states:
   > *"I do not have sufficient information in the current University of Moratuwa knowledge base to answer this question accurately."*
   It directs users to official university channels rather than guessing.
3. **Outdated Information Warnings**: For questions about university officeholders (e.g. *"Who is the current Vice Chancellor?"*) or changing timetables, the system explicitly advises verifying live records at [uom.lk](https://uom.lk).
4. **Source Attribution**: Retrieved sources are listed automatically at the conclusion of each response.

---

## Running Unit Tests

The test suite runs 100% offline and verifies configuration loading, chunking, SQLite storage, semantic cosine retrieval, query contextualization, history truncation, command parsing, and missing key handling.

Run with Python's built-in `unittest`:
```bash
python3 -m unittest discover tests
```

Or run with `pytest`:
```bash
pytest tests/
```

Expected output:
```text
Ran 18 tests in 0.019s

OK
```

---

## Troubleshooting Common Issues

### 1. `GEMINI_API_KEY is not set`
- Ensure you created a `.env` file in the project root (`cp .env.example .env`).
- Verify that `GEMINI_API_KEY` contains your actual key, not `your_api_key_here`.

### 2. `The local knowledge base is currently empty`
- You need to index documents before asking questions:
  ```bash
  python ingest.py
  ```

### 3. `ModuleNotFoundError: No module named 'google.genai'`
- You may have installed packages in a different Python environment or forgot to activate your virtual environment:
  ```bash
  source .venv/bin/activate   # or .venv\Scripts\activate on Windows
  pip install -r requirements.txt
  ```

### 4. `Cannot read PDF / Cannot read DOCX`
- PDF reading requires `pypdf` (`pip install pypdf`).
- Word document reading requires `python-docx` (`pip install python-docx`).
- Both are included in `requirements.txt`.

### 5. `Rate limit or quota exceeded (429)`
- The Gemini client includes automatic exponential backoff retries. If free-tier quotas are temporarily exhausted, wait 30 seconds and retry.

---

## Future Extensibility

The architecture is designed to support future enhancements:
- **Web Search Integration**: `retriever.py` can be extended with a live retrieval layer using official UoM feeds or Google Search grounding.
- **Advanced RAG**: Easy drop-in replacement with hybrid search (BM25 + vector embeddings) or reranking models.
- **Custom Document Ingestion**: Additional faculty handbooks and research publications can simply be dropped into `data/knowledge/` and ingested with `python ingest.py`.
# Sample-Chatbot
