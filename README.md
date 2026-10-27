# QueryFolio — Advanced Document Research & Intelligence RAG

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-FF4F00)](https://www.trychroma.com/)
[![Hugging Face](https://img.shields.io/badge/Hugging_Face-Sentence_Transformers-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![Mermaid.js](https://img.shields.io/badge/Mermaid.js-Interactive_Flowcharts-FF3670)](https://mermaid.js.org/)

**QueryFolio** is a full-stack, enterprise-grade Document Research and Intelligence platform. Built with **FastAPI**, **ChromaDB**, **BM25 Sparse Retrieval**, **Cross-Encoder Neural Reranking**, **Google Gemini / Multi-Provider LLMs**, and an aesthetic **Warm Beige & Burgundy Vanilla UI**, QueryFolio transforms static PDFs, scanned documents, and research papers into interactive, grounded knowledge with live token streaming and interactive architecture flowcharts.

---

## 🏗️ Architecture Pipeline

```text
                               ┌────────────────────────────────────────────────────────┐
                               │                    PDF DOCUMENT                        │
                               │        (Printed Text / Scanned / Handwritten)          │
                               └──────────────────────────┬─────────────────────────────┘
                                                          │
                                         ┌────────────────┴────────────────┐
                                         │                                 │
                         [Standard Text Available]              [Scanned / Image Pages]
                                         │                                 │
                                         ▼                                 ▼
                               ┌───────────────────┐             ┌───────────────────┐
                               │  PyPDF Extractor  │             │  Vision OCR Engine│
                               │  (Page Metadata)  │             │(PyMuPDF + Vision) │
                               └─────────┬─────────┘             └─────────┬─────────┘
                                         │                                 │
                                         └────────────────┬────────────────┘
                                                          ▼
                                        ┌───────────────────────────────────┐
                                        │  Recursive Character Text Chunking│
                                        │  (Chunk: 1000, Overlap: 150)      │
                                        └─────────────────┬─────────────────┘
                                                          ▼
                                ┌─────────────────────────┴─────────────────────────┐
                                │                                                   │
                                ▼                                                   ▼
                     ┌─────────────────────┐                             ┌─────────────────────┐
                     │ Dense Embeddings    │                             │ Sparse Indexing     │
                     │ (all-MiniLM-L6-v2)  │                             │ (BM25 Tokenizer)    │
                     └──────────┬──────────┘                             └──────────┬──────────┘
                                │                                                   │
                                ▼                                                   │
                     ┌─────────────────────┐                                        │
                     │ ChromaDB Collection │                                        │
                     │ (Per-Doc Isolation) │                                        │
                     └──────────┬──────────┘                                        │
                                │                                                   │
                                └─────────────────────────┬─────────────────────────┘
                                                          ▼
                                        ┌───────────────────────────────────┐
                                        │    Representative Sampling        │
                                        │  5–8 Topic-Specific Questions     │
                                        └─────────────────┬─────────────────┘
                                                          │
   User Query ────────────────────────────────────────────┼─────────────────────────────────┐
                                                          ▼                                 │
                                        ┌───────────────────────────────────┐               │
                                        │       Dual Hybrid Retrieval       │               │
                                        │    Dense Vector + BM25 Search     │               │
                                        └─────────────────┬─────────────────┘               │
                                                          ▼                                 │
                                        ┌───────────────────────────────────┐               │
                                        │      Reciprocal Rank Fusion       │               │
                                        │     & Candidate Deduplication     │               │
                                        └─────────────────┬─────────────────┘               │
                                                          ▼                                 │
                                        ┌───────────────────────────────────┐               │
                                        │   Cross-Encoder Neural Rerank     │               │
                                        │  (ms-marco-MiniLM-L-6-v2)         │               │
                                        └─────────────────┬─────────────────┘               │
                                                          ▼                                 │
                                        ┌───────────────────────────────────┐               │
                                        │       Grounded RAG Prompt         │               │
                                        │    (Strict Context & Refusal)     │               │
                                        └─────────────────┬─────────────────┘               │
                                                          ▼                                 │
                                        ┌───────────────────────────────────┐               │
                                        │    Multi-Provider LLM Factory     │               │
                                        │ (Gemini 2.5 Flash / Gemma / etc.) │               │
                                        └─────────────────┬─────────────────┘               │
                                                          ▼                                 │
                                        ┌───────────────────────────────────┐               │
                                        │   Real-Time SSE Token Stream      │               │
                                        │  + Verified Source Citations      │               │
                                        │  + Interactive Mermaid Flowcharts │◄──────────────┘
                                        └───────────────────────────────────┘
```

---

## 🚀 Key Features

### 1. 🔍 Dual Hybrid RAG (Dense Vector + BM25 Sparse Search)
* **Dense Semantic Embeddings**: Utilizes `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional vectors) to capture semantic context.
* **Sparse BM25 Keyword Matching**: Integrates `rank-bm25` for exact phrase, technical acronym, and code snippet precision.
* **Reciprocal Rank Fusion (RRF)**: Merges sparse and dense search rankings to surface the most relevant chunks.

### 2. 🎯 Cross-Encoder Neural Reranking
* Top candidates from hybrid retrieval are rescored using `cross-encoder/ms-marco-MiniLM-L-6-v2`.
* Eliminates semantic false positives, ensuring that only the most contextually grounded chunks are passed into the LLM context window.

### 3. ✍️ Vision OCR for Scanned & Mixed Documents
* Integrated **Vision OCR Service** (`backend/services/ocr_service.py`) using **PyMuPDF (fitz)** paired with **Gemini 2.5 Vision** and **Tesseract OCR** fallback.
* Transparently handles:
  * Standard printed PDFs
  * Scanned image-only PDFs
  * Handwritten notes
  * Mixed pages with diagrams, tables, and embedded illustrations
* Preserves 1-indexed page metadata for all OCR-extracted text.

### 4. ⚡ Real-Time SSE Token Streaming & Abort Control
* Live token-by-token streaming via **Server-Sent Events** (`/api/chat/stream`).
* Client-side `AbortController` support: users can immediately halt ongoing generation with the **Stop Generation** control.

### 5. 📊 Interactive Mermaid Architecture Flowcharts
* Generates interactive diagrams (flowcharts, mindmaps, sequence diagrams) on-demand or when explicitly requested (`"draw a flowchart"`).
* Direct in-browser rendering via Mermaid.js matching the application's Warm Beige & Burgundy theme.
* Interactive diagram toolbar: **Zoom In**, **Zoom Out**, **Reset View**, and **Copy Diagram Definition**.

### 6. 📌 Strictly Grounded Answers & Page Citations
* Enforces strict grounding prompts with zero hallucination. If a query cannot be answered from the document, the model returns a polite standard refusal.
* Every answer includes verifiable page citations with collapsible excerpt drawers.

### 7. 🎨 Refined, Minimal & Aesthetic UI
* Crafted in vanilla **HTML5 / CSS3 / JavaScript** (zero bloated frameworks).
* Curated **Warm Beige & Burgundy** color palette (`#FEF9F4`, `#F5E8E1`, `#8B1E3F`, `#3A1F1F`).
* Responsive sidebar featuring dynamic Knowledge Base capability badges and suggested inquiries.
* Right-anchored user message bubbles and left-anchored AI responses with markdown formatting and syntax-highlighted code blocks.

### 8. 🛡️ Strict Document Isolation
* ChromaDB vector collections isolate chunks with metadata filtering on `document_id`.
* Switching documents or resetting sessions guarantees zero cross-document data leakage.

---

## 📦 Project Directory Structure

```text
PDF_ASSISTANT_RAG_PROJECT/
├── backend/
│   ├── main.py                     # FastAPI application, CORS & static mounting
│   ├── config.py                   # App settings & environment configurations
│   ├── requirements.txt            # Python backend dependencies
│   ├── tests_verify.py             # 13-scenario automated verification suite
│   │
│   ├── routes/
│   │   ├── upload.py               # POST /api/upload (pipeline & OCR extraction)
│   │   ├── chat.py                 # POST /api/chat & GET /api/chat/stream (SSE)
│   │   └── questions.py            # POST /api/suggested-questions
│   │
│   ├── services/
│   │   ├── pdf_service.py          # PDF text extraction & page-aware chunking
│   │   ├── ocr_service.py          # Vision OCR for scanned pages & handwritten notes
│   │   ├── embedding_service.py    # Singleton Hugging Face MiniLM embedding model
│   │   ├── vector_service.py       # ChromaDB vector store & hybrid search
│   │   ├── rag_service.py          # Hybrid RRF, Cross-Encoder rerank & RAG pipeline
│   │   ├── question_service.py     # Representative sampling & question generation
│   │   └── llm_factory.py          # Multi-provider LLM interface (Gemini, etc.)
│   │
│   ├── models/
│   │   └── schemas.py              # Pydantic validation schemas
│   │
│   └── utils/
│       └── helpers.py              # Structured logging & text utilities
│
├── frontend/
│   ├── index.html                  # Minimalist Warm Beige & Burgundy UI
│   ├── css/
│   │   └── style.css               # Vanilla CSS design tokens & animations
│   └── js/
│       └── app.js                  # Frontend controller, SSE streaming & Mermaid
│
├── evaluation/
│   ├── README.md                   # Evaluation benchmark instructions
│   ├── questions.json              # Curated ground-truth test questions
│   └── run_evaluation.py           # Automated RAG metrics benchmark
│
├── data/
│   ├── uploads/                    # Local storage for uploaded PDF files
│   └── chroma/                     # Local persistent ChromaDB vector store
│
├── Dockerfile                      # Production container definition
├── .dockerignore                   # Docker build ignore rules
├── .env.example                    # Template environment configuration
├── .gitignore                      # Git ignore rules
└── README.md                       # Documentation
```

---

## ⚙️ Installation & Setup

### 1. Prerequisites
* **Python**: 3.11 or higher
* **Google Gemini API Key**: [Get API key from Google AI Studio](https://aistudio.google.com/)

### 2. Clone the Repository
```bash
git clone https://github.com/ankitabehera888/PDF_ASSISTANT_RAG_PROJECT-.git
cd PDF_ASSISTANT_RAG_PROJECT-
```

### 3. Set Up Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r backend/requirements.txt
```

### 5. Configure Environment Variables
Create a `.env` file in the project root:
```env
GEMINI_API_KEY=your_gemini_api_key_here
PRIMARY_LLM_MODEL=gemini-2.5-flash
FALLBACK_LLM_MODEL=gemma-4-26b-a4b-it
```

---

## 🏃 Running the Application

### Option A: Local FastAPI Server (Recommended)
FastAPI serves the REST/SSE API endpoints at `/api/*` and hosts the static frontend at `/`.

```bash
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```
Open your browser at: **`http://localhost:8000`**

### Option B: Docker Container
Build and run with Docker:

```bash
docker build -t queryfolio-rag .
docker run -p 8000:8000 --env-file .env queryfolio-rag
```

---

## 🧪 API Endpoints Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Health check and system readiness probe |
| `POST` | `/api/upload` | Upload PDF, run OCR if needed, generate hybrid indices and suggested inquiries |
| `POST` | `/api/chat` | Non-streaming question-answering with citations and diagram generation |
| `GET` | `/api/chat/stream` | **Server-Sent Events (SSE)** real-time streaming answer generation |
| `POST` | `/api/suggested-questions` | Fetch / re-generate topic-specific inquiries for a document ID |

---

## 📊 Verification & Evaluation

### 1. Run the Comprehensive 13-Scenario Verification Suite
QueryFolio includes an automated end-to-end verification script testing printed text, scanned pages, handwritten notes, streaming SSE, flowchart synthesis, refusal handling, and document isolation:

```bash
python backend/tests_verify.py
```

### 2. Run the RAG Evaluation Benchmark
Evaluate retrieval precision, grounding recall, and response latency against curated test cases:

```bash
python evaluation/run_evaluation.py
```

---

## 🔒 Security & Data Privacy

* **Strict Metadata Filtering**: Chunks and vectors are scoped strictly by `document_id`. Queries from one document never retrieve context from another.
* **Transient Session Control**: Clearing chat or swapping documents aborts active network streams and resets client-side context immediately.
* **Local Embedding & Vector Storage**: Embeddings and ChromaDB indexes are computed and stored locally in `data/chroma/`.

---

## 📄 License
This project is licensed under the MIT License.
