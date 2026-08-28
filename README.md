# PDF RAG Assistant with Topic-Specific Suggested Questions

A full-stack, local PDF Question-Answering application built with **FastAPI**, **ChromaDB**, **Hugging Face Sentence Transformers (`all-MiniLM-L6-v2`)**, **Google Gemini**, and a vanilla **HTML5/CSS3/JavaScript** frontend.

Upload any PDF document to automatically extract text, build a persistent vector index, generate **5–8 document-specific suggested questions**, and ask questions with strictly grounded answers and source page citations.

---

## 🏗️ Architecture Pipeline

```text
PDF Upload (.pdf)
    ↓
PyPDF Extraction (Page Metadata preserved)
    ↓
Recursive Character Text Chunking (size: 1000, overlap: 150)
    ↓
Hugging Face Sentence Transformer (all-MiniLM-L6-v2 → 384-dim vectors)
    ↓
ChromaDB Local Vector Database (Filtered by document_id)
    ↓
Representative Sampling & Gemini Topic Analysis
    ↓
5–8 Topic-Specific Suggested Questions (Structured JSON)
    ↓
User Query → Similarity Search → Grounded Gemini RAG → Answer + Source Pages
```

---

## 🚀 Key Features

* **Hugging Face Local Embeddings**: Uses `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions) for both document chunks and user queries.
* **ChromaDB Vector Storage**: Persistent local vector store (`data/chroma/`) with strict metadata filtering per `document_id` to prevent cross-document contamination.
* **Automatic Topic-Specific Questions**: Samples representative content across the document to generate 5–8 non-generic, highly specific questions using Gemini structured output.
* **Strictly Grounded RAG Answers**: Enforces refusal (`"I couldn't find the answer to that in the uploaded PDF."`) when questions are outside document scope.
* **Source Citations**: Displays page numbers (1-indexed) and short excerpts for every answer.
* **Modern Vanilla Frontend**: Sleek dark glassmorphism interface built with Vanilla HTML5, CSS3, and JavaScript (No React, Tailwind, or complex frameworks).

---

## 📦 Project Directory Structure

```text
pdf-rag-assistant/
│
├── backend/
│   ├── main.py                   # FastAPI entrypoint, CORS & static mounting
│   ├── config.py                 # App settings & environment variables
│   │
│   ├── routes/
│   │   ├── upload.py             # POST /api/upload endpoint
│   │   ├── chat.py               # POST /api/chat endpoint
│   │   └── questions.py          # POST /api/suggested-questions endpoint
│   │
│   ├── services/
│   │   ├── pdf_service.py        # PDF text loading & page-aware chunking
│   │   ├── embedding_service.py  # Singleton Hugging Face embedding model
│   │   ├── vector_service.py     # ChromaDB persistence & similarity search
│   │   ├── question_service.py   # Representative chunk sampling & question generation
│   │   └── rag_service.py        # Grounded RAG chain with page citations
│   │
│   ├── models/
│   │   └── schemas.py            # Pydantic request/response schemas
│   │
│   ├── utils/
│   │   └── helpers.py            # Utility functions & logging
│   │
│   └── requirements.txt          # Python dependencies
│
├── frontend/
│   ├── index.html                # Modern glassmorphism web interface
│   ├── css/
│   │   └── style.css             # Vanilla CSS design system
│   └── js/
│       └── app.js                # UI state management & Fetch API integration
│
├── data/
│   ├── uploads/                  # Local storage for uploaded PDF files
│   └── chroma/                   # Local persistent ChromaDB vector store
│
├── .env                          # Local environment variables
├── .env.example                  # Environment template
├── .gitignore                    # Git ignore configuration
└── README.md                     # Documentation
```

---

## ⚙️ Installation & Setup

### 1. Prerequisites
* Python 3.11+

### 2. Set Up Virtual Environment (Windows)
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r backend/requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env` and add your Google Gemini API Key:
```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
```

---

## 🏃 Running the Application

### Option A: Run Backend & Frontend via FastAPI (Recommended)
FastAPI automatically serves the API at `/api/*` and mounts the static web frontend at `/`.

```bash
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Open your browser at:
`http://localhost:8000`

### Option B: Run API Server and Frontend Separately
Start backend API:
```bash
uvicorn backend.main:app --reload --port 8000
```
Open `frontend/index.html` using Live Server or serve statically via Python:
```bash
python -m http.server 3000 --directory frontend
```
Navigate to `http://localhost:3000`.

---

## 🧪 API Endpoints Summary

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Service health status check |
| `POST` | `/api/upload` | Upload PDF file, chunk, store vectors & generate suggested questions |
| `POST` | `/api/chat` | Submit question for document RAG QA |
| `POST` | `/api/suggested-questions` | Fetch / re-generate topic-specific questions for document ID |

---

## 🔮 Future Improvements
* Multi-document similarity search & comparison mode.
* Streaming response chunks via Server-Sent Events (SSE).
* Hybrid keyword-semantic search (BM25 + Dense embeddings) and Cross-Encoder reranking.
* Optical Character Recognition (OCR) support for scanned PDF documents.
* User authentication and persistent chat conversation memory.
