import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Base Directory: RAG_PROJECT root
BASE_DIR = Path(__file__).resolve().parent.parent
_backend_dir = Path(__file__).resolve().parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

# Load .env file from base directory
env_path = BASE_DIR / ".env"
load_dotenv(dotenv_path=env_path)


# AI & LLM Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
PRIMARY_LLM_MODEL = os.getenv("PRIMARY_LLM_MODEL", "gemini-2.5-flash")
FALLBACK_LLM_MODEL = os.getenv("FALLBACK_LLM_MODEL", "gemma-4-26b-a4b-it")

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# Storage Paths
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
CHROMA_DIR = BASE_DIR / "data" / "chroma"

# Ensure storage directories exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# Document Chunking Settings (V2: semantic section-aware chunking)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))

# RAG & Retrieval Settings
DEFAULT_TOP_K = 5
COLLECTION_NAME = "pdf_documents"

# V2 High-Recall Retrieval & Reranker Settings
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", 10))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", 4))
RERANKER_ENABLED = os.getenv("RERANKER_ENABLED", "true").lower() in ("true", "1", "yes")
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() in ("true", "1", "yes")

