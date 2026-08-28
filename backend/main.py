import sys
from pathlib import Path

# Ensure project root & backend dir are in sys.path before importing backend submodules
_file_path = Path(__file__).resolve()
_backend_dir = _file_path.parent
_project_root = _backend_dir.parent

if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.routes import upload, chat, questions
from backend.models.schemas import HealthResponse
from backend.config import BASE_DIR
from backend.utils.helpers import logger

app = FastAPI(
    title="PDF RAG Assistant API",
    description="Full-stack PDF question-answering system with topic-specific suggested questions.",
    version="1.0.0"
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Routers
app.include_router(upload.router)
app.include_router(chat.router)
app.include_router(questions.router)


@app.get("/api/health", response_model=HealthResponse, tags=["health"])
async def health_check():
    """Health check endpoint to verify backend operational status."""
    return HealthResponse(status="ok")


# Mount static frontend directory if it exists
frontend_dir = BASE_DIR / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
    logger.info(f"Mounted static frontend from {frontend_dir}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

