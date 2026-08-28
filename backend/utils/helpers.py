import uuid
import logging
from pathlib import Path
from fastapi import HTTPException, UploadFile

# Setup logger
logger = logging.getLogger("pdf_assistant")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def generate_document_id() -> str:
    """Generate a unique UUID for each uploaded document."""
    return str(uuid.uuid4())


def validate_pdf_file(file: UploadFile) -> None:
    """Validate uploaded file type and size."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")
    
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    if file.content_type and file.content_type != "application/pdf":
        # Some browsers might send application/x-pdf
        if "pdf" not in file.content_type.lower():
            raise HTTPException(status_code=400, detail="Only PDF files are supported.")


def clean_text(text: str) -> str:
    """Clean extracted raw PDF text."""
    if not text:
        return ""
    # Remove null bytes and excessive newlines/spaces
    cleaned = text.replace("\x00", "").strip()
    return cleaned
