from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path

from backend.config import UPLOAD_DIR
from backend.models.schemas import UploadResponse
from backend.services.pdf_service import PDFService
from backend.services.vector_service import VectorService
from backend.services.question_service import QuestionService
from backend.utils.helpers import validate_pdf_file, generate_document_id, logger

router = APIRouter(prefix="/api", tags=["upload"])

# Singleton / Shared service instances
pdf_service = PDFService()
vector_service = VectorService()
question_service = QuestionService()


@router.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Endpoint for uploading a PDF document.
    Pipeline: Validate -> Save -> Parse/Extract -> Chunk -> Embed & Store in ChromaDB -> Generate Suggested Questions.
    """
    validate_pdf_file(file)
    
    document_id = generate_document_id()
    saved_filename = f"{document_id}.pdf"
    file_path = UPLOAD_DIR / saved_filename

    try:
        # Save file to disk
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
            
        with open(file_path, "wb") as f:
            f.write(contents)

        logger.info(f"Saved uploaded PDF to {file_path}")

        # Process PDF (extract pages & split into semantic chunks)
        chunks, total_pages = pdf_service.load_and_process_pdf(
            file_path=str(file_path),
            document_id=document_id,
            original_filename=file.filename
        )

        # Store embeddings in ChromaDB
        total_chunks = vector_service.add_documents(chunks)

        # Generate topic-specific suggested questions
        suggested_questions = question_service.generate_suggested_questions(chunks)

        return UploadResponse(
            success=True,
            document_id=document_id,
            filename=file.filename,
            pages=total_pages,
            chunks=total_chunks,
            suggested_questions=suggested_questions
        )

    except HTTPException:
        # Re-raise explicit HTTP exceptions
        if file_path.exists():
            file_path.unlink()
        raise
    except Exception as e:
        logger.error(f"Unexpected error during PDF processing: {e}")
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail="An error occurred while processing the PDF.")
