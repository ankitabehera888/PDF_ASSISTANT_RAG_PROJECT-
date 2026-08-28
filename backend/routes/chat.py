from fastapi import APIRouter, HTTPException
from backend.models.schemas import ChatRequest, ChatResponse
from backend.services.rag_service import RAGService
from backend.utils.helpers import logger

router = APIRouter(prefix="/api", tags=["chat"])
rag_service = RAGService()


@router.post("/chat", response_model=ChatResponse)
async def chat_with_pdf(request: ChatRequest):
    """
    Endpoint for asking questions about a specific uploaded PDF document.
    """
    if not request.document_id:
        raise HTTPException(status_code=400, detail="document_id is required.")
        
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty.")

    logger.info(f"Chat request received for document_id '{request.document_id}': {request.question}")
    
    return rag_service.answer_question(
        document_id=request.document_id,
        question=request.question
    )
