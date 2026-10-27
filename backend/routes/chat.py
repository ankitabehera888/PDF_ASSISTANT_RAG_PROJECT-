from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from backend.models.schemas import ChatRequest, ChatResponse
from backend.services.rag_service import RAGService
from backend.utils.helpers import logger

router = APIRouter(prefix="/api", tags=["chat"])
rag_service = RAGService()


@router.post("/chat", response_model=ChatResponse)
async def chat_with_pdf(request: ChatRequest):
    """
    Non-streaming endpoint for asking questions about a specific uploaded PDF document.
    Preserved for evaluation pipeline and backward compatibility.
    """
    if not request.document_id:
        raise HTTPException(status_code=400, detail="document_id is required.")
        
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty.")

    logger.info(f"Chat request received for document_id '{request.document_id}': {request.question}")
    
    return rag_service.answer_question(
        document_id=request.document_id,
        question=request.question,
        history=request.history
    )


@router.post("/chat/stream")
async def chat_with_pdf_stream(request: ChatRequest):
    """
    Streaming endpoint for real-time answer generation via Server-Sent Events (SSE).
    
    The response is a text/event-stream with JSON-encoded events:
      - {"type": "status", "message": "..."}     — pipeline progress
      - {"type": "chunk",  "text": "..."}         — streaming answer tokens
      - {"type": "flowchart", "data": {...}}      — Mermaid diagram
      - {"type": "sources",  "data": [...]}       — citations
      - {"type": "done"}                          — stream complete
      - {"type": "error",  "message": "..."}      — error event
    """
    if not request.document_id:
        raise HTTPException(status_code=400, detail="document_id is required.")
        
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty.")

    logger.info(f"Streaming chat request for document_id '{request.document_id}': {request.question}")

    return StreamingResponse(
        rag_service.answer_question_stream(
            document_id=request.document_id,
            question=request.question,
            history=request.history
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
            "Connection": "keep-alive",
        }
    )
