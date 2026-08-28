from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List

from backend.services.vector_service import VectorService
from backend.services.question_service import QuestionService
from backend.utils.helpers import logger

router = APIRouter(prefix="/api", tags=["questions"])

vector_service = VectorService()
question_service = QuestionService()


class SuggestedQuestionsRequest(BaseModel):
    document_id: str


class SuggestedQuestionsListResponse(BaseModel):
    document_id: str
    suggested_questions: List[str]


@router.post("/suggested-questions", response_model=SuggestedQuestionsListResponse)
async def get_suggested_questions(request: SuggestedQuestionsRequest):
    """
    Endpoint to re-generate or retrieve suggested questions for a given document_id.
    """
    if not request.document_id:
        raise HTTPException(status_code=400, detail="document_id is required.")

    # Fetch document chunks from vector store for document_id
    chunks = vector_service.similarity_search(
        query="important key concepts algorithms definitions architecture",
        document_id=request.document_id,
        top_k=15
    )

    if not chunks:
        raise HTTPException(status_code=444 if hasattr(HTTPException, 'status') else 404, detail="Document not found or has no content.")

    questions = question_service.generate_suggested_questions(chunks)
    
    return SuggestedQuestionsListResponse(
        document_id=request.document_id,
        suggested_questions=questions
    )

