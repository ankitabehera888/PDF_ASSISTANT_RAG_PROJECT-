from typing import List, Optional
from pydantic import BaseModel, Field


class SuggestedQuestion(BaseModel):
    question: str = Field(..., description="The specific, topic-focused question derived from the PDF")
    topic: str = Field(..., description="The main concept or topic area the question covers")


class SuggestedQuestionsResponse(BaseModel):
    questions: List[SuggestedQuestion] = Field(..., description="List of 5 to 8 topic-specific suggested questions")


class UploadResponse(BaseModel):
    success: bool
    document_id: str
    filename: str
    pages: int
    chunks: int
    suggested_questions: List[str]



class ChatRequest(BaseModel):
    document_id: str
    question: str


class SourceItem(BaseModel):
    page: int
    text: str
    section: Optional[str] = None
    subsection: Optional[str] = None
    chunk_id: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceItem]
    debug_info: Optional[dict] = None


class HealthResponse(BaseModel):
    status: str

