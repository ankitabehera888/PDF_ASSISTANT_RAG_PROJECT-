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
    ocr_pages: Optional[int] = 0  # How many pages were processed via OCR


class ChatRequest(BaseModel):
    document_id: str
    question: str
    history: Optional[List[dict]] = None


class SourceItem(BaseModel):
    page: int
    text: str
    section: Optional[str] = None
    subsection: Optional[str] = None
    chunk_id: Optional[str] = None
    # OCR/handwriting metadata
    ocr_derived: Optional[bool] = False
    ocr_confidence: Optional[float] = None
    low_confidence: Optional[bool] = False
    page_number: Optional[int] = None
    source_type: Optional[str] = "printed"
    extraction_method: Optional[str] = "digital_text"


class FlowchartData(BaseModel):
    """Structured Mermaid flowchart data with simple English description."""
    type: str = "mermaid"
    definition: str = Field(..., description="Mermaid flowchart definition string")
    description: Optional[str] = Field(None, description="Clear, simple English step-by-step description explaining the flowchart")


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceItem]
    flowchart: Optional[FlowchartData] = None
    debug_info: Optional[dict] = None


class StreamChatResponse(BaseModel):
    """Final payload sent at the end of a streaming response."""
    answer: str
    sources: List[SourceItem]
    flowchart: Optional[FlowchartData] = None
    debug_info: Optional[dict] = None


class HealthResponse(BaseModel):
    status: str
