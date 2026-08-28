from typing import List
from langchain_core.documents import Document

from backend.config import GEMINI_API_KEY
from backend.models.schemas import SuggestedQuestionsResponse, SuggestedQuestion
from backend.services.llm_factory import LLMFactoryService
from backend.utils.helpers import logger


class QuestionService:
    def __init__(self):
        self.api_key = GEMINI_API_KEY
        self.llm_factory = LLMFactoryService()

    def _sample_representative_chunks(self, chunks: List[Document], max_chunks: int = 15) -> str:
        """Sample chunks uniformly across the document to get full context without intro bias."""
        if not chunks:
            return ""
        
        if len(chunks) <= max_chunks:
            selected_chunks = chunks
        else:
            step = len(chunks) / max_chunks
            selected_chunks = [chunks[int(i * step)] for i in range(max_chunks)]
        
        combined_text = []
        for doc in selected_chunks:
            page_info = f"[Page {doc.metadata.get('page', 1)}]"
            combined_text.append(f"{page_info}\n{doc.page_content[:800]}")
            
        return "\n\n---\n\n".join(combined_text)

    def generate_suggested_questions(self, chunks: List[Document]) -> List[str]:
        """
        Generates 5 to 8 topic-specific suggested questions based strictly on representative document content.
        Uses structured output via LLMFactoryService (Primary -> Fallback).
        """
        if not self.api_key or self.api_key == "your_gemini_api_key_here":
            logger.warning("GEMINI_API_KEY is not set. Returning fallback notice.")
            return [
                "Please configure your GEMINI_API_KEY in backend/.env to auto-generate topic-specific questions."
            ]

        sampled_text = self._sample_representative_chunks(chunks)
        if not sampled_text:
            return []

        logger.info("Generating topic-specific suggested questions via LLM Factory...")

        prompt = (
            "You are an expert document analyzer.\n"
            "Analyze the following representative content excerpts from a PDF document and generate 5 to 8 "
            "important, highly specific, topic-focused questions that can be answered using this document.\n\n"
            "STRICT RULES:\n"
            "1. NEVER generate generic questions such as 'What is this PDF about?', 'Summarize the PDF', "
            "'What is the methodology?', 'What are the key findings?', or 'What is the conclusion?' unless "
            "those are explicitly named concepts in the text.\n"
            "2. Identify specific key concepts, formulas, processes, architectures, or facts from the text.\n"
            "3. Craft clear, technical, topic-specific questions directly answerable by the text.\n"
            "4. Return 5 to 8 questions.\n\n"
            f"DOCUMENT EXCERPTS:\n---\n{sampled_text}\n---"
        )

        try:
            response: SuggestedQuestionsResponse = self.llm_factory.generate_structured(
                prompt=prompt,
                schema_class=SuggestedQuestionsResponse,
                temperature=0.2
            )

            if response and response.questions:
                questions_list = [q.question.strip() for q in response.questions if q.question.strip()]
                logger.info(f"Successfully generated {len(questions_list)} topic-specific questions.")
                return questions_list[:8]
        except Exception as e:
            logger.error(f"Failed to generate suggested questions via LLM Factory: {e}")

        # Graceful fallback if Gemini structured output parsing failed
        return [
            "What are the main technical concepts explained in this document?",
            "What specific definitions or formulas are presented in the text?",
            "How do the components described in this document interact with each other?"
        ]

