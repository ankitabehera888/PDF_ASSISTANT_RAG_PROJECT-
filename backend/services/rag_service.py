import re
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document

from backend.config import GEMINI_API_KEY, RETRIEVAL_TOP_K, RERANK_TOP_K, DEBUG_MODE
from backend.services.vector_service import VectorService
from backend.services.reranker_service import RerankerService
from backend.services.llm_factory import LLMFactoryService
from backend.models.schemas import ChatResponse, SourceItem
from backend.utils.helpers import logger
from fastapi import HTTPException


REFUSAL_MESSAGE = "I couldn't find the answer to that in the uploaded PDF."


class RAGService:
    def __init__(self):
        self.api_key = GEMINI_API_KEY
        self.vector_service = VectorService()
        self.reranker_service = RerankerService()
        self.llm_factory = LLMFactoryService()

    def _preprocess_query(self, query: str) -> str:
        """
        Lightweight normalization preserving names, dates, numbers, negations,
        permissions, and question intent.
        """
        if not query:
            return ""
        cleaned = re.sub(r'[\r\n\t]+', ' ', query).strip()
        cleaned = re.sub(r'\s+', ' ', cleaned)
        return cleaned

    def _build_evidence_context(self, chunks: List[Document]) -> str:
        """Format retrieved candidate chunks with section headings and explicit source IDs."""
        context_parts = []
        for doc in chunks:
            chunk_id = doc.metadata.get("chunk_id", "source_chunk")
            page = doc.metadata.get("page", 1)
            section = doc.metadata.get("section", "General")
            subsection = doc.metadata.get("subsection", "")

            sec_info = f"Section: {section}"
            if subsection:
                sec_info += f" | Subsection: {subsection}"

            part_text = (
                f"SOURCE [ID: {chunk_id}]\n"
                f"Page: {page}\n"
                f"{sec_info}\n"
                f"Text:\n{doc.metadata.get('raw_text', doc.page_content)}\n"
            )
            context_parts.append(part_text)

        return "\n---\n".join(context_parts)

    def _extract_verified_sources(self, chunks: List[Document], answer_text: str) -> List[SourceItem]:
        """
        Maps candidate source IDs back to exact metadata (page, section, subsection, excerpt).
        Verifies that cited sources come strictly from the retrieved chunks.
        """
        sources: List[SourceItem] = []
        seen_keys = set()

        for doc in chunks:
            chunk_id = doc.metadata.get("chunk_id")
            page = int(doc.metadata.get("page", 1))
            section = doc.metadata.get("section")
            subsection = doc.metadata.get("subsection")
            raw_text = doc.metadata.get("raw_text", doc.page_content)

            # Unique key per section/page combination
            sec_key = f"{page}_{section}"
            if sec_key in seen_keys:
                continue
            seen_keys.add(sec_key)

            # Excerpt snippet (~150-180 characters)
            excerpt = raw_text.strip().replace("\n", " ")
            if len(excerpt) > 180:
                excerpt = excerpt[:180] + "..."

            sources.append(SourceItem(
                page=page,
                text=excerpt,
                section=section,
                subsection=subsection,
                chunk_id=chunk_id
            ))

        sources.sort(key=lambda s: (s.page, s.section or ""))
        return sources

    def answer_question(
        self,
        document_id: str,
        question: str,
        retrieval_k: int = RETRIEVAL_TOP_K,
        final_k: int = RERANK_TOP_K
    ) -> ChatResponse:
        """
        Executes V2 RAG pipeline:
        1. Preprocess query cleanly.
        2. Initial High-Recall Search in ChromaDB (Retrieves ~10 candidate chunks filtered by document_id).
        3. Cross-Encoder Reranking (Reranks candidates and selects top ~4 best chunks).
        4. Evidence Selection & Formatting with structural section context.
        5. Gemini Grounded Answer Generation with strict refusal & citation guidelines.
        6. Source Verification (Maps used source IDs back to page & section metadata).
        """
        if not question or not question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty.")

        if not self.api_key or self.api_key == "your_gemini_api_key_here":
            return ChatResponse(
                answer="GEMINI_API_KEY is not configured in backend/.env. Please add your API key to proceed.",
                sources=[]
            )

        clean_query = self._preprocess_query(question)
        logger.info(f"V2 RAG pipeline starting for doc_id '{document_id}' | Query: '{clean_query}'")

        # 1. High-recall candidate retrieval (10 candidates)
        candidates = self.vector_service.similarity_search(
            query=clean_query,
            document_id=document_id,
            top_k=retrieval_k
        )

        if not candidates:
            logger.info(f"No candidates found in ChromaDB for document_id '{document_id}'")
            return ChatResponse(
                answer=REFUSAL_MESSAGE,
                sources=[]
            )

        # 2. Cross-Encoder Reranking (Rerank 10 -> 4)
        ranked_chunks, debug_scores = self.reranker_service.rerank(
            query=clean_query,
            candidates=candidates,
            top_k=final_k
        )

        # 3. Format structured evidence context
        evidence_context = self._build_evidence_context(ranked_chunks)

        # 4. Strict Source-Aware Prompt
        system_prompt = (
            "You are a document-grounded question answering assistant.\n\n"
            "Use ONLY the supplied document evidence to answer the user's question.\n\n"
            "STRICT RULES:\n"
            "1. When multiple passages are relevant, prefer the passage that DIRECTLY addresses the user's exact question.\n"
            "2. Do not substitute a general or related policy for a directly applicable section.\n"
            "3. Distinguish clearly between general rules, exceptions, conditions, thresholds, permissions, and prohibitions.\n"
            "4. Preserve numerical values, monetary limits, dates, and names exactly.\n"
            "5. If the answer is not supported by the supplied evidence, output strictly:\n"
            f'"{REFUSAL_MESSAGE}"\n'
            "6. Do NOT use outside knowledge or invent facts.\n"
            "7. Base your answer strictly on the supplied sources below.\n"
            "8. Write answers in a clean, structured format:\n"
            "   - Use a clear heading when covering a distinct topic.\n"
            "   - Use subheadings when containing multiple sub-sections.\n"
            "   - Use short paragraphs for explanations.\n"
            "   - Use bullet points when presenting multiple separate items.\n"
            "   - Use numbered lists when presenting sequential steps or procedures.\n"
            "   - Use bold emphasis for important terms.\n"
            "   - Do not repeat the question as a heading.\n"
            "   - Keep concise while preserving all factual information.\n\n"
            f"SUPPLIED DOCUMENT EVIDENCE:\n{evidence_context}\n\n"
            f"USER QUESTION:\n{clean_query}"
        )


        # 5. Generate Grounded Answer via Centralized LLM Factory
        answer_text = self.llm_factory.generate_text(prompt=system_prompt, temperature=0.1)

        # Refusal check
        if REFUSAL_MESSAGE.lower() in answer_text.lower() or "couldn't find the answer" in answer_text.lower():
            return ChatResponse(
                answer=REFUSAL_MESSAGE,
                sources=[]
            )

        # 6. Extract and verify sources
        sources = self._extract_verified_sources(ranked_chunks, answer_text)

        # Build debug dictionary for development inspection
        debug_info = None
        if DEBUG_MODE:
            debug_info = {
                "initial_candidates_count": len(candidates),
                "final_ranked_count": len(ranked_chunks),
                "reranker_enabled": self.reranker_service.enabled,
                "rankings": debug_scores
            }

        logger.info(f"V2 RAG successfully generated answer with {len(sources)} verified sources.")
        return ChatResponse(
            answer=answer_text,
            sources=sources,
            debug_info=debug_info
        )
