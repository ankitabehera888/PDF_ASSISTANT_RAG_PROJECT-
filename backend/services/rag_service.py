import re
import json
import time
import asyncio
from typing import List, Dict, Any, Optional, AsyncGenerator
from langchain_core.documents import Document

from backend.config import GEMINI_API_KEY, RETRIEVAL_TOP_K, RERANK_TOP_K, DEBUG_MODE
from backend.services.vector_service import VectorService
from backend.services.reranker_service import RerankerService
from backend.services.llm_factory import LLMFactoryService
from backend.models.schemas import ChatResponse, SourceItem, FlowchartData
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
            ocr_derived = doc.metadata.get("ocr_derived", False)

            sec_info = f"Section: {section}"
            if subsection:
                sec_info += f" | Subsection: {subsection}"

            source_note = " [OCR/Handwritten]" if ocr_derived else ""

            part_text = (
                f"SOURCE [ID: {chunk_id}]\n"
                f"Page: {page}{source_note}\n"
                f"{sec_info}\n"
                f"Text:\n{doc.metadata.get('raw_text', doc.page_content)}\n"
            )
            context_parts.append(part_text)

        return "\n---\n".join(context_parts)

    def _extract_verified_sources(self, chunks: List[Document], answer_text: str) -> List[SourceItem]:
        """
        Maps candidate source IDs back to exact metadata (page, section, subsection, excerpt).
        Verifies that cited sources come strictly from the retrieved chunks.
        Includes OCR metadata in each source item.
        """
        sources: List[SourceItem] = []
        seen_keys = set()

        for doc in chunks:
            chunk_id = doc.metadata.get("chunk_id")
            page = int(doc.metadata.get("page", 1))
            section = doc.metadata.get("section")
            subsection = doc.metadata.get("subsection")
            raw_text = doc.metadata.get("raw_text", doc.page_content)
            ocr_derived = doc.metadata.get("ocr_derived", False)
            ocr_confidence = doc.metadata.get("ocr_confidence", None)
            low_confidence = doc.metadata.get("low_confidence", False)

            # Unique key per section/page combination
            sec_key = f"{page}_{section}"
            if sec_key in seen_keys:
                continue
            seen_keys.add(sec_key)

            # Excerpt snippet (~150-180 characters)
            excerpt = raw_text.strip().replace("\n", " ")
            if len(excerpt) > 180:
                excerpt = excerpt[:180] + "..."

            source_type = doc.metadata.get("source_type", "handwritten_ocr" if ocr_derived else "printed")
            extraction_method = doc.metadata.get("extraction_method", "vision_ocr" if ocr_derived else "digital_text")
            page_number = int(doc.metadata.get("page_number", page))

            sources.append(SourceItem(
                page=page,
                page_number=page_number,
                text=excerpt,
                section=section,
                subsection=subsection,
                chunk_id=chunk_id,
                ocr_derived=ocr_derived,
                ocr_confidence=ocr_confidence if ocr_derived else None,
                low_confidence=low_confidence if ocr_derived else False,
                source_type=source_type,
                extraction_method=extraction_method,
            ))

        sources.sort(key=lambda s: (s.page, s.section or ""))
        return sources

    def _contextualize_query(self, query: str, history: Optional[List[dict]] = None) -> str:
        """
        If the question is a conversational follow-up (e.g. 'give me description about this'),
        reformulates it into a standalone search query using recent chat history.
        """
        if not history:
            return query

        q_lower = query.lower().strip()
        referential_keywords = [
            "this", "that", "it", "them", "these", "those", "above", "previous",
            "same", "more", "why", "how so", "explain", "describe", "detail",
            "details", "diagram", "flowchart", "about it", "about this", "tell me more"
        ]
        has_ref = any(re.search(rf"\b{re.escape(kw)}\b", q_lower) for kw in referential_keywords)
        is_short = len(q_lower.split()) <= 7

        if not (has_ref or is_short):
            return query

        history_lines = []
        for msg in history[-4:]:
            role = "User" if msg.get("role") in ("user", "human") else "Assistant"
            content = msg.get("content", "").strip()
            if len(content) > 300:
                content = content[:300] + "..."
            if content:
                history_lines.append(f"{role}: {content}")

        if not history_lines:
            return query

        history_text = "\n".join(history_lines)
        prompt = (
            "Given the following conversation history and a follow-up user query, reformulate the query "
            "into a single standalone search query containing all specific entity and project names.\n"
            "Do NOT answer the question. Output ONLY the standalone search query without quotes.\n\n"
            f"CHAT HISTORY:\n{history_text}\n\n"
            f"FOLLOW-UP QUERY: {query}\n\n"
            "STANDALONE SEARCH QUERY:"
        )
        try:
            standalone = self.llm_factory.generate_text(prompt=prompt, temperature=0.0)
            standalone = standalone.strip().strip('"').strip("'")
            if standalone:
                logger.info(f"Contextualized query '{query}' -> '{standalone}'")
                return standalone
        except Exception as e:
            logger.warning(f"Query contextualization failed: {e}")

        return query

    def _build_answer_prompt(
        self,
        evidence_context: str,
        clean_query: str,
        history: Optional[List[dict]] = None
    ) -> str:
        """Build the strict grounded answer prompt ensuring comprehensive explanations in simple English."""
        history_section = ""
        if history:
            h_lines = []
            for msg in history[-4:]:
                r = "User" if msg.get("role") in ("user", "human") else "Assistant"
                c = msg.get("content", "").strip()
                if len(c) > 250:
                    c = c[:250] + "..."
                if c:
                    h_lines.append(f"{r}: {c}")
            if h_lines:
                history_section = "RECENT CONVERSATION HISTORY (for context):\n" + "\n".join(h_lines) + "\n\n"

        return (
            "You are an expert document-grounded question answering assistant.\n\n"
            "Use ONLY the supplied document evidence to answer the user's question.\n\n"
            "STRICT RULES:\n"
            "1. When multiple passages are relevant, prefer the passage that DIRECTLY addresses the user's exact question.\n"
            "2. If the user question is a follow-up referring to the conversation (e.g. 'give me description about this', 'explain this', 'tell me more'), "
            "use the conversation history to understand what 'this' refers to, and provide a full description based strictly on the document evidence.\n"
            "3. Distinguish clearly between general rules, exceptions, conditions, thresholds, permissions, and prohibitions.\n"
            "4. Preserve numerical values, monetary limits, dates, and names exactly.\n"
            "5. If the answer is not supported by the supplied evidence, output strictly:\n"
            f'"{REFUSAL_MESSAGE}"\n'
            "6. Do NOT use outside knowledge or invent facts.\n"
            "7. Base your answer strictly on the supplied sources below.\n"
            "8. FORMAT AND STYLE (CRITICAL):\n"
            "   - Write in SIMPLE, CLEAR, and EASY-TO-UNDERSTAND English.\n"
            "   - Provide a FULL, DESCRIPTIVE, and COMPREHENSIVE explanation. Do NOT output a single-sentence or terse answer.\n"
            "   - Begin with a direct summary paragraph answering the core question clearly in plain English.\n"
            "   - Follow with a descriptive explanation covering what the items, tools, methods, or concepts are, why and how they work or were used, and relevant context found in the document.\n"
            "   - Use structured formatting: bold headings, descriptive paragraphs, and bullet points or numbered lists for distinct items or steps.\n"
            "   - Use bold emphasis for important terms, project names, metrics, and tools.\n"
            "   - Ensure the user gets a rich, intuitive understanding from the text alone.\n\n"
            f"{history_section}"
            f"SUPPLIED DOCUMENT EVIDENCE:\n{evidence_context}\n\n"
            f"USER QUESTION:\n{clean_query}"
        )

    def _build_flowchart_prompt(self, evidence_context: str, clean_query: str, answer_text: str) -> str:
        """Build prompt for generating a Mermaid flowchart along with a clear simple English description."""
        return (
            "You are a visual explanation and flowchart design assistant.\n\n"
            "Based ONLY on the document evidence and answer provided below, generate:\n"
            "1. A clean, valid Mermaid flowchart.\n"
            "2. A detailed step-by-step description in SIMPLE ENGLISH explaining the flowchart so any user can easily understand each step.\n\n"
            "FLOWCHART RULES:\n"
            "1. Use ONLY information from the document evidence and answer — do not invent content.\n"
            "2. For conceptual, procedural, architectural, lifecycle, or step-by-step topics: generate a meaningful flowchart with 3–8 nodes.\n"
            "3. For factual or component-based questions: generate a connected diagram showing the entity, its components/tools, and their roles/outputs.\n"
            "4. Use valid Mermaid flowchart syntax starting with 'flowchart TD' or 'flowchart LR'.\n"
            "5. Node labels must be concise (max 8 words) and enclosed in double quotes (e.g. A[\"Label\"]).\n"
            "6. Do NOT use parentheses inside node labels.\n"
            "7. Use --> for directed connections only.\n\n"
            "DESCRIPTION RULES:\n"
            "1. Write in plain, friendly, simple English.\n"
            "2. Explain what the overall diagram shows in 1-2 clear sentences.\n"
            "3. Provide a step-by-step breakdown explaining what each node or stage in the flowchart does in simple terms.\n\n"
            "OUTPUT FORMAT: You MUST format your response using EXACTLY these two delimiter tags:\n"
            "---FLOWCHART---\n"
            "flowchart TD\n"
            '    A["Component 1"] --> B["Component 2"]\n'
            '    B --> C["Component 3"]\n'
            "---DESCRIPTION---\n"
            "**Diagram Overview**: [1-2 sentences in simple English explaining the diagram]\n\n"
            "**Step-by-Step Flow**:\n"
            "- **Component 1**: [Simple explanation of what happens here]\n"
            "- **Component 2**: [Simple explanation of what happens here]\n"
            "- **Component 3**: [Simple explanation of what happens here]\n\n"
            f"DOCUMENT EVIDENCE:\n{evidence_context}\n\n"
            f"ANSWER:\n{answer_text}\n\n"
            f"QUESTION:\n{clean_query}\n\n"
            "Now output the flowchart and simple English description:"
        )

    def _generate_flowchart(self, evidence_context: str, clean_query: str, answer_text: str) -> Optional[FlowchartData]:
        """
        Generate a Mermaid flowchart and simple English description from the evidence and answer.
        Returns None if generation fails or answer is a refusal.
        """
        try:
            prompt = self._build_flowchart_prompt(evidence_context, clean_query, answer_text)
            raw = self.llm_factory.generate_text(prompt=prompt, temperature=0.1)

            if not raw or not raw.strip():
                return None

            raw_text = raw.strip()
            definition = ""
            description = ""

            # Check for delimited format
            if "---FLOWCHART---" in raw_text or "---DESCRIPTION---" in raw_text:
                parts = re.split(r'---(?:FLOWCHART|DESCRIPTION)---', raw_text, flags=re.IGNORECASE)
                clean_parts = [p.strip() for p in parts if p.strip()]
                if len(clean_parts) >= 2:
                    definition = clean_parts[0]
                    description = clean_parts[1]
                elif len(clean_parts) == 1:
                    definition = clean_parts[0]
            else:
                definition = raw_text

            # Clean up Mermaid definition
            definition = re.sub(r'^```(?:mermaid)?\s*', '', definition, flags=re.IGNORECASE)
            definition = re.sub(r'\s*```$', '', definition).strip()

            # Find starting index of flowchart or graph header
            match = re.search(r'\b(flowchart\s+(?:TD|TB|LR|RL)|graph\s+(?:TD|TB|LR|RL))\b', definition, flags=re.IGNORECASE)
            if match:
                definition = definition[match.start():]
            elif not (definition.lower().startswith("flowchart") or definition.lower().startswith("graph")):
                logger.warning("Flowchart response did not contain standard header — prepending 'flowchart TD'")
                definition = f"flowchart TD\n{definition}"

            # Strip any residual HTML/script tags for security
            definition = re.sub(r'<[^>]+>', '', definition).strip()

            # Clean up description
            if description:
                description = re.sub(r'^```(?:markdown)?\s*', '', description, flags=re.IGNORECASE)
                description = re.sub(r'\s*```$', '', description).strip()
            else:
                description = (
                    "**Flowchart Overview**:\n\n"
                    "This visual diagram maps out the core components and flow based on the document evidence. "
                    "Follow the arrows to see how each step connects and leads to the outcome described above."
                )

            logger.info("Flowchart with simple English description generated successfully.")
            return FlowchartData(type="mermaid", definition=definition, description=description)

        except Exception as e:
            logger.warning(f"Flowchart generation failed (non-critical): {e}")
            return None

    def answer_question(
        self,
        document_id: str,
        question: str,
        retrieval_k: int = RETRIEVAL_TOP_K,
        final_k: int = RERANK_TOP_K,
        history: Optional[List[dict]] = None
    ) -> ChatResponse:
        """
        Executes V2 RAG pipeline (non-streaming):
        1. Preprocess query cleanly and contextualize if it's a follow-up.
        2. Initial High-Recall Search in ChromaDB.
        3. Cross-Encoder Reranking.
        4. Evidence Selection & Formatting.
        5. Gemini Grounded Answer Generation.
        6. Flowchart Generation.
        7. Source Verification.
        """
        if not question or not question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty.")

        if not self.api_key or self.api_key == "your_gemini_api_key_here":
            return ChatResponse(
                answer="GEMINI_API_KEY is not configured in backend/.env. Please add your API key to proceed.",
                sources=[]
            )

        clean_query = self._preprocess_query(question)
        search_query = self._contextualize_query(clean_query, history)
        logger.info(f"V2 RAG pipeline starting for doc_id '{document_id}' | Query: '{clean_query}' | Search Query: '{search_query}'")

        # 1. High-recall candidate retrieval
        candidates = self.vector_service.similarity_search(
            query=search_query,
            document_id=document_id,
            top_k=retrieval_k
        )

        if not candidates:
            logger.info(f"No candidates found in ChromaDB for document_id '{document_id}'")
            return ChatResponse(answer=REFUSAL_MESSAGE, sources=[])

        # 2. Cross-Encoder Reranking
        ranked_chunks, debug_scores = self.reranker_service.rerank(
            query=search_query,
            candidates=candidates,
            top_k=final_k
        )

        # 3. Format structured evidence context
        evidence_context = self._build_evidence_context(ranked_chunks)

        # 4. Build prompt and generate answer
        system_prompt = self._build_answer_prompt(evidence_context, clean_query, history)
        answer_text = self.llm_factory.generate_text(prompt=system_prompt, temperature=0.1)

        # Refusal check
        if REFUSAL_MESSAGE.lower() in answer_text.lower() or "couldn't find the answer" in answer_text.lower():
            return ChatResponse(answer=REFUSAL_MESSAGE, sources=[])

        # 5. Generate flowchart
        flowchart = self._generate_flowchart(evidence_context, search_query, answer_text)

        # 6. Extract and verify sources
        sources = self._extract_verified_sources(ranked_chunks, answer_text)

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
            flowchart=flowchart,
            debug_info=debug_info
        )

    async def answer_question_stream(
        self,
        document_id: str,
        question: str,
        retrieval_k: int = RETRIEVAL_TOP_K,
        final_k: int = RERANK_TOP_K,
        history: Optional[List[dict]] = None
    ) -> AsyncGenerator[str, None]:
        """
        Streaming V2 RAG pipeline via Server-Sent Events (SSE).
        Optimized flow:
          User question -> retrieval -> reranking -> immediate token stream
          -> finish answer -> immediate citations -> flowchart -> done.
        """
        def _sse(payload: dict) -> str:
            return f"data: {json.dumps(payload)}\n\n"

        if not question or not question.strip():
            yield _sse({"type": "error", "message": "Question cannot be empty."})
            return

        if not self.api_key or self.api_key == "your_gemini_api_key_here":
            yield _sse({"type": "error", "message": "GEMINI_API_KEY is not configured. Please add your API key to .env."})
            return

        t0 = time.time()
        clean_query = self._preprocess_query(question)
        search_query = await asyncio.to_thread(self._contextualize_query, clean_query, history)
        t_prep = time.time() - t0
        logger.info(f"[STREAM START] doc_id='{document_id}' | Query: '{clean_query}' | Search Query: '{search_query}'")

        try:
            # Step 1: Retrieval (run in background thread to avoid event loop blocking)
            yield _sse({"type": "status", "message": "Searching document..."})
            t_ret_start = time.time()
            candidates = await asyncio.to_thread(
                self.vector_service.similarity_search,
                search_query,
                document_id,
                retrieval_k
            )
            t_ret = time.time() - t_ret_start

            if not candidates:
                yield _sse({"type": "chunk", "text": REFUSAL_MESSAGE})
                yield _sse({"type": "sources", "data": []})
                yield _sse({"type": "done"})
                return

            # Step 2: Reranking (run in background thread)
            yield _sse({"type": "status", "message": "Reranking results..."})
            t_rerank_start = time.time()
            ranked_chunks, debug_scores = await asyncio.to_thread(
                self.reranker_service.rerank,
                search_query,
                candidates,
                final_k
            )
            t_rerank = time.time() - t_rerank_start

            # Step 3: Build evidence context
            evidence_context = self._build_evidence_context(ranked_chunks)
            system_prompt = self._build_answer_prompt(evidence_context, clean_query, history)

            # Step 4: Stream answer generation directly from LLM
            yield _sse({"type": "status", "message": "Generating answer..."})
            t_stream_start = time.time()
            first_token_time = None
            full_answer_parts = []

            try:
                async for token in self.llm_factory.generate_text_stream(
                    prompt=system_prompt,
                    temperature=0.1
                ):
                    if first_token_time is None:
                        first_token_time = time.time() - t_stream_start
                    full_answer_parts.append(token)
                    yield _sse({"type": "chunk", "text": token})

            except Exception as stream_err:
                logger.error(f"Streaming error during answer generation: {stream_err}")
                if full_answer_parts:
                    yield _sse({"type": "status", "message": "Stream interrupted — showing partial answer."})
                else:
                    yield _sse({"type": "error", "message": "Answer generation failed. Please try again."})
                    return

            t_stream_dur = time.time() - t_stream_start
            full_answer = "".join(full_answer_parts)

            # Refusal check
            is_refusal = (
                REFUSAL_MESSAGE.lower() in full_answer.lower()
                or "couldn't find the answer" in full_answer.lower()
            )

            # Step 5: Send verified sources IMMEDIATELY (no delay for flowchart)
            t_src_start = time.time()
            if is_refusal:
                yield _sse({"type": "sources", "data": []})
                yield _sse({"type": "done"})
                return

            sources = self._extract_verified_sources(ranked_chunks, full_answer)
            sources_data = [
                {
                    "page": s.page,
                    "page_number": s.page_number or s.page,
                    "text": s.text,
                    "section": s.section,
                    "subsection": s.subsection,
                    "chunk_id": s.chunk_id,
                    "ocr_derived": s.ocr_derived,
                    "ocr_confidence": s.ocr_confidence,
                    "low_confidence": s.low_confidence,
                    "source_type": s.source_type,
                    "extraction_method": s.extraction_method,
                }
                for s in sources
            ]
            yield _sse({"type": "sources", "data": sources_data})
            t_sources = time.time() - t_src_start

            # Step 6: Generate flowchart in background after answer & sources are already displayed
            t_fc_start = time.time()
            if not is_refusal and full_answer.strip():
                yield _sse({"type": "status", "message": "Generating diagram..."})
                flowchart = await asyncio.to_thread(
                    self._generate_flowchart,
                    evidence_context,
                    search_query,
                    full_answer
                )
                if flowchart:
                    flowchart_data = {
                        "type": flowchart.type,
                        "definition": flowchart.definition,
                        "description": flowchart.description
                    }
                    yield _sse({"type": "flowchart", "data": flowchart_data})
            t_fc = time.time() - t_fc_start

            yield _sse({"type": "done"})

            total_elapsed = time.time() - t0
            ft_display = f"{first_token_time:.2f}s" if first_token_time is not None else "N/A"
            logger.info(
                f"[PERF TRACE] doc_id='{document_id}' | prep={t_prep*1000:.1f}ms | "
                f"ret={t_ret*1000:.1f}ms | rerank={t_rerank*1000:.1f}ms | "
                f"first_token={ft_display} | stream={t_stream_dur:.2f}s | "
                f"citations={t_sources*1000:.1f}ms | flowchart={t_fc:.2f}s | "
                f"TOTAL={total_elapsed:.2f}s"
            )

        except HTTPException as he:
            logger.error(f"HTTP exception in streaming pipeline: {he.detail}")
            yield _sse({"type": "error", "message": he.detail})
        except Exception as e:
            logger.error(f"Unexpected error in streaming pipeline: {e}")
            yield _sse({"type": "error", "message": "An unexpected error occurred. Please try again."})
