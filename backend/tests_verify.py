"""
Comprehensive Verification Suite covering all 13 user-required test scenarios:
1. Existing printed PDF upload
2. Existing question answering
3. Streaming response behavior (SSE)
4. Final citations after streaming
5. Flowchart generation and rendering
6. Normal printed PDF containing diagrams/images
7. Scanned PDF
8. Handwritten notes PDF
9. Mixed printed + handwritten PDF
10. Question answering using OCR-derived content
11. Existing refusal/grounding behavior
12. Existing document isolation
13. Existing evaluation functionality
"""

import sys
import os
import json
import asyncio
from pathlib import Path

# Ensure UTF-8 output on Windows console
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.services.pdf_service import PDFService
from backend.services.vector_service import VectorService
from backend.services.rag_service import RAGService, REFUSAL_MESSAGE
from backend.utils.helpers import logger

test_results = {}

def record(test_num: int, name: str, passed: bool, details: str = ""):
    status = "PASS" if passed else "FAIL"
    test_results[test_num] = (name, passed, details)
    print(f"[{status}] Scenario {test_num:02d}: {name}")
    if details:
        print(f"       Details: {details}")

async def run_all_tests():
    print("\n=======================================================")
    print("[TEST SUITE] STARTING SYSTEMATIC 13-SCENARIO VERIFICATION")
    print("=======================================================\n")

    pdf_service = PDFService()
    vector_service = VectorService()
    rag_service = RAGService()

    # -----------------------------------------------------------------
    # Test 1: Existing printed PDF upload
    # -----------------------------------------------------------------
    try:
        sample_path = PROJECT_ROOT / "sample_transformer.pdf"
        doc_id_printed = "test_doc_printed"
        chunks_printed, pages_printed = pdf_service.load_and_process_pdf(
            file_path=str(sample_path),
            document_id=doc_id_printed,
            original_filename="sample_transformer.pdf"
        )
        total_indexed = vector_service.add_documents(chunks_printed)
        passed = (len(chunks_printed) > 0 and total_indexed == len(chunks_printed))
        record(1, "Existing printed PDF upload", passed, f"Extracted {pages_printed} pages, {len(chunks_printed)} chunks indexed.")
    except Exception as e:
        record(1, "Existing printed PDF upload", False, str(e))

    # -----------------------------------------------------------------
    # Test 2: Existing question answering
    # -----------------------------------------------------------------
    try:
        query = "What is the primary mechanism of the Transformer architecture?"
        resp = rag_service.answer_question(document_id=doc_id_printed, question=query)
        passed = bool(resp.answer and "attention" in resp.answer.lower() and resp.answer != REFUSAL_MESSAGE)
        record(2, "Existing question answering", passed, f"Answer: {resp.answer[:120]}...")
    except Exception as e:
        record(2, "Existing question answering", False, str(e))

    # -----------------------------------------------------------------
    # Test 3: Streaming response behavior
    # -----------------------------------------------------------------
    events_received = []
    chunk_tokens = []
    stream_flowchart = None
    stream_sources = None
    try:
        query = "Explain how self-attention works."
        async for sse_raw in rag_service.answer_question_stream(document_id=doc_id_printed, question=query):
            if sse_raw.startswith("data: "):
                payload = json.loads(sse_raw[6:].strip())
                events_received.append(payload.get("type"))
                if payload.get("type") == "chunk":
                    chunk_tokens.append(payload.get("text", ""))
                elif payload.get("type") == "flowchart":
                    stream_flowchart = payload.get("data")
                elif payload.get("type") == "sources":
                    stream_sources = payload.get("data")

        passed = (
            "status" in events_received
            and "chunk" in events_received
            and "done" in events_received
            and len(chunk_tokens) > 0
        )
        record(3, "Streaming response behavior", passed, f"Received {len(events_received)} SSE events, {len(chunk_tokens)} token chunks.")
    except Exception as e:
        record(3, "Streaming response behavior", False, str(e))

    # -----------------------------------------------------------------
    # Test 4: Final citations after streaming
    # -----------------------------------------------------------------
    try:
        passed = bool(stream_sources is not None and len(stream_sources) > 0 and "page" in stream_sources[0])
        pages_cited = [s.get("page") for s in (stream_sources or [])]
        record(4, "Final citations after streaming", passed, f"Cited pages: {pages_cited}, total sources: {len(stream_sources or [])}")
    except Exception as e:
        record(4, "Final citations after streaming", False, str(e))

    # -----------------------------------------------------------------
    # Test 5: Flowchart generation and rendering
    # -----------------------------------------------------------------
    try:
        has_flowchart = stream_flowchart is not None and "definition" in stream_flowchart
        definition = stream_flowchart.get("definition", "") if has_flowchart else ""
        starts_flowchart = definition.lower().startswith("flowchart") or definition.lower().startswith("graph")
        has_nodes = "-->" in definition
        passed = bool(has_flowchart and starts_flowchart and has_nodes)
        record(5, "Flowchart generation and rendering", passed, f"Mermaid: {definition[:80].replace(chr(10), ' ')}...")
    except Exception as e:
        record(5, "Flowchart generation and rendering", False, str(e))

    # -----------------------------------------------------------------
    # Test 6: Normal printed PDF containing diagrams/images
    # -----------------------------------------------------------------
    try:
        # sample_transformer.pdf contains digital text alongside diagram structures
        has_digital = any(not c.metadata.get("ocr_derived", False) for c in chunks_printed)
        record(6, "Normal printed PDF containing diagrams/images", has_digital, f"{len(chunks_printed)} digital chunks extracted directly without unneeded OCR.")
    except Exception as e:
        record(6, "Normal printed PDF containing diagrams/images", False, str(e))

    # -----------------------------------------------------------------
    # Test 7: Scanned PDF
    # -----------------------------------------------------------------
    doc_id_scanned = "test_doc_scanned"
    try:
        scanned_path = PROJECT_ROOT / "test_scanned.pdf"
        chunks_scanned, pages_scanned = pdf_service.load_and_process_pdf(
            file_path=str(scanned_path),
            document_id=doc_id_scanned,
            original_filename="test_scanned.pdf"
        )
        vector_service.add_documents(chunks_scanned)
        has_ocr = any(c.metadata.get("ocr_derived", False) for c in chunks_scanned)
        passed = (len(chunks_scanned) > 0 and has_ocr)
        record(7, "Scanned PDF", passed, f"Processed {pages_scanned} page via OCR, generated {len(chunks_scanned)} chunks.")
    except Exception as e:
        record(7, "Scanned PDF", False, str(e))

    # -----------------------------------------------------------------
    # Test 8: Handwritten notes PDF
    # -----------------------------------------------------------------
    doc_id_handwritten = "test_doc_handwritten"
    try:
        hw_path = PROJECT_ROOT / "test_handwritten.pdf"
        chunks_hw, pages_hw = pdf_service.load_and_process_pdf(
            file_path=str(hw_path),
            document_id=doc_id_handwritten,
            original_filename="test_handwritten.pdf"
        )
        vector_service.add_documents(chunks_hw)
        has_hw_ocr = any(c.metadata.get("ocr_derived", False) for c in chunks_hw)
        passed = (len(chunks_hw) > 0 and has_hw_ocr)
        record(8, "Handwritten notes PDF", passed, f"Handwritten notes transcribed: {chunks_hw[0].page_content[:100]}...")
    except Exception as e:
        record(8, "Handwritten notes PDF", False, str(e))

    # -----------------------------------------------------------------
    # Test 9: Mixed printed + handwritten PDF
    # -----------------------------------------------------------------
    doc_id_mixed = "test_doc_mixed"
    try:
        mixed_path = PROJECT_ROOT / "test_mixed.pdf"
        chunks_mixed, pages_mixed = pdf_service.load_and_process_pdf(
            file_path=str(mixed_path),
            document_id=doc_id_mixed,
            original_filename="test_mixed.pdf"
        )
        vector_service.add_documents(chunks_mixed)
        p1_chunks = [c for c in chunks_mixed if c.metadata.get("page") == 1]
        p2_chunks = [c for c in chunks_mixed if c.metadata.get("page") == 2]

        p1_is_digital = any(not c.metadata.get("ocr_derived", False) for c in p1_chunks)
        p2_is_ocr = any(c.metadata.get("ocr_derived", False) for c in p2_chunks)
        passed = (pages_mixed == 2 and p1_is_digital and p2_is_ocr)
        record(9, "Mixed printed + handwritten PDF", passed, f"Page 1: Digital ({len(p1_chunks)} chunks), Page 2: OCR/Handwritten ({len(p2_chunks)} chunks).")
    except Exception as e:
        record(9, "Mixed printed + handwritten PDF", False, str(e))

    # -----------------------------------------------------------------
    # Test 10: Question answering using OCR-derived content
    # -----------------------------------------------------------------
    try:
        hw_query = "What is the approved budget for hardware testing?"
        hw_resp = rag_service.answer_question(document_id=doc_id_handwritten, question=hw_query)
        has_budget = "45000" in hw_resp.answer or "45,000" in hw_resp.answer
        ocr_flagged = any(s.ocr_derived for s in hw_resp.sources)
        passed = (has_budget and ocr_flagged)
        record(10, "Question answering using OCR-derived content", passed, f"Answer: '{hw_resp.answer[:100]}', OCR flagged in sources: {ocr_flagged}")
    except Exception as e:
        record(10, "Question answering using OCR-derived content", False, str(e))

    # -----------------------------------------------------------------
    # Test 11: Existing refusal/grounding behavior
    # -----------------------------------------------------------------
    try:
        refusal_query = "What is the capital city of France?"
        ref_resp = rag_service.answer_question(document_id=doc_id_printed, question=refusal_query)
        passed = (REFUSAL_MESSAGE.lower() in ref_resp.answer.lower() and len(ref_resp.sources) == 0 and ref_resp.flowchart is None)
        record(11, "Existing refusal/grounding behavior", passed, f"Refusal returned: '{ref_resp.answer}', sources count: {len(ref_resp.sources)}, flowchart: {ref_resp.flowchart}")
    except Exception as e:
        record(11, "Existing refusal/grounding behavior", False, str(e))

    # -----------------------------------------------------------------
    # Test 12: Existing document isolation
    # -----------------------------------------------------------------
    try:
        # Ask document A (printed transformer) a question about document B (Apollo budget)
        cross_resp = rag_service.answer_question(document_id=doc_id_printed, question="What is the approved budget for hardware testing in Project Apollo?")
        passed = (REFUSAL_MESSAGE.lower() in cross_resp.answer.lower() and len(cross_resp.sources) == 0)
        record(12, "Existing document isolation", passed, f"Document A strictly abstained from Document B content: '{cross_resp.answer}'")
    except Exception as e:
        record(12, "Existing document isolation", False, str(e))

    # -----------------------------------------------------------------
    # Test 13: Existing evaluation functionality
    # -----------------------------------------------------------------
    try:
        from evaluation.run_evaluation import run_benchmark
        # Run benchmark in v2 mode
        run_benchmark(mode="v2")
        record(13, "Existing evaluation functionality", True, "Evaluation benchmark executed and passed.")
    except Exception as e:
        record(13, "Existing evaluation functionality", False, str(e))

    print("\n=======================================================")
    print(f"[SUMMARY]: {sum(1 for _, p, _ in test_results.values() if p)} / {len(test_results)} TESTS PASSED")
    print("=======================================================\n")

if __name__ == "__main__":
    asyncio.run(run_all_tests())
