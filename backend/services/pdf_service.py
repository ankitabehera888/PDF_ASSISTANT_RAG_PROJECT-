import re
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from backend.config import CHUNK_SIZE, CHUNK_OVERLAP
from backend.utils.helpers import clean_text, logger
from backend.services.ocr_service import OCRService, page_has_digital_text
from fastapi import HTTPException

# Common section header detection patterns
HEADING_PATTERNS = [
    r'^(?:SECTION|Section|ARTICLE|Article|CHAPTER|Chapter|PART|Part|POLICY|Policy)\s+[\d\w\.:]+.*$',
    r'^\d+(\.\d+)*\s+[A-Z].*$',
    r'^[0-9]+\.\s+[A-Z].*$',
    r'^[A-Z\s]{4,60}$'
]

class PDFService:
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.ocr_service = OCRService()

    def _detect_heading(self, line: str) -> Optional[str]:
        """Detect if a text line is a section heading."""
        line_str = line.strip()
        if not line_str or len(line_str) < 3 or len(line_str) > 100:
            return None
            
        for pattern in HEADING_PATTERNS:
            if re.match(pattern, line_str):
                # Avoid matching simple sentences ending with a period
                if line_str.endswith(".") and not re.match(r'^\d+(\.\d+)*\s+', line_str):
                    continue
                return line_str
        return None

    def load_and_process_pdf(self, file_path: str, document_id: str, original_filename: str) -> Tuple[List[Document], int]:
        """
        Loads PDF, detects sections/headings, splits into semantic chunks, and attaches section metadata.
        For pages with insufficient extractable text (scanned/handwritten/image), falls back to Vision OCR.
        Metadata per chunk:
        { document_id, source, page, page_number, section, subsection, chunk_id, ocr_derived, ocr_confidence, source_type, extraction_method }
        Returns: (chunks: List[Document], total_pages: int)
        """
        logger.info(f"Extracting pages from PDF: {original_filename} (Doc ID: {document_id})")
        
        try:
            loader = PyPDFLoader(file_path)
            raw_docs = loader.load()
        except Exception as e:
            logger.error(f"Failed to load PDF file: {e}")
            raise HTTPException(status_code=400, detail="Failed to parse PDF file. Ensure it is a valid PDF.")

        if not raw_docs:
            raise HTTPException(status_code=400, detail="No readable text could be extracted from this PDF.")

        total_pages = len(raw_docs)
        processed_pages: List[Document] = []
        total_text_length = 0

        current_section: Optional[str] = None
        current_subsection: Optional[str] = None

        ocr_available = self.ocr_service.is_available()

        for i, doc in enumerate(raw_docs):
            cleaned = clean_text(doc.page_content)
            page_num = doc.metadata.get("page", i) + 1
            if isinstance(page_num, int) and page_num == i:
                page_num = i + 1

            digital_len = len(cleaned.strip())
            has_digital = page_has_digital_text(cleaned)

            ocr_derived = False
            ocr_confidence = 1.0
            low_confidence = False
            source_type = "printed"
            extraction_method = "digital_text"

            # Check if page needs OCR (scanned, image, or handwritten notes)
            if not has_digital:
                if ocr_available:
                    logger.info(
                        f"Page {page_num}: detected as image/handwritten page "
                        f"(digital text length={digital_len} chars). Triggering Vision OCR..."
                    )
                    ocr_text, ocr_conf, low_conf, method = self.ocr_service.extract_page_via_ocr(
                        pdf_path=file_path,
                        page_number=int(page_num)
                    )
                    if ocr_text.strip():
                        cleaned = ocr_text
                        ocr_derived = True
                        ocr_confidence = round(ocr_conf, 1)
                        low_confidence = low_conf
                        source_type = "handwritten_ocr"
                        extraction_method = method if method != "none" else "vision_ocr"
                        logger.info(
                            f"Page {page_num}: OCR SUCCESS | extracted_length={len(cleaned)} chars | "
                            f"confidence={ocr_conf:.1f}% | low_conf={low_conf} | method={extraction_method}"
                        )
                    else:
                        logger.warning(f"Page {page_num}: OCR returned no text. Page will be skipped.")
                        continue
                else:
                    logger.warning(
                        f"Page {page_num}: no digital text and OCR unavailable. Page will be skipped."
                    )
                    continue
            else:
                if not cleaned:
                    continue

            total_text_length += len(cleaned)

            logger.info(
                f"[PAGE TRACE] Page {page_num}/{total_pages} processed | "
                f"digital_text_len={digital_len} | "
                f"detected_type={'printed_text' if has_digital else 'image_or_handwritten'} | "
                f"ocr_triggered={not has_digital} | "
                f"final_text_len={len(cleaned)} | "
                f"source_type={source_type} | extraction_method={extraction_method}"
            )

            # Scan lines on this page for section headings
            lines = cleaned.split("\n")
            for line in lines:
                heading = self._detect_heading(line)
                if heading:
                    if re.match(r'^\d+\.\d+\.\d+', heading) or "subsection" in heading.lower():
                        current_subsection = heading
                    else:
                        current_section = heading
                        current_subsection = None

            new_doc = Document(
                page_content=cleaned,
                metadata={
                    "document_id": document_id,
                    "source": original_filename,
                    "page": int(page_num),
                    "page_number": int(page_num),
                    "section": current_section,
                    "subsection": current_subsection,
                    "ocr_derived": ocr_derived,
                    "ocr_confidence": ocr_confidence,
                    "low_confidence": low_confidence,
                    "source_type": source_type,
                    "extraction_method": extraction_method,
                }
            )
            processed_pages.append(new_doc)

        if not processed_pages or total_text_length == 0:
            if not ocr_available:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "No readable text could be extracted from this PDF. "
                        "This may be a scanned or image-based PDF. "
                        "OCR processing is not currently available."
                    )
                )
            raise HTTPException(status_code=400, detail="No readable text could be extracted from this PDF.")

        logger.info(f"Extracted {len(processed_pages)} pages with text. Splitting into semantic chunks...")
        raw_chunks = self.text_splitter.split_documents(processed_pages)

        # Enrich chunks with explicit chunk_ids and section context headers
        final_chunks: List[Document] = []
        for idx, chunk in enumerate(raw_chunks):
            page_num = chunk.metadata.get("page", 1)
            section = chunk.metadata.get("section")
            subsection = chunk.metadata.get("subsection")
            ocr_derived = chunk.metadata.get("ocr_derived", False)
            ocr_confidence = chunk.metadata.get("ocr_confidence", 1.0)
            low_confidence = chunk.metadata.get("low_confidence", False)
            source_type = chunk.metadata.get("source_type", "printed")
            extraction_method = chunk.metadata.get("extraction_method", "digital_text")

            chunk_id = f"{document_id}_p{page_num}_c{idx+1:02d}"

            # Format chunk content with structural context header
            header_parts = []
            if section:
                header_parts.append(f"Section: {section}")
            if subsection:
                header_parts.append(f"Subsection: {subsection}")
            header_parts.append(f"Page: {page_num}")
            if ocr_derived:
                conf_label = f"low-confidence ({ocr_confidence:.0f}%)" if low_confidence else f"{ocr_confidence:.0f}%"
                header_parts.append(f"Source: OCR/Handwritten (confidence: {conf_label})")

            header_prefix = "\n".join(header_parts) + "\n\n"
            enriched_content = header_prefix + chunk.page_content

            final_doc = Document(
                page_content=enriched_content,
                metadata={
                    "document_id": document_id,
                    "source": original_filename,
                    "page": int(page_num),
                    "page_number": int(page_num),
                    "section": section if section else "General",
                    "subsection": subsection if subsection else "",
                    "chunk_id": chunk_id,
                    "raw_text": chunk.page_content,
                    "ocr_derived": ocr_derived,
                    "ocr_confidence": ocr_confidence,
                    "low_confidence": low_confidence,
                    "source_type": source_type,
                    "extraction_method": extraction_method,
                }
            )
            final_chunks.append(final_doc)

        ocr_count = sum(1 for c in final_chunks if c.metadata.get("ocr_derived"))
        printed_count = len(final_chunks) - ocr_count
        logger.info(
            f"[CHUNK TRACE] Document {document_id}: produced {len(final_chunks)} total chunks "
            f"(printed={printed_count}, ocr_derived={ocr_count})"
        )
        for c in final_chunks:
            if c.metadata.get("ocr_derived"):
                logger.info(f"[OCR CHUNK SAMPLE] ID={c.metadata.get('chunk_id')} | metadata={c.metadata}")
                break

        return final_chunks, total_pages
