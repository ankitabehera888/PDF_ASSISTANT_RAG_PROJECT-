import re
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from backend.config import CHUNK_SIZE, CHUNK_OVERLAP
from backend.utils.helpers import clean_text, logger
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
        Metadata per chunk: { document_id, source, page, section, subsection, chunk_id }
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

        for i, doc in enumerate(raw_docs):
            cleaned = clean_text(doc.page_content)
            if not cleaned:
                continue

            total_text_length += len(cleaned)
            page_num = doc.metadata.get("page", i) + 1
            if isinstance(page_num, int) and page_num == i:
                page_num = i + 1

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
                    "section": current_section,
                    "subsection": current_subsection
                }
            )
            processed_pages.append(new_doc)

        if not processed_pages or total_text_length == 0:
            raise HTTPException(status_code=400, detail="No readable text could be extracted from this PDF.")

        logger.info(f"Extracted {len(processed_pages)} pages with text. Splitting into semantic chunks...")
        raw_chunks = self.text_splitter.split_documents(processed_pages)

        # Enrich chunks with explicit chunk_ids and section context headers
        final_chunks: List[Document] = []
        for idx, chunk in enumerate(raw_chunks):
            page_num = chunk.metadata.get("page", 1)
            section = chunk.metadata.get("section")
            subsection = chunk.metadata.get("subsection")

            chunk_id = f"{document_id}_p{page_num}_c{idx+1:02d}"

            # Format chunk content with structural context header
            header_parts = []
            if section:
                header_parts.append(f"Section: {section}")
            if subsection:
                header_parts.append(f"Subsection: {subsection}")
            header_parts.append(f"Page: {page_num}")

            header_prefix = "\n".join(header_parts) + "\n\n"
            enriched_content = header_prefix + chunk.page_content

            final_doc = Document(
                page_content=enriched_content,
                metadata={
                    "document_id": document_id,
                    "source": original_filename,
                    "page": int(page_num),
                    "section": section if section else "General",
                    "subsection": subsection if subsection else "",
                    "chunk_id": chunk_id,
                    "raw_text": chunk.page_content
                }
            )
            final_chunks.append(final_doc)

        logger.info(f"Generated {len(final_chunks)} section-aware chunks for document {document_id}")
        return final_chunks, total_pages
