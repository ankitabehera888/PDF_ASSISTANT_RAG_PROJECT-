"""
OCR Service: Handles extraction of text from scanned, image-based, photographed,
and handwritten PDF pages using Gemini Vision with pypdfium2 page rendering,
featuring multi-model failover (gemini-2.5-flash-lite -> gemini-flash-lite-latest -> gemini-1.5-flash)
and PyTesseract fallback.
"""

import io
import re
import time
import base64
from typing import Optional, Tuple, Dict, Any, List
from pathlib import Path
from backend.utils.helpers import logger
from backend.config import GEMINI_API_KEY, PRIMARY_LLM_MODEL, FALLBACK_LLM_MODEL

# Confidence threshold: pages below this are marked as low-confidence
LOW_CONFIDENCE_THRESHOLD = 40.0
# Min extractable characters & words to consider a page as having usable digital text
DIGITAL_TEXT_MIN_CHARS = 40
DIGITAL_TEXT_MIN_WORDS = 8

# Check pypdfium2 availability (bundled PDFium engine, no Poppler needed)
try:
    import pypdfium2 as pdfium
    PDFIUM_AVAILABLE = True
except ImportError:
    PDFIUM_AVAILABLE = False
    logger.warning("pypdfium2 not installed; PDF page rasterization may rely on pdf2image.")

# Check google.genai availability
try:
    from google import genai
    from google.genai import types
    GENAI_SDK_AVAILABLE = True
except ImportError:
    GENAI_SDK_AVAILABLE = False

# Check pytesseract + pdf2image availability
try:
    import pytesseract
    from pdf2image import convert_from_path
    from PIL import Image
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False


def page_has_digital_text(page_text: str) -> bool:
    """
    Returns True if the page already has usable digital/selectable text.
    Distinguishes normal printed/selectable pages from scanned/handwritten/image pages.
    A page has usable digital text if:
    1. It contains at least 40 alphanumeric characters.
    2. It contains at least 8 valid words.
    3. Alphanumeric density is healthy (>= 25% of characters).
    """
    if not page_text or not page_text.strip():
        return False

    alpha_chars = re.sub(r'[^a-zA-Z0-9]', '', page_text)
    if len(alpha_chars) < DIGITAL_TEXT_MIN_CHARS:
        return False

    words = [w for w in re.split(r'\s+', page_text.strip()) if len(w) >= 2 and any(c.isalnum() for c in w)]
    if len(words) < DIGITAL_TEXT_MIN_WORDS:
        return False

    total_len = len(page_text.strip())
    if total_len > 0 and (len(alpha_chars) / total_len) < 0.25:
        return False

    return True


VISION_OCR_PROMPT = (
    "Transcribe the handwritten/printed content from this page accurately. "
    "Preserve headings, lists, equations, terminology and ordering. "
    "Do not summarize. Do not invent missing text. "
    "If a section is unreadable, indicate uncertainty.\n\n"
    "Preserve structure: use line breaks, bullet points, and numbered lists as shown on the page.\n\n"
    "At the very end of your response on a new line, output a confidence assessment tag in this exact format:\n"
    "[CONFIDENCE: X]\n"
    "where X is an integer between 0 and 100 estimating your transcription clarity (e.g., 95 for clear print/clean handwriting, 60 for messy cursive, 30 for severely degraded/faded text).\n"
    "Do NOT include commentary, greetings, or conversational remarks outside the transcribed content and confidence tag."
)


class OCRService:
    """
    Provides OCR / Vision-based text extraction for scanned, image-based,
    or handwritten PDF pages.

    Architecture:
    1. Primary: Renders page to high-res image via pypdfium2.
    2. Multimodal Vision: Uses Gemini Vision with multi-model fallback chain
       (PRIMARY_LLM_MODEL -> FALLBACK_LLM_MODEL -> gemini-flash-lite-latest -> gemini-1.5-flash).
    3. Fallback: PyTesseract OCR if Tesseract binary is detected.
    4. Graceful degradation: Returns empty string and low_confidence=True if
       all methods fail or text is completely unreadable.
    """

    def __init__(self):
        self.api_key = GEMINI_API_KEY
        self.primary_model = PRIMARY_LLM_MODEL
        self.fallback_model = FALLBACK_LLM_MODEL
        self._genai_client = None
        self._exhausted_models: Dict[str, float] = {}

        # Build candidate vision models chain with deduplication
        candidate_models = [
            self.primary_model,
            self.fallback_model,
            "gemini-flash-lite-latest",
            "gemini-2.5-flash-lite",
            "gemini-1.5-flash",
            "gemini-2.0-flash",
            "gemini-2.5-flash"
        ]
        self.vision_models: List[str] = []
        for m in candidate_models:
            if m and m not in self.vision_models:
                self.vision_models.append(m)

    def _get_genai_client(self):
        """Lazy-initialize Google GenAI client."""
        if self._genai_client is None and self.api_key and GENAI_SDK_AVAILABLE:
            self._genai_client = genai.Client(api_key=self.api_key)
        return self._genai_client

    def is_available(self) -> bool:
        """Check if at least one OCR/vision mechanism is available."""
        has_vision = bool(self.api_key and PDFIUM_AVAILABLE)
        has_tesseract = TESSERACT_AVAILABLE
        return has_vision or has_tesseract

    def _render_page_image(self, pdf_path: str, page_number: int, scale: float = 2.0):
        """
        Renders a 1-indexed PDF page to a PIL Image using pypdfium2 or pdf2image.
        """
        if PDFIUM_AVAILABLE:
            try:
                pdf = pdfium.PdfDocument(pdf_path)
                page_idx = page_number - 1
                if 0 <= page_idx < len(pdf):
                    page = pdf[page_idx]
                    pil_image = page.render(scale=scale).to_pil()
                    return pil_image
            except Exception as e:
                logger.warning(f"pypdfium2 rendering failed for page {page_number}: {e}")

        if TESSERACT_AVAILABLE:
            try:
                images = convert_from_path(
                    pdf_path,
                    dpi=int(150 * scale),
                    first_page=page_number,
                    last_page=page_number
                )
                if images:
                    return images[0]
            except Exception as e:
                logger.warning(f"pdf2image rendering failed for page {page_number}: {e}")

        return None

    def _call_gemini_vision(self, img_bytes: bytes, model_name: str) -> Optional[str]:
        """Call Gemini Vision model with raw JPEG image bytes."""
        # Method A: Google GenAI SDK (direct, fastest)
        client = self._get_genai_client()
        if client:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=[
                        types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                        VISION_OCR_PROMPT
                    ]
                )
                if response and response.text:
                    return response.text
            except Exception as sdk_err:
                err_str = str(sdk_err)
                if "RESOURCE_EXHAUSTED" in err_str or "429" in err_str:
                    logger.warning(f"Vision model '{model_name}' quota exhausted (429) — cooling down for 10m: {err_str[:120]}")
                    self._exhausted_models[model_name] = time.time() + 600
                else:
                    logger.warning(f"Vision model '{model_name}' SDK call failed: {sdk_err}")
                return None

        # Method B: LangChain ChatGoogleGenerativeAI fallback
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            from langchain_core.messages import HumanMessage

            b64_str = base64.b64encode(img_bytes).decode("utf-8")
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=self.api_key,
                temperature=0.0,
                max_retries=1
            )
            msg = HumanMessage(
                content=[
                    {"type": "text", "text": VISION_OCR_PROMPT},
                    {"type": "image_url", "image_url": f"data:image/jpeg;base64,{b64_str}"}
                ]
            )
            resp = llm.invoke([msg])
            if resp and resp.content:
                return resp.content
        except Exception as lc_err:
            err_str = str(lc_err)
            if "RESOURCE_EXHAUSTED" in err_str or "429" in err_str:
                self._exhausted_models[model_name] = time.time() + 600
            logger.warning(f"LangChain vision call to '{model_name}' failed: {lc_err}")

        return None

    def _extract_via_vision_chain(self, pil_image, page_number: int) -> Tuple[str, float, bool, str]:
        """
        Extract text from page image using Gemini Vision with multi-model failover.
        Returns: (extracted_text, confidence, is_low_conf, model_used)
        """
        # Convert PIL image to JPEG bytes
        buf = io.BytesIO()
        if pil_image.mode in ('RGBA', 'LA') or (pil_image.mode == 'P' and 'transparency' in pil_image.info):
            rgb_img = pil_image.convert('RGB')
            rgb_img.save(buf, format='JPEG', quality=90)
        else:
            pil_image.save(buf, format='JPEG', quality=90)
        img_bytes = buf.getvalue()

        # Try models in chain
        now = time.time()
        for model in self.vision_models:
            # Skip model if currently in cooldown
            if model in self._exhausted_models and now < self._exhausted_models[model]:
                continue

            logger.info(f"Page {page_number}: attempting Vision OCR with model '{model}'...")
            raw_text = self._call_gemini_vision(img_bytes, model)
            if raw_text and raw_text.strip():
                confidence = 85.0
                conf_match = re.search(r'\[CONFIDENCE:\s*(\d+)\]', raw_text, flags=re.IGNORECASE)
                if conf_match:
                    confidence = float(conf_match.group(1))
                    clean_extracted = re.sub(r'\[CONFIDENCE:\s*\d+\]', '', raw_text).strip()
                else:
                    clean_extracted = raw_text.strip()

                is_low_conf = confidence < LOW_CONFIDENCE_THRESHOLD
                logger.info(
                    f"Page {page_number}: Vision OCR SUCCESS via '{model}' "
                    f"({len(clean_extracted)} chars, confidence={confidence:.1f}%, low_conf={is_low_conf})"
                )
                return clean_extracted, confidence, is_low_conf, model

        logger.warning(f"Page {page_number}: All Gemini vision models exhausted or failed.")
        return "", 0.0, True, "none"

    def _extract_via_tesseract(self, pil_image, page_number: int) -> Tuple[str, float, bool]:
        """Fallback extraction using PyTesseract."""
        if not TESSERACT_AVAILABLE:
            return "", 0.0, True

        try:
            ocr_data = pytesseract.image_to_data(
                pil_image,
                output_type=pytesseract.Output.DICT,
                config='--oem 3 --psm 6'
            )

            words = []
            confidences = []
            for i, word_text in enumerate(ocr_data.get('text', [])):
                conf = int(ocr_data['conf'][i])
                if conf > 0 and word_text.strip():
                    words.append(word_text)
                    confidences.append(conf)

            if not words:
                return "", 0.0, True

            avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
            is_low_conf = avg_conf < LOW_CONFIDENCE_THRESHOLD

            extracted = " ".join(words)
            logger.info(f"Page {page_number}: PyTesseract OCR success ({len(words)} words, confidence={avg_conf:.1f}%)")
            return extracted, avg_conf, is_low_conf

        except Exception as e:
            logger.warning(f"PyTesseract extraction failed on page {page_number}: {e}")
            return "", 0.0, True

    def extract_page_via_ocr(
        self,
        pdf_path: str,
        page_number: int,
        scale: float = 2.0
    ) -> Tuple[str, float, bool, str]:
        """
        Convert a single PDF page to image and run OCR / Vision extraction.

        Returns:
            (text: str, confidence: float, low_confidence: bool, method: str)
            where method is 'vision_ocr' or 'tesseract_ocr' or 'none'
        """
        pil_image = self._render_page_image(pdf_path, page_number, scale=scale)
        if pil_image is None:
            logger.warning(f"Could not render image for page {page_number}; OCR cannot proceed.")
            return "", 0.0, True, "none"

        # 1. Try Gemini Vision multi-model chain first
        if self.api_key:
            text, conf, low_conf, model_used = self._extract_via_vision_chain(pil_image, page_number)
            if text.strip():
                return text, conf, low_conf, "vision_ocr"

        # 2. Try PyTesseract fallback
        if TESSERACT_AVAILABLE:
            logger.info(f"Page {page_number}: Attempting PyTesseract fallback...")
            text, conf, low_conf = self._extract_via_tesseract(pil_image, page_number)
            if text.strip():
                return text, conf, low_conf, "tesseract_ocr"

        return "", 0.0, True, "none"
