import re
import logging
import pytesseract

from pathlib import Path
from datetime import datetime
from collections import Counter

from pdf2image import convert_from_path
from pypdf import PdfReader

from engine.config import config

logger = logging.getLogger(__name__)

# ============ Process PDF ============
def extract_text_from_pdf(pdf_path: str, max_pages: int = 5) -> str:
    """Extract text and use OCR for scanned PDFs"""
    try:
        reader = PdfReader(pdf_path)
        text = ""

        # Try text extraction first
        for page in reader.pages[:max_pages]:
            text += page.extract_text()

        # If minimal text, use OCR (likely scanned)
        if len(text.strip()) < 100:
            logger.info(f"Scanned PDF detected: {pdf_path}, using OCR")
            images = convert_from_path(pdf_path, first_page=1, last_page=min(max_pages, len(reader.pages)))
            for image in images:
                text += pytesseract.image_to_string(image)

        return text[:config.MAX_CHARACTERS]  # Cap at MAX_CHARACTERS chars for LLM
    except Exception as e:
        logger.error(f"Text extraction failed for {pdf_path}: {e}")
        raise

def extract_pages_from_pdf(pdf_path: str) -> list[str]:
    """Extract text page by page from a PDF."""
    try:
        reader = PdfReader(pdf_path)
        pages: list[str] = []

        # Try text extraction first
        for page in reader.pages[:config.MAX_PAGES]:
            pages.append(page.extract_text() or "")

        # If minimal text, use OCR (likely scanned)
        if len("".join(pages).strip()) < 100:
            logger.info(f"Scanned PDF detected: {pdf_path}, using OCR")
            images = convert_from_path(
                pdf_path,
                first_page=1,
                last_page=min(config.MAX_PAGES, len(reader.pages)),
            )
            pages = [pytesseract.image_to_string(image) for image in images]

        # Drop lines that repeat across many pages (headers/footers)
        return pages
    except Exception as e:
        logger.error(f"Text extraction failed for {pdf_path}: {e}")
        raise

def clean_pages(pages: list[str]) -> list[str]:
    # Drop lines that repeat across many pages (headers/footers)
    line_counts = Counter(
        line.strip() for page in pages for line in set(page.splitlines()) if line.strip()
    )
    # repeated = {l for l, c in line_counts.items() if c > len(pages) * 0.5 and len(pages) > 2}
    repeated = {l for l, c in line_counts.items() if c > len(pages) * 0.5 and len(pages) > 1}

    kept = []
    for page in pages:
        # print(page)
        # print("------------------------------------------------------------------------------------")
        for line in page.splitlines():
            s = line.strip()
            if not s or s in repeated or re.fullmatch(r"(page\s*)?\d+(\s*of\s*\d+)?", s, re.I):
                continue
            kept.append(re.sub(r"\s+", " ", s))

    return kept

# ============ Extract Text ============
def extract_text_from_file(file_path: str) -> str:
    """Extract text from a single file

    Args:
        file_path: Path to file
    """
    start_time = datetime.now()

    try:
        logger.info(f"Extracting text from: {file_path}")

        # Extract text
        pages = extract_pages_from_pdf(file_path)
        cleaned_pages = clean_pages(pages)
        text = "\n".join(cleaned_pages)

        processing_time = (datetime.now() - start_time).total_seconds()

        logger.info(f"Extracted text from {file_path} in {processing_time:.2f}s")
        return text

    except Exception as e:
        logger.error(f"Error extracting text from {file_path}: {e}")
        return ""

# ============ Save Text ============
def save_to_file(file_path: str, text: str, suffix: str = ""):
    # Save extracted text next to the PDF, e.g. report.pdf -> report.txt
    path = Path(file_path)
    stem = path.stem

    if suffix.strip():  # skip blank or whitespace-only suffixes
        stem = f"{stem}{suffix}"

    txt_path = path.with_name(f"{stem}.txt")

    try:
        txt_path.write_text(text, encoding="utf-8")
        logger.info(f"Saved extracted text to {txt_path}")
    except OSError as e:
        logger.warning(f"Could not save text to {txt_path}: {e}")
