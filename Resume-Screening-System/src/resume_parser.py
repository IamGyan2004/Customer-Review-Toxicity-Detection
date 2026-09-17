from io import BytesIO
from pathlib import Path


def extract_text(file_bytes: bytes, filename: str) -> str:
    """Extract text from PDF, DOCX, or plain-text resume uploads."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(file_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if suffix == ".docx":
        from docx import Document

        document = Document(BytesIO(file_bytes))
        return "\n".join(paragraph.text for paragraph in document.paragraphs)
    if suffix in {".txt", ".md"}:
        return file_bytes.decode("utf-8", errors="replace")
    raise ValueError("Unsupported file type. Upload a PDF, DOCX, or TXT resume.")