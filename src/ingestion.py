"""
Step 1: Ingestion
------------------
Reads resumes from disk (PDF, DOCX, TXT) and returns raw text plus
light metadata. Designed to be run over a whole directory of up to
~1000 files, in parallel, without loading everything into memory at once.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


@dataclass
class RawResume:
    resume_id: str          # stable id, e.g. filename stem
    file_path: str
    raw_text: str
    parse_error: str | None = None


def _read_pdf(path: Path) -> str:
    import pdfplumber

    text_chunks = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text_chunks.append(page_text)
    return "\n".join(text_chunks)


def _read_docx(path: Path) -> str:
    import docx

    document = docx.Document(str(path))
    paragraphs = [p.text for p in document.paragraphs]
    # Also pull text out of tables (skills tables are common in resumes)
    for table in document.tables:
        for row in table.rows:
            paragraphs.append(" | ".join(cell.text for cell in row.cells))
    return "\n".join(paragraphs)


def _read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


_READERS = {
    ".pdf": _read_pdf,
    ".docx": _read_docx,
    ".txt": _read_txt,
}


def parse_single_file(path: Path) -> RawResume:
    resume_id = path.stem
    ext = path.suffix.lower()
    if ext not in _READERS:
        return RawResume(resume_id, str(path), "", f"Unsupported file type: {ext}")
    try:
        text = _READERS[ext](path)
        if not text.strip():
            return RawResume(resume_id, str(path), "", "Empty or unreadable content")
        return RawResume(resume_id, str(path), text)
    except Exception as exc:  # noqa: BLE001 - we want to capture and continue
        logger.warning("Failed to parse %s: %s", path, exc)
        return RawResume(resume_id, str(path), "", str(exc))


def list_resume_files(directory: str | Path) -> list[Path]:
    directory = Path(directory)
    return [
        p for p in sorted(directory.rglob("*"))
        if p.suffix.lower() in SUPPORTED_EXTENSIONS and p.is_file()
    ]


def ingest_directory(directory: str | Path, max_workers: int = 8) -> list[RawResume]:
    """
    Parse every supported resume file in `directory` concurrently
    (I/O-bound work, so threads are appropriate here).
    """
    files = list_resume_files(directory)
    results: list[RawResume] = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(parse_single_file, f): f for f in files}
        for future in as_completed(futures):
            results.append(future.result())

    ok = sum(1 for r in results if not r.parse_error)
    logger.info("Ingested %d/%d resumes successfully", ok, len(results))
    return results


def batched(items: list, batch_size: int) -> Iterable[list]:
    """Yield successive batches of `batch_size` from `items`."""
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]
