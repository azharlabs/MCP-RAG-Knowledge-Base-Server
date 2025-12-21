from __future__ import annotations

import csv
import json
import re
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from typing import List

import pdfplumber


class _HTMLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts: List[str] = []

    def handle_data(self, data):
        self.parts.append(data)

    def get_text(self) -> str:
        return " ".join(self.parts)


class EmbedAnything:
    """
    Lightweight extractor modeled after the EmbedAnything interface.

    The implementation focuses on text extraction for common knowledge
    base sources without performing network calls.
    """

    SUPPORTED_EXTENSIONS = {
        ".pdf": "PDF",
        ".txt": "Plain text",
        ".md": "Markdown",
        ".html": "HTML",
        ".htm": "HTML",
        ".csv": "CSV",
        ".json": "JSON",
        ".docx": "Word (.docx)",
        ".doc": "Word (.doc)",
        ".pptx": "PowerPoint",
        ".ppt": "PowerPoint",
    }

    def supported_formats(self) -> List[str]:
        return list(dict.fromkeys(self.SUPPORTED_EXTENSIONS.values()))

    def extract_text(self, file_path: str) -> str:
        path = Path(file_path)
        ext = path.suffix.lower()

        if ext in {".txt", ".md"}:
            return path.read_text(encoding="utf-8", errors="ignore")

        if ext in {".html", ".htm"}:
            stripper = _HTMLStripper()
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                stripper.feed(f.read())
            return stripper.get_text()

        if ext == ".csv":
            rows: List[str] = []
            with open(path, newline="", encoding="utf-8", errors="ignore") as f:
                for row in csv.reader(f):
                    rows.append(", ".join(row))
            return "\n".join(rows)

        if ext == ".json":
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                data = json.load(f)
            return self._flatten_json(data)

        if ext == ".pdf":
            return self._extract_pdf(path)

        if ext in {".doc", ".docx"}:
            return self._extract_docx(path)

        if ext in {".ppt", ".pptx"}:
            return self._extract_pptx(path)

        # Fallback to raw text
        return path.read_text(encoding="utf-8", errors="ignore")

    def _extract_pdf(self, path: Path) -> str:
        text_parts: List[str] = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text_parts.append((page.extract_text() or ""))
        return "\n".join(text_parts)

    def _extract_docx(self, path: Path) -> str:
        if not zipfile.is_zipfile(path):
            return path.read_text(encoding="utf-8", errors="ignore")
        try:
            with zipfile.ZipFile(path) as z:
                xml = z.read("word/document.xml").decode("utf-8", errors="ignore")
            texts = re.findall(r"<w:t[^>]*>(.*?)</w:t>", xml)
            return "\n".join(texts)
        except KeyError:
            return ""

    def _extract_pptx(self, path: Path) -> str:
        if not zipfile.is_zipfile(path):
            return ""
        slide_text: List[str] = []
        try:
            with zipfile.ZipFile(path) as z:
                for name in z.namelist():
                    if name.startswith("ppt/slides/slide") and name.endswith(".xml"):
                        xml = z.read(name).decode("utf-8", errors="ignore")
                        slide_text.extend(re.findall(r"<a:t[^>]*>(.*?)</a:t>", xml))
        except KeyError:
            return ""
        return "\n".join(slide_text)

    def _flatten_json(self, data) -> str:
        parts: List[str] = []

        def _walk(obj):
            if isinstance(obj, dict):
                for value in obj.values():
                    _walk(value)
            elif isinstance(obj, list):
                for item in obj:
                    _walk(item)
            else:
                parts.append(str(obj))

        _walk(data)
        return "\n".join(parts)


__all__ = ["EmbedAnything"]
