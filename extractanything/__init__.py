from __future__ import annotations

from typing import List

from markitdown import MarkItDown


class ExtractAnything:
    """
    Convert documents to Markdown using Microsoft MarkItDown.

    MarkItDown supports a broad set of file types and keeps output
    consistently in Markdown for downstream chunking.
    """

    def __init__(self):
        self._converter = MarkItDown()

    def supported_formats(self) -> List[str]:
        return ["MarkItDown (many formats)"]

    def extract_text(self, file_path: str) -> str:
        result = self._converter.convert(file_path)
        return self._extract_markdown(result)

    def _extract_markdown(self, result) -> str:
        if isinstance(result, str):
            return result.strip()
        for attr in ("text_content", "markdown", "content", "text"):
            value = getattr(result, attr, None)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""


__all__ = ["ExtractAnything"]
