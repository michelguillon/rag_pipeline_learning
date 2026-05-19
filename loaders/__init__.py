"""loaders — format-specific document parsers.

Each loader converts one file format into the common `Paragraph` model
(models/paragraph.py). Adding a new format is adding one loader here;
nothing downstream changes.

    load_docx(path) -> list[Paragraph]   — Microsoft Word .docx
    load_pdf(path)  -> list[Paragraph]   — PDF (Phase 2 stretch goal)
"""

from loaders.docx_loader import load_docx

__all__ = ["load_docx"]
