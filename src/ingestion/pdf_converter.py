"""PDF → Markdown conversion using pymupdf4llm."""

from pathlib import Path
import pymupdf4llm
import logging

logger = logging.getLogger(__name__)


def convert_pdf_to_markdown(pdf_path: Path, output_dir: Path) -> Path:
    """Convert a single PDF to Markdown and save it. Returns the output path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / (pdf_path.stem + ".md")

    logger.info(f"Converting {pdf_path.name} → {md_path.name}")
    md_text = pymupdf4llm.to_markdown(str(pdf_path))

    md_path.write_text(md_text, encoding="utf-8")
    logger.info(f"Saved {md_path} ({len(md_text):,} chars)")
    return md_path


def convert_all_pdfs(docs_dir: Path, output_dir: Path) -> list[Path]:
    """Convert all PDFs in docs_dir to Markdown. Returns list of output paths."""
    pdf_files = list(docs_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDF files found in {docs_dir}")
        return []

    results = []
    for pdf in pdf_files:
        try:
            md_path = convert_pdf_to_markdown(pdf, output_dir)
            results.append(md_path)
        except Exception as e:
            logger.error(f"Failed to convert {pdf.name}: {e}")

    logger.info(f"Converted {len(results)}/{len(pdf_files)} PDFs")
    return results
