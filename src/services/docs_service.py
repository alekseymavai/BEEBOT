"""Сервис управления PDF-инструкциями: конвертация PDF↔DOCX, список, сохранение."""
from __future__ import annotations

import asyncio
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

_PDFS_DIR = Path("data/pdfs")
_WORD_DIR = Path("data/docs_word")


class DocsService:
    """Конвертация и хранение инструкций."""

    def __init__(
        self,
        pdfs_dir: Path = _PDFS_DIR,
        word_dir: Path = _WORD_DIR,
    ) -> None:
        self.pdfs_dir = pdfs_dir
        self.word_dir = word_dir
        self.word_dir.mkdir(parents=True, exist_ok=True)

    def list_pdfs(self) -> list[str]:
        """Вернуть отсортированный список имён инструкций (без .pdf)."""
        return sorted(p.stem for p in self.pdfs_dir.glob("*.pdf"))

    def pdf_to_docx(self, name: str) -> Path:
        """Конвертировать PDF → DOCX. Кешируется в word_dir."""
        pdf_path = self.pdfs_dir / f"{name}.pdf"
        if not pdf_path.exists():
            raise FileNotFoundError(f"{name}: PDF не найден в {self.pdfs_dir}")

        from pdf2docx import Converter  # lazy import

        docx_path = self.word_dir / f"{name}.docx"
        with Converter(str(pdf_path)) as cv:
            cv.convert(str(docx_path), start=0, end=None)
        return docx_path

    def docx_to_pdf(self, docx_path: Path, dest_name: str) -> Path:
        """Конвертировать DOCX → PDF через LibreOffice headless."""
        if not docx_path.exists():
            raise FileNotFoundError(f"DOCX не найден: {docx_path}")
        result = subprocess.run(
            [
                "libreoffice",
                "--headless",
                "--convert-to", "pdf",
                "--outdir", str(self.word_dir),
                str(docx_path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            raise RuntimeError(f"LibreOffice завершился с ошибкой: {result.stderr}")

        converted = self.word_dir / f"{docx_path.stem}.pdf"
        dest = self.pdfs_dir / f"{dest_name}.pdf"
        converted.replace(dest)
        logger.info("Сохранён PDF: %s", dest)
        return dest

    async def rebuild_kb(self) -> None:
        """Пересобрать FAISS-индекс базы знаний (src.build_kb)."""
        logger.info("Запуск пересборки KB...")
        proc = await asyncio.create_subprocess_exec(
            "python", "-m", "src.build_kb",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            logger.error("Ошибка пересборки KB: %s", stderr.decode())
            raise RuntimeError(f"Ошибка пересборки KB: {stderr.decode()}")
        else:
            logger.info("KB пересобрана успешно: %s", stdout.decode()[:200])
