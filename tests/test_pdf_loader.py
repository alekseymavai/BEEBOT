"""Unit tests for src/pdf_loader.py — PDF text extraction module."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from src.pdf_loader import extract_pdf_text, process_all_pdfs


class TestExtractPdfText:
    """Tests for extract_pdf_text function."""

    def test_extract_text_from_valid_pdf(self, tmp_path):
        """Should extract text from a PDF file with readable content."""
        # Create a minimal PDF using PyPDF2
        from PyPDF2 import PdfWriter
        import io

        writer = PdfWriter()
        writer.add_blank_page(width=200, height=200)
        pdf_path = tmp_path / "test.pdf"
        with open(pdf_path, "wb") as f:
            writer.write(f)

        # Blank page returns empty string — verify function handles it gracefully
        result = extract_pdf_text(pdf_path)
        assert isinstance(result, str)

    def test_extract_text_with_mocked_reader(self, tmp_path):
        """Should concatenate text from multiple pages."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")  # minimal placeholder

        mock_page1 = MagicMock()
        mock_page1.extract_text.return_value = "Страница первая"
        mock_page2 = MagicMock()
        mock_page2.extract_text.return_value = "Страница вторая"

        with patch("src.pdf_loader.PdfReader") as mock_reader_cls:
            mock_reader = MagicMock()
            mock_reader.pages = [mock_page1, mock_page2]
            mock_reader_cls.return_value = mock_reader

            result = extract_pdf_text(pdf_path)

        assert "Страница первая" in result
        assert "Страница вторая" in result

    def test_extract_text_handles_none_page(self, tmp_path):
        """Should treat None from extract_text() as empty string."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")

        mock_page = MagicMock()
        mock_page.extract_text.return_value = None

        with patch("src.pdf_loader.PdfReader") as mock_reader_cls:
            mock_reader = MagicMock()
            mock_reader.pages = [mock_page]
            mock_reader_cls.return_value = mock_reader

            result = extract_pdf_text(pdf_path)

        assert result == ""

    def test_extract_text_returns_stripped_text(self, tmp_path):
        """Result should be stripped of leading/trailing whitespace."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")

        mock_page = MagicMock()
        mock_page.extract_text.return_value = "  текст с пробелами  "

        with patch("src.pdf_loader.PdfReader") as mock_reader_cls:
            mock_reader = MagicMock()
            mock_reader.pages = [mock_page]
            mock_reader_cls.return_value = mock_reader

            result = extract_pdf_text(pdf_path)

        assert result == "текст с пробелами"


class TestProcessAllPdfs:
    """Tests for process_all_pdfs function."""

    def test_process_all_pdfs_returns_list_of_dicts(self, tmp_path):
        """Should return a list of document dicts for each valid PDF."""
        # Create a PDF with enough content
        pdf_path = tmp_path / "Прополис.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")

        mock_page = MagicMock()
        # Make text long enough (> 50 chars)
        mock_page.extract_text.return_value = "А" * 100

        with patch("src.pdf_loader.PdfReader") as mock_reader_cls:
            mock_reader = MagicMock()
            mock_reader.pages = [mock_page]
            mock_reader_cls.return_value = mock_reader

            with patch("src.pdf_loader.TEXTS_DIR", tmp_path / "texts"):
                results = process_all_pdfs(pdf_dir=tmp_path)

        assert len(results) == 1
        doc = results[0]
        assert "source" in doc
        assert "text" in doc
        assert "filename" in doc
        assert doc["source"] == "pdf:Прополис"
        assert doc["filename"] == "Прополис.pdf"
        assert len(doc["text"]) > 50

    def test_process_all_pdfs_skips_short_text(self, tmp_path):
        """Should skip PDFs whose extracted text is too short (≤ 50 chars)."""
        pdf_path = tmp_path / "empty.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")

        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Мало"  # < 50 chars

        with patch("src.pdf_loader.PdfReader") as mock_reader_cls:
            mock_reader = MagicMock()
            mock_reader.pages = [mock_page]
            mock_reader_cls.return_value = mock_reader

            with patch("src.pdf_loader.TEXTS_DIR", tmp_path / "texts"):
                results = process_all_pdfs(pdf_dir=tmp_path)

        assert results == []

    def test_process_all_pdfs_no_pdfs(self, tmp_path):
        """Should return empty list when no PDFs exist in directory."""
        with patch("src.pdf_loader.TEXTS_DIR", tmp_path / "texts"):
            results = process_all_pdfs(pdf_dir=tmp_path)
        assert results == []

    def test_process_all_pdfs_saves_txt_file(self, tmp_path):
        """Should write extracted text to a .txt file in TEXTS_DIR."""
        pdf_path = tmp_path / "Перга.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")
        texts_dir = tmp_path / "texts"

        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Перга — уникальный продукт пчеловодства, богатый белками и витаминами."

        with patch("src.pdf_loader.PdfReader") as mock_reader_cls:
            mock_reader = MagicMock()
            mock_reader.pages = [mock_page]
            mock_reader_cls.return_value = mock_reader

            with patch("src.pdf_loader.TEXTS_DIR", texts_dir):
                results = process_all_pdfs(pdf_dir=tmp_path)

        assert len(results) == 1
        txt_path = texts_dir / "Перга.txt"
        assert txt_path.exists()
        content = txt_path.read_text(encoding="utf-8")
        assert "Перга" in content

    def test_process_all_pdfs_multiple_pdfs(self, tmp_path):
        """Should process all PDFs found in directory."""
        for name in ["Прополис.pdf", "Перга.pdf", "Мёд.pdf"]:
            (tmp_path / name).write_bytes(b"%PDF-1.4 fake")

        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Х" * 100

        with patch("src.pdf_loader.PdfReader") as mock_reader_cls:
            mock_reader = MagicMock()
            mock_reader.pages = [mock_page]
            mock_reader_cls.return_value = mock_reader

            with patch("src.pdf_loader.TEXTS_DIR", tmp_path / "texts"):
                results = process_all_pdfs(pdf_dir=tmp_path)

        assert len(results) == 3

    def test_process_all_pdfs_source_format(self, tmp_path):
        """Source field should follow 'pdf:<stem>' format."""
        (tmp_path / "Настойка ПЖВМ.pdf").write_bytes(b"%PDF-1.4 fake")

        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Д" * 100

        with patch("src.pdf_loader.PdfReader") as mock_reader_cls:
            mock_reader = MagicMock()
            mock_reader.pages = [mock_page]
            mock_reader_cls.return_value = mock_reader

            with patch("src.pdf_loader.TEXTS_DIR", tmp_path / "texts"):
                results = process_all_pdfs(pdf_dir=tmp_path)

        assert results[0]["source"] == "pdf:Настойка ПЖВМ"
