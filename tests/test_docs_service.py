import asyncio
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

from src.services.docs_service import DocsService


@pytest.fixture
def tmp_pdfs(tmp_path):
    """Создаём 2 тестовых PDF-файла."""
    (tmp_path / "Перга.pdf").write_bytes(b"%PDF-1.4 test")
    (tmp_path / "Прополис.pdf").write_bytes(b"%PDF-1.4 test")
    return tmp_path


def test_list_pdfs_returns_sorted_names(tmp_pdfs):
    svc = DocsService(pdfs_dir=tmp_pdfs, word_dir=tmp_pdfs / "word")
    names = svc.list_pdfs()
    assert names == ["Перга", "Прополис"]


def test_list_pdfs_empty_dir(tmp_path):
    svc = DocsService(pdfs_dir=tmp_path, word_dir=tmp_path / "word")
    assert svc.list_pdfs() == []


def test_pdf_to_docx_returns_path(tmp_pdfs):
    """pdf_to_docx должен вернуть путь к DOCX в word_dir."""
    import sys
    import types

    svc = DocsService(pdfs_dir=tmp_pdfs, word_dir=tmp_pdfs / "word")

    mock_instance = MagicMock()
    mock_instance.__enter__ = MagicMock(return_value=mock_instance)
    mock_instance.__exit__ = MagicMock(return_value=False)
    mock_instance.convert.return_value = None

    MockConverter = MagicMock(return_value=mock_instance)

    fake_pdf2docx = types.ModuleType("pdf2docx")
    fake_pdf2docx.Converter = MockConverter

    with patch.dict(sys.modules, {"pdf2docx": fake_pdf2docx}):
        result = svc.pdf_to_docx("Перга")

    assert result == tmp_pdfs / "word" / "Перга.docx"
    assert result.parent == tmp_pdfs / "word"


def test_pdf_to_docx_raises_if_not_found(tmp_pdfs):
    svc = DocsService(pdfs_dir=tmp_pdfs, word_dir=tmp_pdfs / "word")
    with pytest.raises(FileNotFoundError, match="Несуществующий"):
        svc.pdf_to_docx("Несуществующий")


def test_docx_to_pdf_calls_libreoffice(tmp_pdfs):
    svc = DocsService(pdfs_dir=tmp_pdfs, word_dir=tmp_pdfs / "word")
    docx_file = tmp_pdfs / "word" / "Перга.docx"
    svc.word_dir.mkdir(parents=True, exist_ok=True)
    docx_file.write_bytes(b"PK fake docx content")

    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stderr = ""
        # LibreOffice создаёт PDF рядом с DOCX
        (tmp_pdfs / "word" / "Перга.pdf").write_bytes(b"%PDF-1.4")
        result = svc.docx_to_pdf(docx_file, dest_name="Перга")

    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]
    assert "libreoffice" in cmd
    assert "--convert-to" in cmd
    assert "pdf" in cmd
    assert result == tmp_pdfs / "Перга.pdf"


def test_docx_to_pdf_raises_on_failure(tmp_pdfs):
    svc = DocsService(pdfs_dir=tmp_pdfs, word_dir=tmp_pdfs / "word")
    docx_file = tmp_pdfs / "word" / "Test.docx"
    svc.word_dir.mkdir(parents=True, exist_ok=True)
    docx_file.write_bytes(b"PK fake")

    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 1
        mock_run.return_value.stderr = "libreoffice error"
        with pytest.raises(RuntimeError, match="LibreOffice"):
            svc.docx_to_pdf(docx_file, dest_name="Test")


@pytest.mark.asyncio
async def test_rebuild_kb_raises_on_failure(tmp_pdfs):
    svc = DocsService(pdfs_dir=tmp_pdfs, word_dir=tmp_pdfs / "word")

    mock_proc = MagicMock()
    mock_proc.communicate = AsyncMock(return_value=(b"", b"build error"))
    mock_proc.returncode = 1

    with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        with pytest.raises(RuntimeError, match="Ошибка пересборки KB"):
            await svc.rebuild_kb()


@pytest.mark.asyncio
async def test_rebuild_kb_runs_module(tmp_pdfs):
    svc = DocsService(pdfs_dir=tmp_pdfs, word_dir=tmp_pdfs / "word")

    mock_proc = MagicMock()
    mock_proc.communicate = AsyncMock(return_value=(b"ok", b""))
    mock_proc.returncode = 0

    with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
        await svc.rebuild_kb()

    mock_exec.assert_called_once_with(
        "python", "-m", "src.build_kb",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
