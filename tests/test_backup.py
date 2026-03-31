"""Тесты для BackupManager (src/backup.py)."""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.backup import BackupManager, _KEEP_DAILY, _YD_DAILY, _YD_ROOT, _YD_WEEKLY


# ---------------------------------------------------------------------------
# Фикстуры
# ---------------------------------------------------------------------------

@pytest.fixture
def memory_db(tmp_path: Path) -> Path:
    db = tmp_path / "memory.db"
    db.write_bytes(b"SQLite-fake-content")
    return db


def _make_manager(token: str = "test_token", memory_db: Path | None = None, crm=None):
    m = BackupManager(token=token, memory_db_path=memory_db, crm=crm)
    return m


# ---------------------------------------------------------------------------
# available property
# ---------------------------------------------------------------------------

def test_available_with_token():
    m = _make_manager(token="abc123")
    assert m.available is True


def test_not_available_without_token():
    m = _make_manager(token="")
    assert m.available is False


# ---------------------------------------------------------------------------
# _do_daily
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_do_daily_uploads_file(memory_db: Path):
    m = _make_manager(memory_db=memory_db)
    mock_yd = AsyncMock()
    mock_yd.upload = AsyncMock()
    mock_yd.listdir = AsyncMock(return_value=_aiter([]))
    m._yd = mock_yd

    path = await m._do_daily()

    assert path is not None
    assert path.startswith(_YD_DAILY)
    assert path.endswith(".db")
    mock_yd.upload.assert_called_once()
    # Первый аргумент — BytesIO с содержимым файла
    buf_arg = mock_yd.upload.call_args[0][0]
    assert buf_arg.read() == b"SQLite-fake-content"


@pytest.mark.asyncio
async def test_do_daily_returns_none_if_no_db():
    m = _make_manager(memory_db=Path("/nonexistent/memory.db"))
    m._yd = AsyncMock()
    result = await m._do_daily()
    assert result is None


@pytest.mark.asyncio
async def test_do_daily_returns_none_on_upload_error(memory_db: Path):
    m = _make_manager(memory_db=memory_db)
    mock_yd = AsyncMock()
    mock_yd.upload = AsyncMock(side_effect=OSError("network error"))
    m._yd = mock_yd
    result = await m._do_daily()
    assert result is None


# ---------------------------------------------------------------------------
# _do_weekly
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_do_weekly_skipped_without_crm():
    m = _make_manager()
    m._yd = AsyncMock()
    result = await m._do_weekly()
    assert result is None


@pytest.mark.asyncio
async def test_do_weekly_exports_csv():
    crm = AsyncMock()
    order = MagicMock()
    order.id = 42
    order.date = "01.03.2026"
    order.status = "Доставлен"
    order.client_name = "Иван"
    order.total = 1500
    order.address = "Москва"
    crm.get_orders = AsyncMock(return_value=[order])

    m = _make_manager(crm=crm)
    mock_yd = AsyncMock()
    mock_yd.upload = AsyncMock()
    m._yd = mock_yd

    path = await m._do_weekly(force=True)
    assert path is not None
    assert path.startswith(_YD_WEEKLY)
    assert path.endswith(".csv")

    buf_arg = mock_yd.upload.call_args[0][0]
    content = buf_arg.read().decode("utf-8-sig")
    assert "Иван" in content
    assert "1500" in content


# ---------------------------------------------------------------------------
# _cleanup_daily
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cleanup_keeps_last_30():
    m = _make_manager()
    mock_yd = AsyncMock()

    # 35 файлов — удалить должны быть первые 5
    def _item(n):
        i = MagicMock()
        i.name = n
        return i

    names = [f"memory_2026-01-{i:02d}.db" for i in range(1, 36)]
    items = [_item(n) for n in names]

    mock_yd.listdir = MagicMock(return_value=_aiter(items))
    mock_yd.remove = AsyncMock()
    m._yd = mock_yd

    await m._cleanup_daily()

    assert mock_yd.remove.call_count == 5
    removed = [c.args[0] for c in mock_yd.remove.call_args_list]
    for i in range(1, 6):
        assert f"{_YD_DAILY}/memory_2026-01-{i:02d}.db" in removed


@pytest.mark.asyncio
async def test_cleanup_nothing_to_delete():
    m = _make_manager()
    mock_yd = AsyncMock()

    def _item(n):
        i = MagicMock()
        i.name = n
        return i

    names = [f"memory_2026-01-{i:02d}.db" for i in range(1, 11)]
    items = [_item(n) for n in names]
    mock_yd.listdir = MagicMock(return_value=_aiter(items))
    mock_yd.remove = AsyncMock()
    m._yd = mock_yd

    await m._cleanup_daily()
    mock_yd.remove.assert_not_called()


# ---------------------------------------------------------------------------
# run() — no token
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_exits_if_no_token():
    m = _make_manager(token="")
    # Не должен зависнуть в цикле
    await m.run()


# ---------------------------------------------------------------------------
# backup_now()
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_backup_now_no_token():
    m = _make_manager(token="")
    result = await m.backup_now()
    assert "error" in result


@pytest.mark.asyncio
async def test_backup_now_returns_paths(memory_db: Path):
    crm = AsyncMock()
    crm.get_orders = AsyncMock(return_value=[])
    m = _make_manager(memory_db=memory_db, crm=crm)

    mock_yd = AsyncMock()
    mock_yd.exists = AsyncMock(return_value=True)
    mock_yd.upload = AsyncMock()
    mock_yd.listdir = MagicMock(return_value=_aiter([]))
    # Pre-inject yd client so backup_now() uses it directly
    m._yd = mock_yd

    # Patch _ensure_dirs to skip remote folder creation
    m._ensure_dirs = AsyncMock()

    result = await m.backup_now()

    assert "daily" in result
    assert "weekly" in result


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

async def _aiter(items):
    for item in items:
        yield item
