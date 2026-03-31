"""Тесты для src/tunnel_monitor.py — Фаза 12.1."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from src.tunnel_monitor import TunnelMonitor


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_monitor(alert_fn=None, host="127.0.0.1", port=8990, interval=60):
    return TunnelMonitor(alert_fn=alert_fn, host=host, port=port, interval=interval)


async def _run_one_cycle(monitor: TunnelMonitor) -> None:
    """Запустить один шаг цикла run() без sleep."""
    healthy = await monitor.check_once()
    prev = monitor._healthy
    if prev is None:
        monitor._healthy = healthy
    elif healthy != prev:
        monitor._healthy = healthy
        if not healthy:
            await monitor._send_alert("⚠️ Groq-туннель недоступен — бот работает в режиме FAQ")
        else:
            await monitor._send_alert("✅ Groq-туннель восстановлен — бот работает в штатном режиме")


# ---------------------------------------------------------------------------
# check_once tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_healthy_when_port_open():
    """check_once() возвращает True, когда TCP-коннект успешен."""
    monitor = _make_monitor()

    async def _mock_open(*args, **kwargs):
        reader = AsyncMock()
        writer = AsyncMock()
        writer.close = lambda: None
        writer.wait_closed = AsyncMock()
        return reader, writer

    with patch("asyncio.open_connection", side_effect=_mock_open):
        result = await monitor.check_once()

    assert result is True
    assert monitor._dev_mode is False


@pytest.mark.asyncio
async def test_unhealthy_when_port_closed():
    """check_once() возвращает False при таймауте (порт слушается, но не отвечает)."""
    monitor = _make_monitor()

    with patch("asyncio.open_connection", side_effect=asyncio.TimeoutError()):
        result = await monitor.check_once()

    assert result is False
    assert monitor._dev_mode is False


@pytest.mark.asyncio
async def test_dev_mode_when_connection_refused():
    """check_once() возвращает True и устанавливает dev_mode при ConnectionRefusedError."""
    monitor = _make_monitor()

    with patch("asyncio.open_connection", side_effect=ConnectionRefusedError()):
        result = await monitor.check_once()

    assert result is True
    assert monitor._dev_mode is True


# ---------------------------------------------------------------------------
# alert tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_alert_on_state_change_down():
    """Алерт вызывается при переходе up→down."""
    alert_fn = AsyncMock()
    monitor = _make_monitor(alert_fn=alert_fn)
    monitor._healthy = True   # симулируем предыдущее состояние — up

    with patch("asyncio.open_connection", side_effect=asyncio.TimeoutError()):
        await _run_one_cycle(monitor)

    alert_fn.assert_awaited_once()
    call_text = alert_fn.call_args[0][0]
    assert "недоступен" in call_text.lower()
    assert monitor.is_healthy is False


@pytest.mark.asyncio
async def test_alert_on_state_change_recovery():
    """Алерт вызывается при переходе down→up."""
    alert_fn = AsyncMock()
    monitor = _make_monitor(alert_fn=alert_fn)
    monitor._healthy = False  # симулируем предыдущее состояние — down

    async def _mock_open(*args, **kwargs):
        reader = AsyncMock()
        writer = AsyncMock()
        writer.close = lambda: None
        writer.wait_closed = AsyncMock()
        return reader, writer

    with patch("asyncio.open_connection", side_effect=_mock_open):
        await _run_one_cycle(monitor)

    alert_fn.assert_awaited_once()
    call_text = alert_fn.call_args[0][0]
    assert "восстановлен" in call_text.lower()
    assert monitor.is_healthy is True


@pytest.mark.asyncio
async def test_no_alert_if_state_unchanged_healthy():
    """Алерт НЕ вызывается если состояние не изменилось (was healthy, still healthy)."""
    alert_fn = AsyncMock()
    monitor = _make_monitor(alert_fn=alert_fn)
    monitor._healthy = True

    async def _mock_open(*args, **kwargs):
        reader = AsyncMock()
        writer = AsyncMock()
        writer.close = lambda: None
        writer.wait_closed = AsyncMock()
        return reader, writer

    with patch("asyncio.open_connection", side_effect=_mock_open):
        await _run_one_cycle(monitor)

    alert_fn.assert_not_awaited()


@pytest.mark.asyncio
async def test_no_alert_if_state_unchanged_unhealthy():
    """Алерт НЕ вызывается если состояние не изменилось (was down, still down)."""
    alert_fn = AsyncMock()
    monitor = _make_monitor(alert_fn=alert_fn)
    monitor._healthy = False

    with patch("asyncio.open_connection", side_effect=asyncio.TimeoutError()):
        await _run_one_cycle(monitor)

    alert_fn.assert_not_awaited()


# ---------------------------------------------------------------------------
# is_healthy property tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_is_healthy_reflects_last_check_false():
    """is_healthy отражает результат последней проверки — False."""
    monitor = _make_monitor()
    monitor._healthy = False
    assert monitor.is_healthy is False


@pytest.mark.asyncio
async def test_is_healthy_reflects_last_check_true():
    """is_healthy отражает результат последней проверки — True."""
    monitor = _make_monitor()
    monitor._healthy = True
    assert monitor.is_healthy is True


@pytest.mark.asyncio
async def test_is_healthy_default_before_first_check():
    """is_healthy = True до первой проверки (graceful start)."""
    monitor = _make_monitor()
    assert monitor._healthy is None
    assert monitor.is_healthy is True


@pytest.mark.asyncio
async def test_is_healthy_true_in_dev_mode():
    """is_healthy = True в dev-режиме (порт не слушается)."""
    monitor = _make_monitor()
    monitor._dev_mode = True
    monitor._healthy = False   # даже если internal state=False, dev_mode побеждает
    assert monitor.is_healthy is True


# ---------------------------------------------------------------------------
# first-check init (no alert on first cycle)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_no_alert_on_first_check_even_if_down():
    """Алерт НЕ вызывается при первой проверке, даже если туннель недоступен."""
    alert_fn = AsyncMock()
    monitor = _make_monitor(alert_fn=alert_fn)
    # _healthy = None — первый чек
    assert monitor._healthy is None

    with patch("asyncio.open_connection", side_effect=asyncio.TimeoutError()):
        await _run_one_cycle(monitor)

    alert_fn.assert_not_awaited()
    assert monitor._healthy is False
    assert monitor.is_healthy is False
