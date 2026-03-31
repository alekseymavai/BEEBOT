"""Tests for CrmSnapshot low stock alert — Phase 11.3."""

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.crm_snapshot import CrmSnapshot, LOW_STOCK_THRESHOLD, _LOW_STOCK_DEBOUNCE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_product(product_id: int, name: str, stock: int):
    p = MagicMock()
    p.id = product_id
    p.name = name
    p.stock = stock
    return p


def _make_snapshot(alert_fn=None) -> CrmSnapshot:
    crm = MagicMock()
    return CrmSnapshot(crm, alert_fn=alert_fn)


# ---------------------------------------------------------------------------
# _check_low_stock
# ---------------------------------------------------------------------------

class TestLowStockAlert:
    @pytest.mark.asyncio
    async def test_alert_when_stock_below_threshold(self):
        alert_fn = AsyncMock()
        snap = _make_snapshot(alert_fn=alert_fn)
        snap.products = [_make_product(1, "Прополис", LOW_STOCK_THRESHOLD - 1)]

        await snap._check_low_stock()

        alert_fn.assert_called_once()
        msg = alert_fn.call_args[0][0]
        assert "Прополис" in msg
        assert str(LOW_STOCK_THRESHOLD - 1) in msg

    @pytest.mark.asyncio
    async def test_no_alert_when_stock_at_threshold(self):
        alert_fn = AsyncMock()
        snap = _make_snapshot(alert_fn=alert_fn)
        snap.products = [_make_product(1, "Мёд", LOW_STOCK_THRESHOLD)]

        await snap._check_low_stock()

        alert_fn.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_alert_when_stock_above_threshold(self):
        alert_fn = AsyncMock()
        snap = _make_snapshot(alert_fn=alert_fn)
        snap.products = [_make_product(1, "Перга", LOW_STOCK_THRESHOLD + 10)]

        await snap._check_low_stock()

        alert_fn.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_alert_when_stock_is_none(self):
        alert_fn = AsyncMock()
        snap = _make_snapshot(alert_fn=alert_fn)
        p = _make_product(1, "Гомогенат", 0)
        p.stock = None  # unknown stock
        snap.products = [p]

        await snap._check_low_stock()

        alert_fn.assert_not_called()

    @pytest.mark.asyncio
    async def test_debounce_prevents_repeated_alerts(self):
        """Same product should not be alerted twice within debounce window."""
        alert_fn = AsyncMock()
        snap = _make_snapshot(alert_fn=alert_fn)
        snap.products = [_make_product(1, "Прополис", 2)]

        await snap._check_low_stock()
        await snap._check_low_stock()  # second call within debounce

        assert alert_fn.call_count == 1

    @pytest.mark.asyncio
    async def test_alert_after_debounce_period_expires(self):
        """Alert should fire again after debounce period."""
        alert_fn = AsyncMock()
        snap = _make_snapshot(alert_fn=alert_fn)
        snap.products = [_make_product(1, "Прополис", 2)]

        # Simulate first alert happened > debounce ago
        snap._low_stock_alerted[1] = time.monotonic() - (_LOW_STOCK_DEBOUNCE + 1)

        await snap._check_low_stock()

        alert_fn.assert_called_once()

    @pytest.mark.asyncio
    async def test_alerts_multiple_low_stock_products(self):
        """Multiple low stock products should each get their own alert."""
        alert_fn = AsyncMock()
        snap = _make_snapshot(alert_fn=alert_fn)
        snap.products = [
            _make_product(1, "Прополис", 1),
            _make_product(2, "Перга", 3),
            _make_product(3, "Мёд", 50),   # not low
        ]

        await snap._check_low_stock()

        assert alert_fn.call_count == 2
        calls_text = " ".join(str(c) for c in alert_fn.call_args_list)
        assert "Прополис" in calls_text
        assert "Перга" in calls_text
        assert "Мёд" not in calls_text

    @pytest.mark.asyncio
    async def test_no_alert_when_alert_fn_is_none(self):
        """Snapshot without alert_fn should never call anything."""
        snap = _make_snapshot(alert_fn=None)
        snap.products = [_make_product(1, "Прополис", 1)]
        # Should not raise even without alert_fn
        # _check_low_stock only called when alert_fn is set (guard in refresh)
        # but calling directly should be safe too
        await snap._check_low_stock()  # no exception expected

    @pytest.mark.asyncio
    async def test_alert_error_does_not_propagate(self):
        """If alert_fn raises, _check_low_stock should swallow the error."""
        alert_fn = AsyncMock(side_effect=Exception("Telegram down"))
        snap = _make_snapshot(alert_fn=alert_fn)
        snap.products = [_make_product(1, "Прополис", 1)]

        # Should not raise
        await snap._check_low_stock()

    @pytest.mark.asyncio
    async def test_alert_includes_product_name_and_stock(self):
        alert_fn = AsyncMock()
        snap = _make_snapshot(alert_fn=alert_fn)
        snap.products = [_make_product(42, "ПЖВМ", 0)]

        await snap._check_low_stock()

        msg = alert_fn.call_args[0][0]
        assert "ПЖВМ" in msg
        assert "0" in msg


# ---------------------------------------------------------------------------
# CrmSnapshot constructor
# ---------------------------------------------------------------------------

class TestCrmSnapshotInit:
    def test_default_has_no_alert_fn(self):
        crm = MagicMock()
        snap = CrmSnapshot(crm)
        assert snap._alert_fn is None

    def test_custom_alert_fn_stored(self):
        crm = MagicMock()
        fn = AsyncMock()
        snap = CrmSnapshot(crm, alert_fn=fn)
        assert snap._alert_fn is fn

    def test_low_stock_alerted_starts_empty(self):
        crm = MagicMock()
        snap = CrmSnapshot(crm)
        assert snap._low_stock_alerted == {}
