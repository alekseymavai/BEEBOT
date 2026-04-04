"""Тесты DashboardService — агрегация данных дашборда."""

import pytest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

from src.services.dashboard_service import DashboardService


def _make_order(id=1, status="Новый", total=1000.0, client_id=1,
                date=None, delivery_method="СДЭК"):
    return SimpleNamespace(
        id=id, status=status, total=total, client_id=client_id,
        date=date or datetime(2026, 3, 1, tzinfo=timezone.utc),
        delivery_method=delivery_method,
    )


def _make_item(order_id=1, product_id=10, product_name="Перга",
               quantity=2, total=500.0):
    return SimpleNamespace(
        order_id=order_id, product_id=product_id,
        product_name=product_name, quantity=quantity, total=total,
    )


class TestDashboardService:
    @pytest.mark.asyncio
    async def test_get_stats_no_crm(self):
        svc = DashboardService(crm=None)
        stats = await svc.get_stats("all")
        assert stats["total_orders"] == 0

    @pytest.mark.asyncio
    async def test_get_stats_all(self):
        crm = AsyncMock()
        crm.get_dashboard_stats.return_value = {
            "total_orders": 10, "total_clients": 5,
            "total_revenue": 50000.0, "avg_order": 5000.0,
            "new_orders": 3, "delivered_orders": 4,
        }
        svc = DashboardService(crm=crm)
        stats = await svc.get_stats("all")
        assert stats["total_orders"] == 10
        crm.get_dashboard_stats.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_stats_period(self):
        crm = AsyncMock()
        crm.get_orders.return_value = [
            _make_order(total=1000),
            _make_order(id=2, total=2000, client_id=2),
        ]
        svc = DashboardService(crm=crm)
        stats = await svc.get_stats("30d")
        assert stats["total_orders"] == 2
        assert stats["total_revenue"] == 3000.0
        assert stats["total_clients"] == 2

    @pytest.mark.asyncio
    async def test_get_alerts(self):
        crm = AsyncMock()
        crm.get_orders.return_value = [
            _make_order(status="Новый", date=datetime(2020, 1, 1, tzinfo=timezone.utc)),
        ]
        crm.get_products.return_value = [
            SimpleNamespace(stock=2, in_stock=True),
        ]
        svc = DashboardService(crm=crm)
        alerts = await svc.get_alerts()
        assert alerts["stale_new"] == 1
        assert alerts["low_stock"] == 1

    @pytest.mark.asyncio
    async def test_get_charts(self):
        crm = AsyncMock()
        crm.get_orders.return_value = [
            _make_order(id=1, status="Новый"),
        ]
        crm.get_order_items_bulk.return_value = [
            _make_item(order_id=1, product_name="Мёд", quantity=3, total=900),
        ]
        svc = DashboardService(crm=crm)
        charts = await svc.get_charts("all")
        assert len(charts["top_products"]) == 1
        assert charts["top_products"][0]["name"] == "Мёд"
        assert "funnel" in charts
        assert "monthly" in charts

    def test_filter_by_date(self):
        old = _make_order(date=datetime(2020, 1, 1, tzinfo=timezone.utc))
        new = _make_order(date=datetime(2026, 3, 1, tzinfo=timezone.utc))
        since = datetime(2025, 1, 1, tzinfo=timezone.utc)
        result = DashboardService._filter_by_date([old, new], since)
        assert len(result) == 1

    def test_aggregate_monthly(self):
        orders = [
            _make_order(date=datetime(2026, 3, 1, tzinfo=timezone.utc), total=1000),
            _make_order(date=datetime(2026, 3, 15, tzinfo=timezone.utc), total=2000),
        ]
        result = DashboardService._aggregate_monthly(orders)
        assert result["revenue"] == [3000.0]
        assert len(result["labels"]) == 1
