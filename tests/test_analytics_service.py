"""Тесты AnalyticsService — аналитика продаж."""

import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock
from datetime import datetime

from src.services.analytics_service import (
    AnalyticsService,
    keyword_classify,
    filter_by_period,
    format_orders_report,
    format_top_products_report,
    format_clients_report,
)


def _make_order(id=1, number="001", status="Новый", total=1000.0, client_id=1,
                client_name="Иван", date=None, delivery_method="СДЭК", source="Telegram"):
    return SimpleNamespace(
        id=id, number=number, status=status, total=total,
        client_id=client_id, client_name=client_name,
        date=date or datetime(2026, 3, 1),
        delivery_method=delivery_method, source=source,
    )


def _make_item(id=1, order_id=1, product_id=10, product_name="Перга", quantity=2, total=500.0):
    return SimpleNamespace(
        id=id, order_id=order_id, product_id=product_id,
        product_name=product_name, quantity=quantity, total=total,
    )


def _make_crm(orders=None, items=None):
    crm = AsyncMock()
    crm.get_orders.return_value = orders or []
    crm.get_order_items_bulk.return_value = items or []
    return crm


class TestKeywordClassify:
    def test_top_products(self):
        period, report = keyword_classify("топ продуктов за неделю")
        assert period == "week"
        assert report == "top"

    def test_packaging(self):
        period, report = keyword_classify("что нужно фасовать")
        assert report == "packaging"

    def test_default_summary(self):
        period, report = keyword_classify("общая статистика")
        assert period == "all"
        assert report == "summary"

    def test_month_period(self):
        period, _ = keyword_classify("продажи за месяц")
        assert period == "month"


class TestFilterByPeriod:
    def test_all_returns_everything(self):
        orders = [_make_order()]
        result = filter_by_period(orders, "all")
        assert len(result) == 1

    def test_week_filters_old(self):
        old = _make_order(date=datetime(2020, 1, 1))
        recent = _make_order(date=datetime.now())
        result = filter_by_period([old, recent], "week")
        assert len(result) == 1


class TestFormatReports:
    def test_orders_report_empty(self):
        result = format_orders_report([], "all")
        assert "нет данных" in result

    def test_orders_report_with_data(self):
        orders = [_make_order(total=1000), _make_order(total=2000)]
        result = format_orders_report(orders, "all")
        assert "3,000" in result or "3 000" in result

    def test_top_products_report(self):
        orders = [_make_order(id=1)]
        items = {1: [_make_item(product_name="Перга", quantity=5, total=500)]}
        result = format_top_products_report(orders, items, "all")
        assert "Перга" in result

    def test_clients_report(self):
        orders = [
            _make_order(client_id=1, client_name="Иван"),
            _make_order(client_id=1, client_name="Иван"),
            _make_order(client_id=2, client_name="Мария"),
        ]
        result = format_clients_report(orders, "all")
        assert "Повторные: 1" in result


class TestAnalyticsServiceHandleQuery:
    @pytest.mark.asyncio
    async def test_handle_query_no_crm(self):
        svc = AnalyticsService(crm=None)
        result = await svc.handle_query("статистика")
        assert "нет данных" in result

    @pytest.mark.asyncio
    async def test_handle_query_with_crm(self):
        orders = [_make_order()]
        items = [_make_item(order_id=1)]
        crm = _make_crm(orders=orders, items=items)
        svc = AnalyticsService(crm=crm)
        result = await svc.handle_query("общая статистика")
        assert "1" in result  # хотя бы 1 заказ

    @pytest.mark.asyncio
    async def test_get_sales_summary(self):
        orders = [_make_order(total=1500)]
        crm = _make_crm(orders=orders)
        svc = AnalyticsService(crm=crm)
        summary = await svc.get_sales_summary("all")
        assert summary["total_orders"] == 1
        assert summary["total_revenue"] == 1500.0
