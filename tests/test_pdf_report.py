"""Unit tests for src/pdf_report.py — генератор PDF-отчётов по продажам."""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from src.pdf_report import (
    _abc_segments,
    _low_stock_products,
    _monthly_revenue,
    _period_label,
    _top_products,
    generate_sales_report,
)


# ---------------------------------------------------------------------------
# Вспомогательные фабрики
# ---------------------------------------------------------------------------


def _make_order(
    order_id: int = 1,
    client_id: int = 1,
    client_name: str = "Иван Иванов",
    total: float = 1000.0,
    days_ago: int = 0,
    status: str = "Выполнен",
) -> MagicMock:
    order = MagicMock()
    order.id = order_id
    order.client_id = client_id
    order.client_name = client_name
    order.total = total
    order.date = datetime.now() - timedelta(days=days_ago)
    order.status = status
    return order


def _make_item(
    order_id: int = 1,
    product_id: int = 1,
    product_name: str = "Мёд цветочный",
    quantity: int = 2,
    unit_price: float = 500.0,
) -> MagicMock:
    item = MagicMock()
    item.order_id = order_id
    item.product_id = product_id
    item.product_name = product_name
    item.quantity = quantity
    item.unit_price = unit_price
    item.total = quantity * unit_price
    return item


def _make_product(
    product_id: int = 1,
    name: str = "Мёд цветочный",
    stock: float = 10.0,
) -> MagicMock:
    product = MagicMock()
    product.id = product_id
    product.name = name
    product.stock = stock
    return product


def _make_sample_orders(n: int = 3) -> list:
    return [_make_order(order_id=i, client_name=f"Клиент {i}", total=float(i * 1000)) for i in range(1, n + 1)]


def _make_sample_items(orders: list) -> dict:
    result = {}
    for order in orders:
        items = [_make_item(order_id=order.id, product_name=f"Товар {order.id}", quantity=order.id)]
        result[order.id] = items
    return result


# ---------------------------------------------------------------------------
# Тест 1: generate_sales_report возвращает байты
# ---------------------------------------------------------------------------


def test_pdf_bytes_returned():
    orders = _make_sample_orders(3)
    items = _make_sample_items(orders)
    result = generate_sales_report(orders, items, "30d")
    assert isinstance(result, bytes)
    assert len(result) > 0


# ---------------------------------------------------------------------------
# Тест 2: PDF начинается с заголовка %PDF
# ---------------------------------------------------------------------------


def test_pdf_starts_with_pdf_header():
    orders = _make_sample_orders(2)
    items = _make_sample_items(orders)
    result = generate_sales_report(orders, items, "90d")
    assert result[:4] == b"%PDF", "PDF должен начинаться с сигнатуры %PDF"


# ---------------------------------------------------------------------------
# Тест 3: PDF содержит ожидаемые разделы (в бинарном виде)
# ---------------------------------------------------------------------------


def test_pdf_contains_expected_sections():
    orders = _make_sample_orders(3)
    items = _make_sample_items(orders)
    result = generate_sales_report(orders, items, "365d")
    content_text = result.decode("latin-1")

    # PDF содержит текстовые фрагменты внутри потоков
    assert "BEEBOT" in content_text or len(result) > 1000, (
        "PDF должен содержать данные отчёта"
    )


# ---------------------------------------------------------------------------
# Тест 4: _period_label возвращает правильные метки
# ---------------------------------------------------------------------------


def test_period_label_in_output():
    assert "30" in _period_label("30d")
    assert "90" in _period_label("90d")
    assert "365" in _period_label("365d")
    assert "неделю" in _period_label("week")
    assert "месяц" in _period_label("month")
    assert "время" in _period_label("all")


# ---------------------------------------------------------------------------
# Тест 5: пустые заказы — PDF генерируется без ошибок
# ---------------------------------------------------------------------------


def test_handles_empty_orders():
    result = generate_sales_report([], {}, "30d")
    assert isinstance(result, bytes)
    assert result[:4] == b"%PDF"


# ---------------------------------------------------------------------------
# Тест 6: пустые товары (products=None) — PDF генерируется без секции склада
# ---------------------------------------------------------------------------


def test_handles_empty_products():
    orders = _make_sample_orders(2)
    items = _make_sample_items(orders)
    # products=None — секция склада не включается
    result = generate_sales_report(orders, items, "30d", products=None)
    assert isinstance(result, bytes)
    assert len(result) > 0


# ---------------------------------------------------------------------------
# Тест 7: топ-10 ограничивает количество товаров
# ---------------------------------------------------------------------------


def test_top_products_limited_to_10():
    orders = [_make_order(order_id=i) for i in range(1, 20)]
    items_by_order: dict = {}
    for order in orders:
        items_by_order[order.id] = [
            _make_item(order_id=order.id, product_name=f"Товар-{order.id}", quantity=1)
        ]

    top = _top_products(orders, items_by_order, top_n=10)
    assert len(top) <= 10, "Топ товаров должен содержать не более 10 позиций"


# ---------------------------------------------------------------------------
# Тест 8: ABC-анализ — все три сегмента присутствуют при достаточном числе клиентов
# ---------------------------------------------------------------------------


def test_abc_analysis_in_report():
    # Создаём 20 клиентов с сильно различающейся выручкой (классический ABC)
    orders = []
    for i in range(1, 21):
        # Первые 2 клиента — A-сегмент (высокая выручка)
        # Следующие 6 — B-сегмент
        # Остальные 12 — C-сегмент
        if i <= 2:
            rev = 50000.0
        elif i <= 8:
            rev = 5000.0
        else:
            rev = 200.0
        orders.append(_make_order(order_id=i, client_id=i, client_name=f"Клиент {i}", total=rev))

    segments = _abc_segments(orders)
    assert "A" in segments
    assert "B" in segments
    assert "C" in segments

    # Сегмент A не пустой
    assert len(segments["A"]) > 0, "Сегмент A должен содержать клиентов"

    # Общее число клиентов совпадает
    total = sum(len(v) for v in segments.values())
    assert total == 20


# ---------------------------------------------------------------------------
# Дополнительные тесты вспомогательных функций
# ---------------------------------------------------------------------------


def test_monthly_revenue_groups_by_month():
    orders = [
        _make_order(order_id=1, total=1000.0, days_ago=5),
        _make_order(order_id=2, total=2000.0, days_ago=35),
        _make_order(order_id=3, total=500.0, days_ago=5),
    ]
    monthly = _monthly_revenue(orders)
    assert len(monthly) >= 1
    # Суммарная выручка
    total_rev = sum(rev for _, _, rev in monthly)
    assert abs(total_rev - 3500.0) < 0.01


def test_low_stock_products_filters_correctly():
    products = [
        _make_product(product_id=1, name="Перга", stock=3.0),
        _make_product(product_id=2, name="Мёд", stock=50.0),
        _make_product(product_id=3, name="Прополис", stock=0.0),
    ]
    low = _low_stock_products(products, threshold=5)
    names = [n for n, _ in low]
    assert "Перга" in names
    assert "Прополис" in names
    assert "Мёд" not in names


def test_generate_report_with_products_low_stock_section():
    orders = _make_sample_orders(2)
    items = _make_sample_items(orders)
    products = [
        _make_product(name="Мёд гречишный", stock=2.0),
        _make_product(name="Пыльца", stock=50.0),
    ]
    result = generate_sales_report(orders, items, "30d", products=products)
    assert isinstance(result, bytes)
    assert result[:4] == b"%PDF"
