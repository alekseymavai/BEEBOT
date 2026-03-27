"""Тесты для IntegramClient (src/integram_client.py).

Используют моки IntegramAPI — реальных запросов к Integram не делают.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.integram_client import (
    IntegramAuthError,
    IntegramClient,
    IntegramError,
    IntegramNotFoundError,
)
from src.integram_api import IntegramAPIError
from src.models import Client, Order, OrderItem, Product


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _make_client(**kwargs) -> IntegramClient:
    """Создать IntegramClient с замоканным IntegramAPI."""
    with patch("src.integram_client.IntegramAPI"):
        client = IntegramClient(**kwargs)
    # _api уже MagicMock, делаем его методы async
    client._api.authenticate = AsyncMock()
    client._api.close = AsyncMock()
    client._api.get_products = AsyncMock(return_value=[])
    client._api.get_clients = AsyncMock(return_value=[])
    client._api.get_orders = AsyncMock(return_value=[])
    client._api.get_order_items = AsyncMock(return_value=[])
    client._api.get_all_objects = AsyncMock(return_value=[])
    client._api.create_object = AsyncMock(return_value=1)
    client._api.set_requisites = AsyncMock()
    client._api.set_reference_field = AsyncMock()
    client._api.delete_object = AsyncMock()
    return client


# ---------------------------------------------------------------------------
# Тесты: authenticate
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_authenticate_success():
    client = _make_client()
    await client.authenticate()
    client._api.authenticate.assert_called_once()
    assert client._authenticated is True


@pytest.mark.asyncio
async def test_authenticate_raises_on_api_error():
    """IntegramAPIError из _api.authenticate → IntegramAuthError."""
    client = _make_client()
    client._api.authenticate.side_effect = IntegramAPIError("invalid credentials")

    with pytest.raises(IntegramAuthError):
        await client.authenticate()


# ---------------------------------------------------------------------------
# Тесты: get_products
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_products_returns_list():
    client = _make_client()

    client._api.get_products.return_value = [
        {
            "id": 1,
            "name": "Перга",
            "category": "Продукты пчеловодства",
            "price": 500.0,
            "in_stock": True,
        },
        {
            "id": 2,
            "name": "Прополис (сухой + настойка)",
            "category": "Настойки",
            "price": 300.0,
            "in_stock": True,
        },
    ]

    products = await client.get_products()

    assert len(products) == 2
    assert all(isinstance(p, Product) for p in products)
    assert products[0].name == "Перга"
    assert products[1].name == "Прополис (сухой + настойка)"


@pytest.mark.asyncio
async def test_get_products_filters_out_of_stock():
    """По умолчанию in_stock_only=True — товары не в наличии отфильтровываются."""
    client = _make_client()

    client._api.get_products.return_value = [
        {"id": 1, "name": "Перга", "in_stock": True},
        {"id": 2, "name": "Прополис", "in_stock": False},
    ]

    products = await client.get_products()
    assert len(products) == 1
    assert products[0].name == "Перга"


@pytest.mark.asyncio
async def test_get_products_all_including_out_of_stock():
    """in_stock_only=False — возвращаются все товары."""
    client = _make_client()

    client._api.get_products.return_value = [
        {"id": 1, "name": "Перга", "in_stock": True},
        {"id": 2, "name": "Прополис", "in_stock": False},
    ]

    products = await client.get_products(in_stock_only=False)
    assert len(products) == 2


@pytest.mark.asyncio
async def test_get_product_by_name_found():
    client = _make_client()

    client._api.get_products.return_value = [
        {"id": 1, "name": "Перга", "in_stock": True},
        {"id": 2, "name": "Прополис", "in_stock": False},
    ]

    product = await client.get_product_by_name("перга")

    assert product is not None
    assert product.id == 1
    assert product.name == "Перга"


@pytest.mark.asyncio
async def test_get_product_by_name_not_found():
    client = _make_client()

    client._api.get_products.return_value = [
        {"id": 1, "name": "Перга", "in_stock": True},
    ]

    product = await client.get_product_by_name("Несуществующий товар")
    assert product is None


# ---------------------------------------------------------------------------
# Тесты: get_or_create_client
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_or_create_client_existing():
    """Клиент уже существует — возвращаем найденного."""
    client = _make_client()

    client._api.get_clients.return_value = [
        {
            "id": 42,
            "name": "Иванов Иван",
            "telegram_id": "123456",
            "phone": "",
            "telegram_username": "",
            "address": "",
            "city": "",
            "source": "",
        },
    ]

    result = await client.get_or_create_client(123456)

    assert result.id == 42
    assert result.full_name == "Иванов Иван"
    assert result.telegram_id == 123456


@pytest.mark.asyncio
async def test_get_or_create_client_creates_new():
    """Клиент не найден — создаём нового."""
    client = _make_client()

    client._api.get_clients.return_value = []  # никого нет
    client._api.create_object.return_value = 99

    result = await client.get_or_create_client(777)

    assert result.id == 99
    assert result.telegram_id == 777
    client._api.create_object.assert_called_once()


@pytest.mark.asyncio
async def test_update_client():
    client = _make_client()

    await client.update_client(42, full_name="Новое Имя", city="Москва")

    client._api.set_requisites.assert_called_once()
    call_args = client._api.set_requisites.call_args
    # set_requisites(obj_id, table_id, requisites)
    assert call_args[0][0] == 42  # obj_id


# ---------------------------------------------------------------------------
# Тесты: create_order
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_create_order():
    client = _make_client()
    client._api.create_object.return_value = 10

    order = await client.create_order(
        client_id=42,
        items=[{"product_id": 1, "quantity": 2, "unit_price": 500.0}],
        delivery_method="СДЭК",
    )

    assert isinstance(order, Order)
    assert order.id == 10
    assert order.status == "Новый"
    assert order.client_id == 42

    # create_object вызван для заказа + для каждой позиции
    assert client._api.create_object.call_count == 2


# ---------------------------------------------------------------------------
# Тесты: update_order_status
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_update_order_status():
    client = _make_client()

    await client.update_order_status(10, "Подтверждён")

    client._api.set_reference_field.assert_called_once()
    call_args = client._api.set_reference_field.call_args
    assert call_args[0][0] == 10  # order_id


@pytest.mark.asyncio
async def test_update_order_status_unknown_status():
    """Неизвестный статус — ничего не делаем."""
    client = _make_client()

    await client.update_order_status(10, "НесуществующийСтатус")

    client._api.set_requisites.assert_not_called()


# ---------------------------------------------------------------------------
# Тесты: get_orders
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_orders_all():
    client = _make_client()

    client._api.get_orders.return_value = [
        {
            "id": 1,
            "number": "ORD-001",
            "client_id": 42,
            "date": "13.03.2026 00:00:00",
            "status": "Новый",
        }
    ]

    orders = await client.get_orders()

    assert len(orders) == 1
    assert orders[0].number == "ORD-001"


@pytest.mark.asyncio
async def test_get_orders_filter_by_client():
    client = _make_client()

    client._api.get_orders.return_value = [
        {"id": 1, "number": "ORD-001", "client_id": 42, "date": "13.03.2026 00:00:00", "status": "Новый"},
        {"id": 2, "number": "ORD-002", "client_id": 99, "date": "14.03.2026 00:00:00", "status": "Новый"},
    ]

    orders = await client.get_orders(client_id=42)

    assert len(orders) == 1
    assert orders[0].client_id == 42


@pytest.mark.asyncio
async def test_get_orders_filter_by_status():
    client = _make_client()

    client._api.get_orders.return_value = [
        {"id": 1, "number": "ORD-001", "client_id": 42, "date": "13.03.2026 00:00:00", "status": "Новый"},
        {"id": 2, "number": "ORD-002", "client_id": 42, "date": "14.03.2026 00:00:00", "status": "Отправлен"},
    ]

    orders = await client.get_orders(status="Отправлен")

    assert len(orders) == 1
    assert orders[0].status == "Отправлен"


# ---------------------------------------------------------------------------
# Тесты: get_order
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_order_found():
    client = _make_client()

    client._api.get_orders.return_value = [
        {
            "id": 5,
            "number": "ORD-005",
            "client_id": 7,
            "date": "12.03.2026 08:30:00",
            "status": "Отправлен",
            "tracking_number": "ABC123",
        },
    ]

    order = await client.get_order(5)

    assert order.id == 5
    assert order.tracking_number == "ABC123"


@pytest.mark.asyncio
async def test_get_order_not_found():
    client = _make_client()

    client._api.get_orders.return_value = []

    with pytest.raises(IntegramNotFoundError):
        await client.get_order(999)


# ---------------------------------------------------------------------------
# Тесты: add_order_item
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_add_order_item():
    client = _make_client()
    client._api.create_object.return_value = 55

    item_id = await client.add_order_item(order_id=10, product_id=3, qty=2, price=300.0)

    assert item_id == 55
    client._api.create_object.assert_called_once()


# ---------------------------------------------------------------------------
# Тесты: context manager
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_context_manager():
    client = _make_client()

    async with client as c:
        assert c is client

    client._api.authenticate.assert_called_once()
    client._api.close.assert_called_once()


# ---------------------------------------------------------------------------
# Тесты: Pydantic модели
# ---------------------------------------------------------------------------

def test_product_model():
    product = Product(
        id=1,
        **{"Название": "Перга", "Цена": 500.0, "В наличии": True}
    )
    assert product.name == "Перга"
    assert product.price == 500.0
    assert product.in_stock is True


def test_client_model():
    client = Client(
        id=42,
        **{"ФИО": "Иванов Иван", "Telegram ID": 123456}
    )
    assert client.full_name == "Иванов Иван"
    assert client.telegram_id == 123456


def test_order_model():
    order = Order(
        id=1,
        client_id=42,
        **{
            "Номер": "ORD-001",
            "Дата": datetime(2026, 3, 13),
            "Статус": "Новый",
        }
    )
    assert order.number == "ORD-001"
    assert order.status == "Новый"
    assert order.items == []


def test_order_item_model():
    item = OrderItem(
        id=1,
        order_id=10,
        product_id=2,
        **{"Количество": 3, "Цена за шт.": 200.0, "Сумма": 600.0}
    )
    assert item.quantity == 3
    assert item.unit_price == 200.0
    assert item.total == 600.0
