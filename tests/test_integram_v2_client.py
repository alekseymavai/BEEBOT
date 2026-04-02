"""Тесты для IntegramV2Client (src/integram_v2_client.py).

Mock _call_tool — реальных HTTP-запросов к ai2o.online не делают.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest

from src.integram_v2_client import (
    IntegramV2Client,
    IntegramV2AuthError,
    IntegramV2Error,
    IntegramV2NotFoundError,
    _extract_ref_id,
    _extract_ref_name,
    _parse_float,
    _parse_bool,
)
from src.models import Client, Order, OrderItem, Product


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_client() -> IntegramV2Client:
    """Создать IntegramV2Client с замоканным _call_tool."""
    client = IntegramV2Client(
        base_url="https://test.example.com",
        email="test@test.com",
        password="test",
        workspace="test",
    )
    client._token = "fake-token"
    client._token_exp = 9999999999
    client._authenticated = True
    client._call_tool = AsyncMock(return_value={})
    return client


SAMPLE_PRODUCTS_RESPONSE = {
    "rows": [
        {
            "id": 591,
            "name": "Перга",
            "Название": "Перга",
            "Категория": "Продукты пчеловодства (id:157)",
            "Цена": "1400",
            "Остаток": "24",
            "В наличии": "true",
            "Артикул UDS": "",
            "Короткое название": "",
        },
        {
            "id": 603,
            "name": "Прополис",
            "Название": "Прополис",
            "Категория": "Настойки (id:159)",
            "Цена": "1000",
            "Остаток": "0",
            "В наличии": "false",
        },
    ],
}

SAMPLE_CLIENTS_RESPONSE = {
    "rows": [
        {
            "id": 100,
            "name": "Иванов Иван",
            "ФИО": "Иванов Иван",
            "Телефон": "+79001234567",
            "Telegram ID": "123456",
            "Telegram Username": "ivanov",
            "Город": "Москва",
        },
    ],
}

SAMPLE_ORDERS_RESPONSE = {
    "rows": [
        {
            "id": 200,
            "name": "TG-20260402-1200",
            "Дата": "2026-04-02T12:00:00",
            "Клиент": "Иванов Иван (id:100)",
            "Статус": "Новый (id:179)",
            "Способ доставки": "СДЭК (id:191)",
            "Сумма товаров": "2400",
            "Итого": "2900",
            "Стоимость доставки": "500",
            "Трек-номер": "",
            "Источник": "Telegram (id:19)",
        },
    ],
}

SAMPLE_ITEMS_RESPONSE = {
    "rows": [
        {
            "id": 300,
            "Заказ": "TG-20260402-1200 (id:200)",
            "Товар": "Перга (id:591)",
            "Количество": "2",
            "Цена за шт.": "1400",
            "Сумма": "2800",
        },
    ],
}


# ---------------------------------------------------------------------------
# Тесты: authenticate
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_authenticate_success():
    client = IntegramV2Client(
        base_url="https://test.example.com",
        email="test@test.com",
        password="test",
        workspace="test",
    )
    import httpx
    req = httpx.Request("POST", "https://test.example.com/api/v2/iam/login")
    mock_response = httpx.Response(200, json={"accessToken": "token123"}, request=req)

    with patch.object(client._client, "post", new_callable=AsyncMock, return_value=mock_response):
        await client.authenticate()

    assert client._authenticated is True
    assert client._token == "token123"


@pytest.mark.asyncio
async def test_authenticate_no_token():
    client = IntegramV2Client(
        base_url="https://test.example.com",
        email="test@test.com",
        password="test",
        workspace="test",
    )
    import httpx
    req = httpx.Request("POST", "https://test.example.com/api/v2/iam/login")
    mock_response = httpx.Response(200, json={"ok": True}, request=req)

    with patch.object(client._client, "post", new_callable=AsyncMock, return_value=mock_response):
        with pytest.raises(IntegramV2AuthError, match="No accessToken"):
            await client.authenticate()


# ---------------------------------------------------------------------------
# Тесты: get_products
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_products_returns_list():
    client = _make_client()
    client._call_tool.return_value = SAMPLE_PRODUCTS_RESPONSE

    products = await client.get_products(in_stock_only=False)

    assert len(products) == 2
    assert all(isinstance(p, Product) for p in products)
    assert products[0].name == "Перга"
    assert products[0].price == 1400.0
    assert products[0].category == "Продукты пчеловодства"
    assert products[1].name == "Прополис"


@pytest.mark.asyncio
async def test_get_products_filters_out_of_stock():
    client = _make_client()
    client._call_tool.return_value = SAMPLE_PRODUCTS_RESPONSE

    products = await client.get_products(in_stock_only=True)

    assert len(products) == 1
    assert products[0].name == "Перга"


@pytest.mark.asyncio
async def test_get_product_by_name():
    client = _make_client()
    client._call_tool.return_value = SAMPLE_PRODUCTS_RESPONSE

    product = await client.get_product_by_name("перга")
    assert product is not None
    assert product.id == 591

    missing = await client.get_product_by_name("несуществующий")
    assert missing is None


# ---------------------------------------------------------------------------
# Тесты: get_clients
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_clients():
    client = _make_client()
    client._call_tool.return_value = SAMPLE_CLIENTS_RESPONSE

    clients = await client.get_clients()

    assert len(clients) == 1
    assert isinstance(clients[0], Client)
    assert clients[0].full_name == "Иванов Иван"
    assert clients[0].telegram_id == 123456
    assert clients[0].city == "Москва"


@pytest.mark.asyncio
async def test_get_or_create_client_existing():
    client = _make_client()
    client._call_tool.return_value = SAMPLE_CLIENTS_RESPONSE

    result = await client.get_or_create_client(123456)

    assert result.id == 100
    assert result.full_name == "Иванов Иван"
    # Should NOT call create_object (only list_objects for get_clients)
    assert client._call_tool.call_count == 1


@pytest.mark.asyncio
async def test_get_or_create_client_creates_new():
    client = _make_client()

    # First call: get_clients returns empty, second call: create_object
    client._call_tool.side_effect = [
        {"rows": []},  # get_clients
        {"id": 999, "typeId": 52, "name": "Новый Клиент"},  # create_object
    ]

    result = await client.get_or_create_client(
        telegram_id=999999,
        full_name="Новый Клиент",
        phone="+79001111111",
    )

    assert result.id == 999
    assert result.telegram_id == 999999
    assert client._call_tool.call_count == 2


# ---------------------------------------------------------------------------
# Тесты: orders
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_orders():
    client = _make_client()
    client._call_tool.return_value = SAMPLE_ORDERS_RESPONSE

    orders = await client.get_orders()

    assert len(orders) == 1
    assert isinstance(orders[0], Order)
    assert orders[0].id == 200
    assert orders[0].client_id == 100
    assert orders[0].status == "Новый"
    assert orders[0].total == 2900.0


@pytest.mark.asyncio
async def test_get_orders_filter_by_status():
    client = _make_client()
    client._call_tool.return_value = SAMPLE_ORDERS_RESPONSE

    orders = await client.get_orders(status="Отправлен")
    assert len(orders) == 0

    orders = await client.get_orders(status="Новый")
    assert len(orders) == 1


@pytest.mark.asyncio
async def test_get_order_not_found():
    client = _make_client()
    client._call_tool.return_value = None

    with pytest.raises(IntegramV2NotFoundError):
        await client.get_order(9999)


@pytest.mark.asyncio
async def test_create_order():
    client = _make_client()
    # create_object for order, then create_object for each item
    client._call_tool.side_effect = [
        {"id": 500, "typeId": 60, "name": "TG-test"},  # create order
        {"id": 501, "typeId": 78, "name": "item"},  # create item
    ]

    order = await client.create_order(
        client_id=100,
        items=[{"product_id": 591, "quantity": 2, "unit_price": 1400}],
        delivery_address="Москва, ул. Тестовая",
        source="Telegram",
    )

    assert order.id == 500
    assert order.items_total == 2800
    assert order.total == 2800
    assert client._call_tool.call_count == 2  # order + 1 item


@pytest.mark.asyncio
async def test_update_order_status():
    client = _make_client()
    # First call: get_order (for from_status), second: update_object
    client._call_tool.side_effect = [
        SAMPLE_ORDERS_RESPONSE["rows"][0],  # get_order
        None,  # update_object
    ]

    await client.update_order_status(200, "Подтверждён")

    assert client._call_tool.call_count == 2


@pytest.mark.asyncio
async def test_update_order_status_unknown():
    client = _make_client()

    await client.update_order_status(200, "НесуществующийСтатус")
    # Should not call _call_tool at all
    client._call_tool.assert_not_called()


# ---------------------------------------------------------------------------
# Тесты: order items
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_order_items():
    client = _make_client()
    client._call_tool.return_value = SAMPLE_ITEMS_RESPONSE

    items = await client.get_order_items(200)

    assert len(items) == 1
    assert isinstance(items[0], OrderItem)
    assert items[0].product_id == 591
    assert items[0].quantity == 2
    assert items[0].unit_price == 1400.0


@pytest.mark.asyncio
async def test_get_order_items_filters_by_order():
    client = _make_client()
    client._call_tool.return_value = SAMPLE_ITEMS_RESPONSE

    items = await client.get_order_items(999)
    assert len(items) == 0


@pytest.mark.asyncio
async def test_add_order_item():
    client = _make_client()
    client._call_tool.return_value = {"id": 400}

    item_id = await client.add_order_item(200, 591, 3, 1400.0)

    assert item_id == 400
    call_args = client._call_tool.call_args
    assert call_args[0][0] == "create_object"


@pytest.mark.asyncio
async def test_delete_order_item():
    client = _make_client()
    await client.delete_order_item(300)
    client._call_tool.assert_called_once_with("delete_object", {"objectId": 300})


# ---------------------------------------------------------------------------
# Тесты: products CRUD
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_create_product():
    client = _make_client()
    client._call_tool.return_value = {"id": 700}

    product_id = await client.create_product(
        "Новый мёд", category="Мёд", price=1200, stock=50,
    )

    assert product_id == 700


@pytest.mark.asyncio
async def test_delete_product_soft():
    client = _make_client()
    await client.delete_product(591)

    call_args = client._call_tool.call_args
    assert call_args[0][0] == "update_object"
    assert call_args[0][1]["fields"]["В наличии"] is False


# ---------------------------------------------------------------------------
# Тесты: dashboard
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_dashboard_stats():
    client = _make_client()
    client._call_tool.side_effect = [
        SAMPLE_ORDERS_RESPONSE,    # get_orders
        SAMPLE_CLIENTS_RESPONSE,   # get_clients
        SAMPLE_PRODUCTS_RESPONSE,  # get_products
    ]

    stats = await client.get_dashboard_stats()

    assert stats["total_orders"] == 1
    assert stats["total_clients"] == 1
    assert stats["total_products"] == 2
    assert stats["total_revenue"] == 2900.0
    assert stats["active_orders"] == 1


# ---------------------------------------------------------------------------
# Тесты: checklist
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_update_order_checklist():
    client = _make_client()
    await client.update_order_checklist(200, cdek_confirmed=True, stock_checked=False)

    call_args = client._call_tool.call_args
    fields = call_args[0][1]["fields"]
    assert fields["Адрес СДЭК уточнён"] is True
    assert fields["Наличие проверено"] is False


# ---------------------------------------------------------------------------
# Тесты: helpers
# ---------------------------------------------------------------------------

def test_extract_ref_name():
    assert _extract_ref_name("Перга (id:591)") == "Перга"
    assert _extract_ref_name("Простой текст") == "Простой текст"
    assert _extract_ref_name(None) is None


def test_extract_ref_id():
    assert _extract_ref_id("Перга (id:591)") == 591
    assert _extract_ref_id("123") == 123
    assert _extract_ref_id(None) is None
    assert _extract_ref_id("текст") is None


def test_parse_float():
    assert _parse_float("1400") == 1400.0
    assert _parse_float("1,5") == 1.5
    assert _parse_float(None) is None
    assert _parse_float("") is None


def test_parse_bool():
    assert _parse_bool("true") is True
    assert _parse_bool("1") is True
    assert _parse_bool("false") is False
    assert _parse_bool(None) is None


# ---------------------------------------------------------------------------
# Тесты: context manager
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_context_manager():
    import httpx
    req = httpx.Request("POST", "https://test.example.com/api/v2/iam/login")
    mock_response = httpx.Response(200, json={"accessToken": "tok"}, request=req)

    client = IntegramV2Client(
        base_url="https://test.example.com",
        email="t@t.com", password="t", workspace="t",
    )
    with patch.object(client._client, "post", new_callable=AsyncMock, return_value=mock_response):
        with patch.object(client._client, "aclose", new_callable=AsyncMock):
            async with client as c:
                assert c._authenticated is True
