"""Python-клиент для Integram CRM v2 (ai2o.online).

Совместим по интерфейсу с IntegramClient (src/integram_client.py),
но работает с API v2 через /api/v2/{workspace}/ai/tool.

Конфигурация через .env:
  INTEGRAM_V2_URL       — базовый URL (https://ai2o.online)
  INTEGRAM_V2_EMAIL     — email для аутентификации
  INTEGRAM_V2_PASSWORD  — пароль
  INTEGRAM_V2_WORKSPACE — slug воркспейса (alekseymavai)
"""

from __future__ import annotations

import logging
import time as _time
from datetime import datetime
from typing import Any, Optional

import httpx

from src.integram_v2_constants import (
    TABLE_PRODUCTS, TABLE_CLIENTS, TABLE_ORDERS, TABLE_ORDER_ITEMS,
    TABLE_STATUS_HISTORY, TABLE_SOURCES, TABLE_CATEGORIES,
    TABLE_STATUSES, TABLE_DELIVERY,
    REQ_PRODUCT_NAME, REQ_PRODUCT_CATEGORY, REQ_PRODUCT_PRICE,
    REQ_PRODUCT_WEIGHT, REQ_PRODUCT_DESC, REQ_PRODUCT_INSTOCK,
    REQ_PRODUCT_SKU, REQ_PRODUCT_SHORT, REQ_PRODUCT_STOCK,
    REQ_CLIENT_NAME, REQ_CLIENT_PHONE, REQ_CLIENT_TG_ID,
    REQ_CLIENT_TG_USER, REQ_CLIENT_ADDRESS, REQ_CLIENT_CITY,
    REQ_CLIENT_COMMENT, REQ_CLIENT_SOURCE,
    REQ_ORDER_DATE, REQ_ORDER_ADDRESS, REQ_ORDER_DELIVERY_COST,
    REQ_ORDER_ITEMS_TOTAL, REQ_ORDER_TOTAL, REQ_ORDER_TRACKING,
    REQ_ORDER_COMMENT, REQ_ORDER_CLIENT, REQ_ORDER_STATUS,
    REQ_ORDER_DELIVERY_METHOD, REQ_ORDER_SOURCE, REQ_ORDER_MESSENGER,
    REQ_ORDER_SHIPPED_DATE, REQ_ORDER_DELIVERED_DATE,
    REQ_ITEM_QTY, REQ_ITEM_PRICE, REQ_ITEM_SUM,
    REQ_ITEM_PRODUCT, REQ_ITEM_ORDER,
    STATUS_IDS, DELIVERY_IDS, SOURCE_IDS, CATEGORY_IDS,
)
from src.models import Client, Order, OrderItem, Product
from src.phone_utils import normalize_phone

logger = logging.getLogger(__name__)


class IntegramV2Error(Exception):
    """Базовое исключение клиента Integram v2."""


class IntegramV2AuthError(IntegramV2Error):
    """Ошибка аутентификации."""


class IntegramV2NotFoundError(IntegramV2Error):
    """Запись не найдена."""


class IntegramV2Client:
    """Async-клиент для Integram CRM v2 (ai2o.online).

    Все CRUD-операции идут через /api/v2/{workspace}/ai/tool.

    Использование::

        client = IntegramV2Client()
        await client.authenticate()
        products = await client.get_products()
    """

    def __init__(
        self,
        base_url: str | None = None,
        email: str | None = None,
        password: str | None = None,
        workspace: str | None = None,
    ) -> None:
        import os
        self._base_url = (base_url or os.getenv("INTEGRAM_V2_URL", "https://ai2o.online")).rstrip("/")
        self._email = email or os.getenv("INTEGRAM_V2_EMAIL", "")
        self._password = password or os.getenv("INTEGRAM_V2_PASSWORD", "")
        self._workspace = workspace or os.getenv("INTEGRAM_V2_WORKSPACE", "alekseymavai")
        self._client = httpx.AsyncClient(timeout=30)
        self._token: str | None = None
        self._token_exp: float = 0
        self._authenticated = False

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    async def authenticate(self) -> None:
        """Аутентификация через /api/v2/iam/login."""
        try:
            resp = await self._client.post(
                f"{self._base_url}/api/v2/iam/login",
                json={"email": self._email, "password": self._password},
            )
            resp.raise_for_status()
            data = resp.json()
            self._token = data.get("accessToken")
            if not self._token:
                raise IntegramV2AuthError(f"No accessToken in response: {data}")
            self._token_exp = _time.time() + 3500  # ~1 hour
            self._authenticated = True
            logger.info("Аутентификация в Integram v2 прошла успешно.")
        except httpx.HTTPError as e:
            raise IntegramV2AuthError(f"Auth failed: {e}") from e

    async def _ensure_auth(self) -> None:
        """Проверить и обновить токен если нужно."""
        if not self._token or _time.time() > self._token_exp:
            await self.authenticate()

    async def close(self) -> None:
        """Закрыть HTTP-сессию."""
        await self._client.aclose()

    async def __aenter__(self) -> "IntegramV2Client":
        await self.authenticate()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Low-level tool call
    # ------------------------------------------------------------------

    async def _call_tool(self, name: str, args: dict, retries: int = 3) -> Any:
        """Вызвать AI-tool через /api/v2/{workspace}/ai/tool."""
        await self._ensure_auth()
        last_err = None
        for attempt in range(retries):
            try:
                resp = await self._client.post(
                    f"{self._base_url}/api/v2/{self._workspace}/ai/tool",
                    json={"name": name, "args": args, "skipHitl": True},
                    headers={"Authorization": f"Bearer {self._token}"},
                )
                data = resp.json()
                if data.get("ok"):
                    return data.get("data", data)
                err = data.get("error", {})
                if err.get("code") == "AUTH_REQUIRED":
                    await self.authenticate()
                    continue
                last_err = f"{err.get('code')}: {err.get('message')}"
            except httpx.HTTPError as e:
                last_err = str(e)
            if attempt < retries - 1:
                await _async_sleep(2 ** attempt)
        raise IntegramV2Error(f"Tool '{name}' failed after {retries} retries: {last_err}")

    # ------------------------------------------------------------------
    # Товары
    # ------------------------------------------------------------------

    async def get_products(self, in_stock_only: bool = True) -> list[Product]:
        """Получить список товаров."""
        data = await self._call_tool("list_objects", {"typeId": TABLE_PRODUCTS, "limit": 500})
        products = []
        for row in data.get("rows", []):
            in_stock = _parse_bool(row.get("В наличии"))
            stock = _parse_float(row.get("Остаток"))
            product = Product(
                id=row["id"],
                name=row.get("Название", row.get("name", "")),
                category=_extract_ref_name(row.get("Категория")),
                price=_parse_float(row.get("Цена")),
                weight=_parse_float(row.get("Вес")),
                description=row.get("Описание"),
                in_stock=in_stock if in_stock is not None else True,
                sku_uds=row.get("Артикул UDS"),
                short_name=row.get("Короткое название"),
                stock=stock,
            )
            if in_stock_only and not product.in_stock:
                continue
            products.append(product)
        return products

    async def get_product_by_name(self, name: str) -> Optional[Product]:
        """Найти товар по названию."""
        products = await self.get_products(in_stock_only=False)
        name_lower = name.lower()
        for p in products:
            if p.name.lower() == name_lower:
                return p
        return None

    async def get_product_by_sku(self, sku_uds: str) -> Optional[Product]:
        """Найти товар по артикулу UDS."""
        if not sku_uds:
            return None
        products = await self.get_products(in_stock_only=False)
        for p in products:
            if p.sku_uds and p.sku_uds == sku_uds:
                return p
        return None

    async def create_product(self, name: str, **kwargs: Any) -> int:
        """Создать товар. Возвращает ID."""
        fields: dict[str, Any] = {"Название": name}
        if kwargs.get("category") and kwargs["category"] in CATEGORY_IDS:
            fields["Категория"] = int(CATEGORY_IDS[kwargs["category"]])
        if kwargs.get("price") is not None:
            fields["Цена"] = kwargs["price"]
        if kwargs.get("weight") is not None:
            fields["Вес"] = kwargs["weight"]
        if kwargs.get("description"):
            fields["Описание"] = kwargs["description"]
        if kwargs.get("in_stock") is not None:
            fields["В наличии"] = bool(kwargs["in_stock"])
        if kwargs.get("sku_uds"):
            fields["Артикул UDS"] = kwargs["sku_uds"]
        if kwargs.get("short_name"):
            fields["Короткое название"] = kwargs["short_name"]
        if kwargs.get("stock") is not None:
            fields["Остаток"] = kwargs["stock"]
        result = await self._call_tool("create_object", {"typeId": TABLE_PRODUCTS, "fields": fields})
        return result["id"]

    async def update_product(self, product_id: int, **kwargs: Any) -> None:
        """Обновить поля товара."""
        fields: dict[str, Any] = {}
        if "name" in kwargs:
            fields["Название"] = kwargs["name"]
        if "price" in kwargs:
            fields["Цена"] = kwargs["price"]
        if "weight" in kwargs:
            fields["Вес"] = kwargs["weight"]
        if "description" in kwargs:
            fields["Описание"] = kwargs["description"]
        if "in_stock" in kwargs:
            fields["В наличии"] = bool(kwargs["in_stock"])
        if "sku_uds" in kwargs:
            fields["Артикул UDS"] = kwargs["sku_uds"]
        if "short_name" in kwargs:
            fields["Короткое название"] = kwargs["short_name"]
        if "stock" in kwargs:
            fields["Остаток"] = kwargs["stock"]
        if "category" in kwargs and kwargs["category"] in CATEGORY_IDS:
            fields["Категория"] = int(CATEGORY_IDS[kwargs["category"]])
        if fields:
            await self._call_tool("update_object", {"objectId": product_id, "fields": fields})

    async def update_product_stock(self, product_id: int, stock: int) -> None:
        """Обновить остаток товара."""
        await self._call_tool("update_object", {
            "objectId": product_id,
            "fields": {"Остаток": stock, "В наличии": stock > 0},
        })

    # ------------------------------------------------------------------
    # Клиенты
    # ------------------------------------------------------------------

    async def get_clients(self) -> list[Client]:
        """Получить всех клиентов."""
        data = await self._call_tool("list_objects", {"typeId": TABLE_CLIENTS, "limit": 5000})
        return [
            Client(
                id=row["id"],
                full_name=row.get("ФИО", row.get("name", "")),
                phone=row.get("Телефон") or None,
                telegram_id=_parse_int(row.get("Telegram ID")),
                telegram_username=row.get("Telegram Username") or None,
                address=row.get("Адрес") or None,
                city=row.get("Город") or None,
                source=_extract_ref_name(row.get("Источник")),
            )
            for row in data.get("rows", [])
        ]

    async def get_or_create_client(self, telegram_id: int, **kwargs: Any) -> Client:
        """Получить клиента по Telegram ID / телефону или создать нового."""
        clients = await self.get_clients()

        if telegram_id:
            for c in clients:
                if c.telegram_id == telegram_id:
                    return c

        phone = kwargs.get("phone")
        if phone:
            phone = normalize_phone(phone) or phone
            phone_digits = "".join(ch for ch in phone if ch.isdigit())
            for c in clients:
                if c.phone and "".join(ch for ch in c.phone if ch.isdigit()) == phone_digits:
                    return c

        full_name = kwargs.get("full_name", f"Telegram {telegram_id}")
        fields: dict[str, Any] = {"ФИО": full_name}
        if phone:
            fields["Телефон"] = phone
        if telegram_id:
            fields["Telegram ID"] = telegram_id
        if kwargs.get("telegram_username"):
            fields["Telegram Username"] = kwargs["telegram_username"]
        if kwargs.get("address"):
            fields["Адрес"] = kwargs["address"]
        if kwargs.get("city"):
            fields["Город"] = kwargs["city"]
        source = kwargs.get("source", "Telegram")
        if source in SOURCE_IDS:
            fields["Источник"] = int(SOURCE_IDS[source])

        result = await self._call_tool("create_object", {"typeId": TABLE_CLIENTS, "fields": fields})
        obj_id = result["id"]
        logger.info("Создан клиент '%s' (id=%d) в Integram v2.", full_name, obj_id)

        return Client(
            id=obj_id,
            full_name=full_name,
            phone=phone,
            telegram_id=telegram_id or None,
            telegram_username=kwargs.get("telegram_username"),
            address=kwargs.get("address"),
            city=kwargs.get("city"),
            source=source,
        )

    async def update_client(self, client_id: int, **kwargs: Any) -> None:
        """Обновить данные клиента."""
        fields: dict[str, Any] = {}
        if "full_name" in kwargs:
            fields["ФИО"] = kwargs["full_name"]
        if "phone" in kwargs:
            val = kwargs["phone"]
            fields["Телефон"] = normalize_phone(str(val)) or val if val else ""
        if "telegram_id" in kwargs:
            fields["Telegram ID"] = kwargs["telegram_id"]
        if "telegram_username" in kwargs:
            fields["Telegram Username"] = kwargs["telegram_username"]
        if "address" in kwargs:
            fields["Адрес"] = kwargs["address"]
        if "city" in kwargs:
            fields["Город"] = kwargs["city"]
        source = kwargs.get("source")
        if source and source in SOURCE_IDS:
            fields["Источник"] = int(SOURCE_IDS[source])
        if fields:
            await self._call_tool("update_object", {"objectId": client_id, "fields": fields})
            logger.info("Клиент %d обновлён: %s", client_id, list(kwargs.keys()))

    # ------------------------------------------------------------------
    # Заказы
    # ------------------------------------------------------------------

    async def get_orders(
        self,
        client_id: Optional[int] = None,
        status: Optional[str] = None,
    ) -> list[Order]:
        """Получить список заказов."""
        data = await self._call_tool("list_objects", {"typeId": TABLE_ORDERS, "limit": 5000})
        orders = []
        for row in data.get("rows", []):
            order = self._row_to_order(row)
            if client_id is not None and order.client_id != client_id:
                continue
            if status is not None and order.status != status:
                continue
            orders.append(order)
        return orders

    async def get_order(self, order_id: int) -> Order:
        """Получить заказ по ID."""
        data = await self._call_tool("get_object", {"objectId": order_id, "typeId": TABLE_ORDERS})
        if not data:
            raise IntegramV2NotFoundError(f"Заказ {order_id} не найден.")
        return self._row_to_order(data)

    async def create_order(
        self,
        client_id: int,
        items: list[dict],
        **kwargs: Any,
    ) -> Order:
        """Создать заказ в Integram CRM v2."""
        items_total = kwargs.get("items_total") or sum(
            i.get("quantity", 1) * i.get("unit_price", 0) for i in items
        )
        delivery_cost = kwargs.get("delivery_cost", 0)
        total = kwargs.get("total") or (items_total + delivery_cost)
        number = kwargs.get("number", f"TG-{datetime.now().strftime('%Y%m%d-%H%M')}")
        status = kwargs.get("status", "Новый")
        source = kwargs.get("source", "Telegram")
        delivery_method = kwargs.get("delivery_method", "")
        order_date = kwargs.get("date") or datetime.now()

        fields: dict[str, Any] = {"Дата": order_date.strftime("%Y-%m-%dT%H:%M:%S")}
        if client_id:
            fields["Клиент"] = client_id
        if status in STATUS_IDS:
            fields["Статус"] = int(STATUS_IDS[status])
        if delivery_method in DELIVERY_IDS:
            fields["Способ доставки"] = int(DELIVERY_IDS[delivery_method])
        if source in SOURCE_IDS:
            fields["Источник"] = int(SOURCE_IDS[source])
        if kwargs.get("delivery_address"):
            fields["Адрес доставки"] = kwargs["delivery_address"]
        if delivery_cost:
            fields["Стоимость доставки"] = delivery_cost
        fields["Сумма товаров"] = items_total
        fields["Итого"] = total
        if kwargs.get("messenger"):
            fields["Мессенджер"] = kwargs["messenger"]

        result = await self._call_tool("create_object", {"typeId": TABLE_ORDERS, "fields": fields})
        obj_id = result["id"]
        logger.info("Создан заказ '%s' (id=%d). Итого: %.0f ₽", number, obj_id, total)

        # Создать позиции
        for item_data in items:
            qty = item_data.get("quantity", 1)
            price = item_data.get("unit_price", 0)
            item_fields = {
                "Заказ": obj_id,
                "Товар": item_data["product_id"],
                "Количество": qty,
                "Цена за шт.": price,
                "Сумма": qty * price,
            }
            await self._call_tool("create_object", {"typeId": TABLE_ORDER_ITEMS, "fields": item_fields})

        return Order(
            id=obj_id, number=number, client_id=client_id, date=datetime.now(),
            status=status, delivery_method=delivery_method,
            delivery_address=kwargs.get("delivery_address"),
            delivery_cost=delivery_cost, items_total=items_total,
            total=total, source=source, items=[],
        )

    async def update_order_status(
        self, order_id: int, status: str, *, from_status: str | None = None, comment: str = ""
    ) -> None:
        """Обновить статус заказа."""
        if status not in STATUS_IDS:
            logger.warning("Неизвестный статус '%s'.", status)
            return

        if from_status is None:
            try:
                current = await self.get_order(order_id)
                from_status = current.status
            except Exception:
                from_status = ""

        fields: dict[str, Any] = {"Статус": int(STATUS_IDS[status])}
        if status == "Отправлен":
            fields["Дата отправки"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        if status == "Доставлен":
            fields["Дата доставки"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        await self._call_tool("update_object", {"objectId": order_id, "fields": fields})
        logger.info("Статус заказа %d: '%s' → '%s'", order_id, from_status, status)

    async def update_order(self, order_id: int, **kwargs: Any) -> None:
        """Обновить поля заказа."""
        fields: dict[str, Any] = {}
        field_map = {
            "delivery_address": "Адрес доставки",
            "delivery_cost": "Стоимость доставки",
            "items_total": "Сумма товаров",
            "total": "Итого",
            "comment": "Комментарий",
            "tracking_number": "Трек-номер",
        }
        for py_key, col_name in field_map.items():
            if py_key in kwargs:
                fields[col_name] = kwargs[py_key]
        if "delivery_method" in kwargs and kwargs["delivery_method"] in DELIVERY_IDS:
            fields["Способ доставки"] = int(DELIVERY_IDS[kwargs["delivery_method"]])
        if "status" in kwargs and kwargs["status"] in STATUS_IDS:
            fields["Статус"] = int(STATUS_IDS[kwargs["status"]])
        if "source" in kwargs and kwargs["source"] in SOURCE_IDS:
            fields["Источник"] = int(SOURCE_IDS[kwargs["source"]])
        if fields:
            await self._call_tool("update_object", {"objectId": order_id, "fields": fields})
            logger.info("Заказ %d обновлён: %s", order_id, list(kwargs.keys()))

    async def get_order_items(self, order_id: int) -> list[OrderItem]:
        """Получить позиции заказа."""
        data = await self._call_tool("list_objects", {"typeId": TABLE_ORDER_ITEMS, "limit": 5000})
        items = []
        for row in data.get("rows", []):
            ref_order = _extract_ref_id(row.get("Заказ"))
            if ref_order != order_id:
                continue
            items.append(OrderItem(
                id=row["id"],
                order_id=order_id,
                product_id=_extract_ref_id(row.get("Товар")) or 0,
                product_name=_extract_ref_name(row.get("Товар")),
                quantity=int(_parse_float(row.get("Количество")) or 1),
                unit_price=_parse_float(row.get("Цена за шт.")) or 0,
                total=_parse_float(row.get("Сумма")) or 0,
            ))
        return items

    async def get_order_items_bulk(self) -> list[OrderItem]:
        """Получить ВСЕ позиции заказов."""
        data = await self._call_tool("list_objects", {"typeId": TABLE_ORDER_ITEMS, "limit": 10000})
        return [
            OrderItem(
                id=row["id"],
                order_id=_extract_ref_id(row.get("Заказ")) or 0,
                product_id=_extract_ref_id(row.get("Товар")) or 0,
                product_name=_extract_ref_name(row.get("Товар")),
                quantity=int(_parse_float(row.get("Количество")) or 1),
                unit_price=_parse_float(row.get("Цена за шт.")) or 0,
                total=_parse_float(row.get("Сумма")) or 0,
            )
            for row in data.get("rows", [])
        ]

    async def add_order_item(
        self, order_id: int, product_id: int, qty: int, price: float,
    ) -> int:
        """Добавить позицию к заказу."""
        fields = {
            "Заказ": order_id,
            "Товар": product_id,
            "Количество": qty,
            "Цена за шт.": int(price),
            "Сумма": int(qty * price),
        }
        result = await self._call_tool("create_object", {"typeId": TABLE_ORDER_ITEMS, "fields": fields})
        return result["id"]

    async def recalculate_order_totals(self, order_id: int) -> dict:
        """Пересчитать суммы заказа."""
        items = await self.get_order_items(order_id)
        items_total = sum(i.quantity * i.unit_price for i in items)
        order = await self.get_order(order_id)
        delivery_cost = order.delivery_cost or 0
        total = items_total + delivery_cost
        await self.update_order(order_id, items_total=items_total, total=total)
        return {"items_total": items_total, "delivery_cost": delivery_cost, "total": total}

    async def get_client_telegram_id(self, client_id: int) -> Optional[int]:
        """Получить Telegram ID клиента."""
        clients = await self.get_clients()
        for c in clients:
            if c.id == client_id and c.telegram_id:
                return c.telegram_id
        return None

    # ------------------------------------------------------------------
    # Вспомогательные
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_order(row: dict) -> Order:
        """Конвертировать row из list_objects в Order."""
        date_str = row.get("Дата", "")
        try:
            date = datetime.fromisoformat(date_str) if date_str else datetime.now()
        except ValueError:
            date = datetime.now()

        return Order(
            id=row.get("id", 0),
            number=row.get("name", ""),
            client_id=_extract_ref_id(row.get("Клиент")) or 0,
            client_name=_extract_ref_name(row.get("Клиент")),
            date=date,
            status=_extract_ref_name(row.get("Статус")) or "",
            delivery_method=_extract_ref_name(row.get("Способ доставки")),
            delivery_address=row.get("Адрес доставки"),
            delivery_cost=_parse_float(row.get("Стоимость доставки")),
            items_total=_parse_float(row.get("Сумма товаров")),
            total=_parse_float(row.get("Итого")),
            tracking_number=row.get("Трек-номер"),
            source=_extract_ref_name(row.get("Источник")),
            comment=row.get("Комментарий"),
            messenger=row.get("Мессенджер"),
            items=[],
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_float(val: Any) -> Optional[float]:
    if val is None or val == "":
        return None
    try:
        return float(str(val).replace(",", "."))
    except (ValueError, TypeError):
        return None


def _parse_int(val: Any) -> Optional[int]:
    if val is None or val == "":
        return None
    try:
        return int(float(str(val)))
    except (ValueError, TypeError):
        return None


def _parse_bool(val: Any) -> Optional[bool]:
    if val is None or val == "":
        return None
    s = str(val).lower()
    return s in ("true", "1", "yes")


def _extract_ref_name(val: Any) -> Optional[str]:
    """Извлечь имя из ref-значения вида 'Название (id:123)' или просто строки."""
    if val is None:
        return None
    s = str(val)
    if " (id:" in s:
        return s.split(" (id:")[0]
    return s or None


def _extract_ref_id(val: Any) -> Optional[int]:
    """Извлечь ID из ref-значения вида 'Название (id:123)'."""
    if val is None:
        return None
    s = str(val)
    if "(id:" in s:
        try:
            return int(s.split("(id:")[1].rstrip(")"))
        except (ValueError, IndexError):
            pass
    try:
        return int(s)
    except (ValueError, TypeError):
        return None


async def _async_sleep(seconds: float) -> None:
    """Async sleep helper."""
    import asyncio
    await asyncio.sleep(seconds)
