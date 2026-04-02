"""Python-клиент для Integram CRM (ai2o.ru).

Обёртка над IntegramAPI (реальный HTTP-клиент) для runtime-кода бота.
Возвращает типизированные Pydantic-модели (Product, Client, Order).

Конфигурация через .env:
  INTEGRAM_URL      — базовый URL (https://ai2o.ru)
  INTEGRAM_LOGIN    — логин
  INTEGRAM_PASSWORD — пароль
  INTEGRAM_DB       — имя базы (bibot)
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Optional

from src.integram_api import (
    IntegramAPI,
    IntegramAPIError,
    TABLE_ORDERS,
    TABLE_CLIENTS,
    TABLE_PRODUCTS,
    TABLE_ORDER_ITEMS,
    REQ_ORDER_ADDRESS,
    REQ_ORDER_DELIVERY_COST,
    REQ_ORDER_ITEMS_TOTAL,
    REQ_ORDER_TOTAL,
    REQ_ORDER_TRACKING,
    REQ_ORDER_COMMENT,
    REQ_ORDER_CLIENT,
    REQ_ORDER_STATUS,
    REQ_ORDER_DELIVERY_METHOD,
    REQ_ORDER_SOURCE,
    REQ_ORDER_MESSENGER,
    REQ_ORDER_DATE,
    REQ_ORDER_SHIPPED_DATE,
    REQ_ORDER_DELIVERED_DATE,
    TABLE_STATUS_HISTORY,
    REQ_HISTORY_ORDER,
    REQ_HISTORY_STATUS_FROM,
    REQ_HISTORY_STATUS_TO,
    REQ_HISTORY_DATE,
    REQ_HISTORY_COMMENT,
    REQ_CLIENT_PHONE,
    REQ_CLIENT_TG_ID,
    REQ_CLIENT_TG_USER,
    REQ_CLIENT_ADDRESS,
    REQ_CLIENT_CITY,
    REQ_CLIENT_SOURCE,
    REQ_ITEM_QTY,
    REQ_ITEM_PRICE,
    REQ_ITEM_SUM,
    REQ_ITEM_PRODUCT,
    REQ_ITEM_ORDER,
    REQ_PRODUCT_PRICE,
    REQ_PRODUCT_WEIGHT,
    REQ_PRODUCT_DESC,
    REQ_PRODUCT_INSTOCK,
    REQ_PRODUCT_SKU,
    REQ_PRODUCT_CATEGORY,
    REQ_PRODUCT_SHORT,
    REQ_PRODUCT_STOCK,
)
from src.crm_constants import STATUS_IDS, DELIVERY_IDS, SOURCE_IDS, CATEGORY_IDS
from src.models import Client, Order, OrderItem, Product
from src.phone_utils import normalize_phone

logger = logging.getLogger(__name__)


class IntegramError(Exception):
    """Базовое исключение клиента Integram."""


class IntegramAuthError(IntegramError):
    """Ошибка аутентификации."""


class IntegramNotFoundError(IntegramError):
    """Запись не найдена."""


class IntegramClient:
    """Async-клиент для работы с Integram CRM.

    Делегирует HTTP-вызовы в IntegramAPI, возвращает Pydantic-модели.

    Использование::

        client = IntegramClient()
        await client.authenticate()
        products = await client.get_products()
    """

    def __init__(self, **kwargs: Any) -> None:
        self._api = IntegramAPI()
        self._authenticated = False

    @property
    def api(self) -> IntegramAPI:
        """Низкоуровневый API-клиент (для user management и прочих raw-операций)."""
        return self._api

    async def authenticate(self) -> None:
        """Авторизация в Integram CRM."""
        try:
            await self._api.authenticate()
            self._authenticated = True
            logger.info("Аутентификация в Integram прошла успешно.")
        except IntegramAPIError as e:
            raise IntegramAuthError(str(e)) from e

    async def close(self) -> None:
        """Закрыть HTTP-сессию."""
        await self._api.close()

    async def __aenter__(self) -> "IntegramClient":
        await self.authenticate()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Товары
    # ------------------------------------------------------------------

    async def get_products(self, in_stock_only: bool = True) -> list[Product]:
        """Получить список товаров."""
        raw = await self._api.get_products()
        products = []
        for item in raw:
            product = Product(
                id=item["id"],
                name=item["name"],
                category=item.get("category"),
                price=item.get("price"),
                weight=item.get("weight"),
                description=item.get("description"),
                in_stock=item.get("in_stock", True),
                sku_uds=item.get("sku_uds"),
                short_name=item.get("short_name"),
                stock=item.get("stock"),
            )
            if in_stock_only and not product.in_stock:
                continue
            products.append(product)
        return products

    async def get_product_by_name(self, name: str) -> Optional[Product]:
        """Найти товар по названию."""
        products = await self.get_products(in_stock_only=False)
        name_lower = name.lower()
        for product in products:
            if product.name.lower() == name_lower:
                return product
        return None

    async def get_product_by_sku(self, sku_uds: str) -> Optional[Product]:
        """Найти товар по артикулу UDS (поле sku_uds)."""
        if not sku_uds:
            return None
        products = await self.get_products(in_stock_only=False)
        for product in products:
            if product.sku_uds and product.sku_uds == sku_uds:
                return product
        return None

    # ------------------------------------------------------------------
    # Клиенты
    # ------------------------------------------------------------------

    async def get_clients(self) -> list[Client]:
        """Получить всех клиентов."""
        raw = await self._api.get_clients()
        return [
            Client(
                id=c["id"],
                full_name=c["name"],
                phone=c.get("phone") or None,
                telegram_id=int(c["telegram_id"]) if c.get("telegram_id") else None,
                telegram_username=c.get("telegram_username") or None,
                address=c.get("address") or None,
                city=c.get("city") or None,
                source=c.get("source") or None,
            )
            for c in raw
        ]

    async def get_or_create_client(
        self,
        telegram_id: int,
        **kwargs: Any,
    ) -> Client:
        """Получить клиента по Telegram ID / телефону или создать нового."""
        clients = await self.get_clients()

        # Поиск по telegram_id
        if telegram_id:
            for c in clients:
                if c.telegram_id == telegram_id:
                    return c

        # Поиск по телефону
        phone = kwargs.get("phone")
        if phone:
            phone = normalize_phone(phone) or phone  # нормализуем для поиска
            phone_digits = "".join(ch for ch in phone if ch.isdigit())
            for c in clients:
                if c.phone and "".join(ch for ch in c.phone if ch.isdigit()) == phone_digits:
                    return c

        # Создать нового клиента
        full_name = kwargs.get("full_name", f"Telegram {telegram_id}")
        reqs: dict[str, str] = {}
        if phone:
            reqs[REQ_CLIENT_PHONE] = phone  # уже нормализован выше
        if telegram_id:
            reqs[REQ_CLIENT_TG_ID] = str(telegram_id)
        if kwargs.get("telegram_username"):
            reqs[REQ_CLIENT_TG_USER] = kwargs["telegram_username"]
        if kwargs.get("address"):
            reqs[REQ_CLIENT_ADDRESS] = kwargs["address"]
        if kwargs.get("city"):
            reqs[REQ_CLIENT_CITY] = kwargs["city"]
        source = kwargs.get("source", "Telegram")
        if source in SOURCE_IDS:
            reqs[REQ_CLIENT_SOURCE] = SOURCE_IDS[source]

        obj_id = await self._api.create_object(TABLE_CLIENTS, full_name, reqs)
        logger.info("Создан клиент '%s' (id=%d) в Integram.", full_name, obj_id)

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
        """Обновить данные клиента в Integram CRM."""
        field_map = {
            "full_name": None,  # val — обновляется отдельно
            "phone": REQ_CLIENT_PHONE,
            "telegram_id": REQ_CLIENT_TG_ID,
            "telegram_username": REQ_CLIENT_TG_USER,
            "address": REQ_CLIENT_ADDRESS,
            "city": REQ_CLIENT_CITY,
        }
        reqs: dict[str, str] = {}
        for py_key, req_id in field_map.items():
            if py_key in kwargs and req_id:
                val = kwargs[py_key]
                if py_key == "phone" and val:
                    val = normalize_phone(str(val)) or val
                reqs[req_id] = str(val)
        source = kwargs.get("source")
        if source and source in SOURCE_IDS:
            reqs[REQ_CLIENT_SOURCE] = SOURCE_IDS[source]

        if reqs:
            await self._api.set_requisites(client_id, TABLE_CLIENTS, reqs)
            logger.info("Клиент %d обновлён: %s", client_id, list(kwargs.keys()))

    # ------------------------------------------------------------------
    # Заказы
    # ------------------------------------------------------------------

    async def get_orders(
        self,
        client_id: Optional[int] = None,
        status: Optional[str] = None,
    ) -> list[Order]:
        """Получить список заказов с фильтрацией."""
        raw = await self._api.get_orders()
        orders = []
        for item in raw:
            order = self._dict_to_order(item)
            if client_id is not None and order.client_id != client_id:
                continue
            if status is not None and order.status != status:
                continue
            orders.append(order)
        return orders

    async def get_order(self, order_id: int) -> Order:
        """Получить заказ по ID (быстрый единичный запрос)."""
        item = await self._api.get_order_by_id(order_id)
        if item is None:
            raise IntegramNotFoundError(f"Заказ {order_id} не найден.")
        return self._dict_to_order(item)

    async def create_order(
        self,
        client_id: int,
        items: list[dict],
        **kwargs: Any,
    ) -> Order:
        """Создать заказ в Integram CRM.

        Args:
            client_id:  ID клиента в Integram.
            items:      Список позиций [{product_id, quantity, unit_price}].
            **kwargs:   delivery_method, delivery_address, delivery_cost,
                        items_total, total, source, number, status.
        """
        # Подсчёт сумм
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

        # Собрать реквизиты
        reqs: dict[str, str] = {}
        reqs[REQ_ORDER_DATE] = order_date.strftime("%d.%m.%Y 00:00:00")
        if client_id:
            reqs[REQ_ORDER_CLIENT] = str(client_id)
        if status in STATUS_IDS:
            reqs[REQ_ORDER_STATUS] = STATUS_IDS[status]
        if delivery_method in DELIVERY_IDS:
            reqs[REQ_ORDER_DELIVERY_METHOD] = DELIVERY_IDS[delivery_method]
        if source in SOURCE_IDS:
            reqs[REQ_ORDER_SOURCE] = SOURCE_IDS[source]
        if kwargs.get("delivery_address"):
            reqs[REQ_ORDER_ADDRESS] = kwargs["delivery_address"]
        if delivery_cost:
            reqs[REQ_ORDER_DELIVERY_COST] = str(delivery_cost)
        reqs[REQ_ORDER_ITEMS_TOTAL] = str(items_total)
        reqs[REQ_ORDER_TOTAL] = str(total)
        if kwargs.get("messenger"):
            reqs[REQ_ORDER_MESSENGER] = kwargs["messenger"]

        obj_id = await self._api.create_object(TABLE_ORDERS, number, reqs)
        logger.info("Создан заказ '%s' (id=%d) в Integram. Итого: %.0f ₽", number, obj_id, total)

        # Создать позиции заказа
        for item_data in items:
            qty = item_data.get("quantity", 1)
            price = item_data.get("unit_price", 0)
            item_reqs = {
                REQ_ITEM_ORDER: str(obj_id),
                REQ_ITEM_PRODUCT: str(item_data["product_id"]),
                REQ_ITEM_QTY: str(qty),
                REQ_ITEM_PRICE: str(price),
                REQ_ITEM_SUM: str(qty * price),
            }
            item_name = f"Позиция заказа {number}"
            await self._api.create_object(TABLE_ORDER_ITEMS, item_name, item_reqs)
            logger.info(
                "Создана позиция заказа: product=%s, qty=%s, sum=%s",
                item_data["product_id"], qty, qty * price,
            )

        return Order(
            id=obj_id,
            number=number,
            client_id=client_id,
            date=datetime.now(),
            status=status,
            delivery_method=delivery_method,
            delivery_address=kwargs.get("delivery_address"),
            delivery_cost=delivery_cost,
            items_total=items_total,
            total=total,
            source=source,
            items=[],
        )

    async def update_order_status(
        self, order_id: int, status: str, *, from_status: str | None = None, comment: str = ""
    ) -> None:
        """Обновить статус заказа в Integram CRM и записать в Историю статусов."""
        if status not in STATUS_IDS:
            logger.warning("Неизвестный статус '%s'. Допустимые: %s", status, list(STATUS_IDS.keys()))
            return

        # Получить текущий статус (если не передан) для записи в историю
        if from_status is None:
            try:
                current_order = await self.get_order(order_id)
                from_status = current_order.status
            except Exception:
                from_status = ""

        # Обновить статус через set_reference_field (_m_del + _m_set)
        await self._api.set_reference_field(order_id, REQ_ORDER_STATUS, STATUS_IDS[status])

        # Дата отправки / доставки — через _m_save (не reference-поля, работает корректно)
        date_reqs: dict[str, str] = {}
        if status == "Отправлен":
            date_reqs[REQ_ORDER_SHIPPED_DATE] = datetime.now().strftime("%d.%m.%Y 00:00:00")
        if status == "Доставлен":
            date_reqs[REQ_ORDER_DELIVERED_DATE] = datetime.now().strftime("%d.%m.%Y 00:00:00")
        if date_reqs:
            await self._api.set_requisites(order_id, TABLE_ORDERS, date_reqs)

        logger.info("Статус заказа %d: '%s' → '%s'", order_id, from_status, status)

        # Записать в Историю статусов (best-effort: не прерывать при ошибке)
        try:
            await self._log_status_history(order_id, from_status, status, comment)
        except Exception as exc:
            logger.warning("Не удалось записать историю статуса заказа %d: %s", order_id, exc)

    async def _log_status_history(
        self, order_id: int, from_status: str, to_status: str, comment: str = ""
    ) -> None:
        """Создать запись в таблице «История статусов» в Integram."""
        name = f"{from_status or '—'} → {to_status}"
        reqs: dict[str, str] = {
            REQ_HISTORY_ORDER: str(order_id),
            REQ_HISTORY_DATE: datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
        }
        if from_status and from_status in STATUS_IDS:
            reqs[REQ_HISTORY_STATUS_FROM] = STATUS_IDS[from_status]
        if to_status and to_status in STATUS_IDS:
            reqs[REQ_HISTORY_STATUS_TO] = STATUS_IDS[to_status]
        if comment:
            reqs[REQ_HISTORY_COMMENT] = comment

        await self._api.create_object(TABLE_STATUS_HISTORY, name, reqs)

    # Статусы, в которых разрешено редактирование заказа
    EDITABLE_STATUSES = {"Новый", "Подтверждён", "В сборке"}

    async def update_order(self, order_id: int, **kwargs: Any) -> None:
        """Обновить поля заказа (адрес, доставка, комментарий и т.д.).

        Допустимые kwargs:
            delivery_address, delivery_method, delivery_cost,
            comment, items_total, total.
        """
        field_map = {
            "delivery_address": REQ_ORDER_ADDRESS,
            "delivery_cost": REQ_ORDER_DELIVERY_COST,
            "items_total": REQ_ORDER_ITEMS_TOTAL,
            "total": REQ_ORDER_TOTAL,
            "comment": REQ_ORDER_COMMENT,
            "tracking_number": REQ_ORDER_TRACKING,
        }
        reqs: dict[str, str] = {}
        for py_key, req_id in field_map.items():
            if py_key in kwargs:
                reqs[req_id] = str(kwargs[py_key])

        if reqs:
            await self._api.set_requisites(order_id, TABLE_ORDERS, reqs)

        # Справочные поля (delivery_method, status, source) — через set_reference_field
        if "delivery_method" in kwargs:
            dm = kwargs["delivery_method"]
            if dm in DELIVERY_IDS:
                await self._api.set_reference_field(order_id, REQ_ORDER_DELIVERY_METHOD, DELIVERY_IDS[dm])
        if "status" in kwargs:
            st = kwargs["status"]
            if st in STATUS_IDS:
                await self._api.set_reference_field(order_id, REQ_ORDER_STATUS, STATUS_IDS[st])
        if "source" in kwargs:
            src = kwargs["source"]
            if src in SOURCE_IDS:
                await self._api.set_reference_field(order_id, REQ_ORDER_SOURCE, SOURCE_IDS[src])

        logger.info("Заказ %d обновлён: %s", order_id, list(kwargs.keys()))

    async def update_order_checklist(
        self, order_id: int,
        cdek_confirmed: bool | None = None,
        client_notified: bool | None = None,
        stock_checked: bool | None = None,
    ) -> None:
        """Обновить чеклист-поля заказа (bool)."""
        from src.crm_constants import (
            REQ_ORDER_CDEK_CONFIRMED, REQ_ORDER_CLIENT_NOTIFIED, REQ_ORDER_STOCK_CHECKED,
        )
        reqs: dict[str, str] = {}
        if cdek_confirmed is not None:
            reqs[REQ_ORDER_CDEK_CONFIRMED] = "1" if cdek_confirmed else "0"
        if client_notified is not None:
            reqs[REQ_ORDER_CLIENT_NOTIFIED] = "1" if client_notified else "0"
        if stock_checked is not None:
            reqs[REQ_ORDER_STOCK_CHECKED] = "1" if stock_checked else "0"
        if reqs:
            await self._api.set_requisites(order_id, TABLE_ORDERS, reqs)

    async def get_order_items(self, order_id: int) -> list[OrderItem]:
        """Получить позиции заказа."""
        raw = await self._api.get_order_items(order_id)
        return [
            OrderItem(
                id=item["id"],
                order_id=item["order_id"] or order_id,
                product_id=item["product_id"] or 0,
                product_name=item.get("product_name"),
                quantity=item["quantity"],
                unit_price=item["unit_price"],
                total=item["total"],
            )
            for item in raw
        ]

    async def get_order_items_bulk(self) -> list[OrderItem]:
        """Получить ВСЕ позиции заказов (для аналитики)."""
        raw = await self._api.get_order_items(order_id=None)
        return [
            OrderItem(
                id=item["id"],
                order_id=item["order_id"] or 0,
                product_id=item["product_id"] or 0,
                product_name=item.get("product_name"),
                quantity=item["quantity"],
                unit_price=item["unit_price"],
                total=item["total"],
            )
            for item in raw
        ]

    async def add_order_item(
        self, order_id: int, product_id: int, qty: int, price: float,
    ) -> int:
        """Добавить позицию к заказу. Возвращает ID новой позиции."""
        # _m_new не сохраняет reference-поля и числовые реквизиты.
        # Правильный порядок: создать объект → set_requisites → set_reference_field ×2
        item_id = await self._api.create_object(TABLE_ORDER_ITEMS, "Позиция заказа")
        await self._api.set_requisites(item_id, TABLE_ORDER_ITEMS, {
            REQ_ITEM_QTY:   str(qty),
            REQ_ITEM_PRICE: str(int(price)),
            REQ_ITEM_SUM:   str(int(qty * price)),
        })
        await self._api.set_reference_field(item_id, REQ_ITEM_ORDER, str(order_id))
        if product_id:
            await self._api.set_reference_field(item_id, REQ_ITEM_PRODUCT, str(product_id))
        logger.info(
            "Добавлена позиция: order=%d, product=%d, qty=%d, sum=%.0f",
            order_id, product_id, qty, qty * price,
        )
        return item_id

    async def update_order_item(
        self, item_id: int, qty: Optional[int] = None, price: Optional[float] = None,
    ) -> None:
        """Обновить количество/цену позиции заказа."""
        reqs: dict[str, str] = {}
        if qty is not None:
            reqs[REQ_ITEM_QTY] = str(qty)
        if price is not None:
            reqs[REQ_ITEM_PRICE] = str(price)
        if qty is not None and price is not None:
            reqs[REQ_ITEM_SUM] = str(qty * price)
        elif qty is not None:
            # Нужна текущая цена для пересчёта суммы
            pass  # будет пересчитано при recalculate
        if reqs:
            await self._api.set_requisites(item_id, TABLE_ORDER_ITEMS, reqs)
            logger.info("Позиция %d обновлена: %s", item_id, reqs)

    async def delete_order_item(self, item_id: int) -> None:
        """Удалить позицию заказа."""
        await self._api.delete_object(item_id)
        logger.info("Позиция %d удалена", item_id)

    async def recalculate_order_totals(self, order_id: int) -> dict:
        """Пересчитать суммы заказа на основе позиций.

        Returns:
            {items_total, delivery_cost, total}
        """
        items = await self.get_order_items(order_id)
        items_total = sum(i.quantity * i.unit_price for i in items)

        # Получить текущую стоимость доставки
        order = await self.get_order(order_id)
        delivery_cost = order.delivery_cost or 0
        total = items_total + delivery_cost

        await self.update_order(
            order_id,
            items_total=items_total,
            total=total,
        )
        return {
            "items_total": items_total,
            "delivery_cost": delivery_cost,
            "total": total,
        }

    # ------------------------------------------------------------------
    # Вспомогательные методы
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Товары: CRUD
    # ------------------------------------------------------------------

    async def create_product(self, name: str, **kwargs: Any) -> int:
        """Создать товар. Возвращает ID."""
        reqs: dict[str, str] = {}
        if kwargs.get("price") is not None:
            reqs[REQ_PRODUCT_PRICE] = str(kwargs["price"])
        if kwargs.get("weight") is not None:
            reqs[REQ_PRODUCT_WEIGHT] = str(kwargs["weight"])
        if kwargs.get("description"):
            reqs[REQ_PRODUCT_DESC] = kwargs["description"]
        if kwargs.get("in_stock"):
            reqs[REQ_PRODUCT_INSTOCK] = "1"
        if kwargs.get("sku_uds"):
            reqs[REQ_PRODUCT_SKU] = kwargs["sku_uds"]
        if kwargs.get("category") and kwargs["category"] in CATEGORY_IDS:
            reqs[REQ_PRODUCT_CATEGORY] = CATEGORY_IDS[kwargs["category"]]
        if kwargs.get("short_name"):
            reqs[REQ_PRODUCT_SHORT] = kwargs["short_name"]
        if kwargs.get("stock") is not None:
            reqs[REQ_PRODUCT_STOCK] = str(kwargs["stock"])
        return await self._api.create_object(TABLE_PRODUCTS, name, reqs)

    async def update_product(self, product_id: int, **kwargs: Any) -> None:
        """Обновить поля товара."""
        reqs: dict[str, str] = {}
        if kwargs.get("price") is not None:
            reqs[REQ_PRODUCT_PRICE] = str(kwargs["price"])
        if kwargs.get("weight") is not None:
            reqs[REQ_PRODUCT_WEIGHT] = str(kwargs["weight"])
        if kwargs.get("description") is not None:
            reqs[REQ_PRODUCT_DESC] = kwargs["description"]
        if kwargs.get("in_stock") is not None:
            reqs[REQ_PRODUCT_INSTOCK] = "1" if kwargs["in_stock"] else ""
        if kwargs.get("sku_uds") is not None:
            reqs[REQ_PRODUCT_SKU] = kwargs["sku_uds"]
        if kwargs.get("category") is not None:
            cat_id = CATEGORY_IDS.get(kwargs["category"])
            if cat_id:
                reqs[REQ_PRODUCT_CATEGORY] = cat_id
        if kwargs.get("short_name") is not None:
            reqs[REQ_PRODUCT_SHORT] = kwargs["short_name"]
        if kwargs.get("stock") is not None:
            reqs[REQ_PRODUCT_STOCK] = str(kwargs["stock"])
        if kwargs.get("name") is not None:
            await self._api.update_object_value(product_id, kwargs["name"])
        if reqs:
            await self._api.set_requisites(product_id, TABLE_PRODUCTS, reqs)

    async def delete_product(self, product_id: int) -> None:
        """Снять товар с продажи (soft delete — in_stock = false)."""
        await self._api.set_requisites(product_id, TABLE_PRODUCTS, {
            REQ_PRODUCT_INSTOCK: "",
        })

    async def update_product_stock(self, product_id: int, stock: int) -> None:
        """Обновить остаток товара."""
        await self._api.set_requisites(product_id, TABLE_PRODUCTS, {
            REQ_PRODUCT_STOCK: str(stock),
        })

    # ------------------------------------------------------------------
    # Дашборд и аналитика
    # ------------------------------------------------------------------

    async def get_dashboard_stats(self) -> dict[str, Any]:
        """Статистика для дашборда."""
        return await self._api.get_dashboard_stats()

    async def get_order_history(self, order_id: int) -> list[dict]:
        """Получить историю статусов заказа из таблицы «История статусов»."""
        from src.crm_constants import (
            TABLE_STATUS_HISTORY, REQ_HISTORY_ORDER,
            REQ_HISTORY_STATUS_FROM, REQ_HISTORY_STATUS_TO,
            REQ_HISTORY_DATE, REQ_HISTORY_COMMENT,
        )
        from src.integram_api import _strip_html, _extract_ref_text

        # Обратный маппинг ID → название статуса
        status_by_id = {v: k for k, v in STATUS_IDS.items()}

        records = await self._api.get_all_objects(TABLE_STATUS_HISTORY)
        result = []
        for rec in records:
            reqs = rec.get("reqs", {})
            ref_order_raw = reqs.get(REQ_HISTORY_ORDER, "")
            # Integram хранит REF как HTML-ссылку; ID заказа — в тексте или как число
            ref_order_id_str = _extract_ref_text(ref_order_raw) or _strip_html(ref_order_raw)
            try:
                ref_order_id = int(ref_order_id_str)
            except (ValueError, TypeError):
                ref_order_id = 0

            if ref_order_id != order_id:
                continue

            from_id = _strip_html(reqs.get(REQ_HISTORY_STATUS_FROM, ""))
            to_id = _strip_html(reqs.get(REQ_HISTORY_STATUS_TO, ""))
            result.append({
                "id": rec.get("id"),
                "date": _strip_html(reqs.get(REQ_HISTORY_DATE, "")),
                "from_status": status_by_id.get(from_id, from_id or "—"),
                "to_status": status_by_id.get(to_id, to_id or "—"),
                "comment": _strip_html(reqs.get(REQ_HISTORY_COMMENT, "")),
            })

        result.sort(key=lambda x: x.get("date", ""))
        return result

    # ------------------------------------------------------------------
    # Партии отправки
    # ------------------------------------------------------------------

    async def get_batches(self) -> list[dict]:
        """Получить список партий отправки."""
        return await self._api.get_batches()

    async def create_batch(
        self,
        date: datetime,
        delivery_method: str = "",
        note: str = "",
    ) -> int:
        """Создать новую партию отправки. Возвращает ID партии."""
        from src.crm_constants import (
            TABLE_BATCHES, REQ_BATCH_DATE, REQ_BATCH_DELIVERY, REQ_BATCH_NOTE,
        )
        name = f"Партия {date.strftime('%d.%m.%Y')}"
        if delivery_method:
            name += f" — {delivery_method}"
        reqs: dict[str, str] = {
            REQ_BATCH_DATE: date.strftime("%d.%m.%Y 00:00:00"),
        }
        if delivery_method:
            reqs[REQ_BATCH_DELIVERY] = delivery_method
        if note:
            reqs[REQ_BATCH_NOTE] = note
        batch_id = await self._api.create_object(TABLE_BATCHES, name, reqs)
        logger.info("Создана партия отправки '%s' (id=%d)", name, batch_id)
        return batch_id

    async def set_order_batch(self, order_id: int, batch_id: Optional[int]) -> None:
        """Назначить или снять партию для заказа."""
        from src.crm_constants import REQ_ORDER_BATCH
        await self._api.set_requisites(
            order_id, TABLE_ORDERS,
            {REQ_ORDER_BATCH: str(batch_id) if batch_id else ""},
        )

    async def update_batch_count(self, batch_id: int) -> None:
        """Пересчитать и сохранить кол-во заказов в партии."""
        from src.crm_constants import TABLE_BATCHES, REQ_BATCH_COUNT
        orders = await self._api.get_orders()
        count = sum(1 for o in orders if o.get("batch_id") == batch_id)
        await self._api.set_requisites(batch_id, TABLE_BATCHES, {REQ_BATCH_COUNT: str(count)})

    async def add_health_profile(
        self, telegram_id: int, fact: str, symptom_name: str = ""
    ) -> bool:
        """Создать запись в «Профиль здоровья» для клиента по telegram_id.

        Returns:
            True если запись создана, False если клиент не найден в CRM.
        """
        from src.crm_constants import (
            TABLE_HEALTH_PROFILE, REQ_HEALTH_CLIENT, REQ_HEALTH_SOURCE, REQ_HEALTH_DATE,
        )
        # Найти client_id по telegram_id
        clients = await self.get_clients()
        client = next(
            (c for c in clients if c.telegram_id and int(c.telegram_id) == telegram_id),
            None,
        )
        if not client:
            return False

        reqs: dict[str, str] = {
            REQ_HEALTH_CLIENT: str(client.id),
            REQ_HEALTH_SOURCE: fact[:500],
            REQ_HEALTH_DATE: datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
        }
        name = f"{client.name or '?'}: {fact[:60]}"
        await self._api.create_object(TABLE_HEALTH_PROFILE, name, reqs)
        logger.info("Профиль здоровья: добавлен факт для клиента %d (tg=%d)", client.id, telegram_id)
        return True

    async def get_client_telegram_id(self, client_id: int) -> Optional[int]:
        """Получить Telegram ID клиента по ID в CRM."""
        clients = await self.get_clients()
        for c in clients:
            if c.id == client_id and c.telegram_id:
                return c.telegram_id
        return None

    # ------------------------------------------------------------------
    # Вспомогательные методы
    # ------------------------------------------------------------------

    @staticmethod
    def _dict_to_order(item: dict) -> Order:
        """Конвертировать dict из IntegramAPI.get_orders() в Order."""
        date_str = item.get("date", "")
        try:
            if "/" in date_str:
                parts = date_str.split()
                month, day, year = parts[0].split("/")
                hour = minute = 0
                if len(parts) > 1 and ":" in parts[1]:
                    tp = parts[1].split(":")
                    hour, minute = int(tp[0]), int(tp[1]) if len(tp) > 1 else 0
                date = datetime(int(year), int(month), int(day), hour, minute)
            elif "." in date_str:
                parts = date_str.split()
                day, month, year = parts[0].split(".")
                hour = minute = second = 0
                if len(parts) > 1 and ":" in parts[1]:
                    tp = parts[1].split(":")
                    hour = int(tp[0])
                    minute = int(tp[1]) if len(tp) > 1 else 0
                    second = int(tp[2]) if len(tp) > 2 else 0
                date = datetime(int(year), int(month), int(day), hour, minute, second)
            else:
                date = datetime.fromisoformat(date_str) if date_str else datetime.now()
        except (ValueError, IndexError):
            date = datetime.now()

        return Order(
            id=item.get("id", 0),
            number=item.get("number", ""),
            client_id=item.get("client_id") or 0,
            client_name=item.get("client_name"),
            date=date,
            status=item.get("status", ""),
            delivery_method=item.get("delivery_method"),
            delivery_address=item.get("delivery_address"),
            delivery_cost=item.get("delivery_cost"),
            items_total=item.get("items_total"),
            total=item.get("total"),
            tracking_number=item.get("tracking_number"),
            source=item.get("source"),
            comment=item.get("comment"),
            messenger=item.get("messenger"),
            month=item.get("month"),
            batch_id=item.get("batch_id"),
            items=[],
        )
