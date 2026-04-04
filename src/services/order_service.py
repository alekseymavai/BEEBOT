"""OrderService — единый источник правды для операций с заказами.

Объединяет логику из:
- LogistAgent.create_order (Telegram-бот)
- web/routers/orders.py (веб-панель)
- integrations/uds.py (UDS-синхронизация)
- admin.py (TG-команды)
- delivery/tracker.py (авто-трекинг)

Не знает про Telegram, FastAPI, Redis. Чистая бизнес-логика.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Optional

from src.crm_constants import STATUS_IDS
from src.models import Order, OrderItem

from src.services.event_emitter import events

logger = logging.getLogger(__name__)

# Допустимые переходы статусов
_STATUS_FLOW = {
    "Новый": {"Подтверждён", "Отменён"},
    "Подтверждён": {"В сборке", "Отменён"},
    "В сборке": {"Отправлен", "Отменён"},
    "Отправлен": {"Доставлен"},
    "Доставлен": set(),
    "Отменён": set(),
}

EDITABLE_STATUSES = {"Новый", "Подтверждён", "В сборке"}

_WAREHOUSE_TRANSITIONS = {
    ("Подтверждён", "В сборке"),
    ("В сборке", "Отправлен"),
}


class OrderService:
    """Управление заказами — создание, статусы, позиции."""

    def __init__(self, crm, notifier=None):
        """
        Args:
            crm: IntegramClient (или любой объект с тем же интерфейсом).
            notifier: NotificationService (опционально).
        """
        self._crm = crm
        self._notifier = notifier

    # ------------------------------------------------------------------
    # Создание заказа
    # ------------------------------------------------------------------

    async def create_order(
        self,
        client_id: int,
        items: list[dict],
        *,
        delivery_method: Optional[str] = None,
        delivery_address: Optional[str] = None,
        delivery_cost: float = 0,
        source: str = "Telegram",
        number: Optional[str] = None,
        status: str = "Новый",
        date: Optional[datetime] = None,
        comment: Optional[str] = None,
    ) -> Order:
        """Создать заказ в CRM.

        Args:
            client_id: ID клиента в CRM.
            items: [{product_id, quantity, unit_price}, ...].
            delivery_method: СДЭК / Почта России / Самовывоз.
            delivery_address: Адрес доставки.
            delivery_cost: Стоимость доставки.
            source: Telegram / UDS / ВК / Instagram / ...
            number: Номер заказа (auto если не указан).
            status: Начальный статус.
            date: Дата заказа (now если не указана).
            comment: Комментарий.

        Returns:
            Order — созданный заказ.
        """
        items_total = sum(
            i.get("quantity", 1) * i.get("unit_price", 0) for i in items
        )
        total = items_total + delivery_cost

        order = await self._crm.create_order(
            client_id=client_id,
            items=items,
            delivery_method=delivery_method,
            delivery_address=delivery_address,
            delivery_cost=delivery_cost,
            items_total=items_total,
            total=total,
            source=source,
            number=number,
            status=status,
            date=date,
            comment=comment,
        )

        logger.info(
            "Заказ создан: %s | клиент=%d | %.0f₽ | %d позиций | %s",
            order.number, client_id, total, len(items), source,
        )

        if self._notifier:
            await self._notifier.on_order_created(order, items)

        # Событие для SSE, Redis, подписчиков
        await events.emit("order.created", {
            "order_id": order.id, "number": order.number,
            "total": total, "source": source,
        })

        return order

    # ------------------------------------------------------------------
    # Клиент + Заказ (удобный метод для бота и UDS)
    # ------------------------------------------------------------------

    async def create_order_with_client(
        self,
        telegram_id: int,
        full_name: str,
        phone: str,
        items: list[dict],
        *,
        address: Optional[str] = None,
        telegram_username: Optional[str] = None,
        delivery_method: Optional[str] = None,
        delivery_cost: float = 0,
        source: str = "Telegram",
        **kwargs: Any,
    ) -> Order:
        """Найти/создать клиента и создать заказ."""
        client = await self._crm.get_or_create_client(
            telegram_id=telegram_id,
            full_name=full_name,
            phone=phone,
            address=address,
            telegram_username=telegram_username,
            source=source,
        )

        # Обновить данные клиента (могли измениться)
        await self._crm.update_client(
            client.id,
            full_name=full_name,
            phone=phone,
            address=address,
        )

        return await self.create_order(
            client_id=client.id,
            items=items,
            delivery_method=delivery_method,
            delivery_address=address,
            delivery_cost=delivery_cost,
            source=source,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Статус
    # ------------------------------------------------------------------

    async def update_status(
        self,
        order_id: int,
        new_status: str,
        *,
        comment: str = "",
        role: str = "admin",
    ) -> Order:
        """Обновить статус заказа.

        Args:
            order_id: ID заказа.
            new_status: Новый статус (должен быть в STATUS_IDS).
            comment: Комментарий (напр. "авто-трекер СДЭК").
            role: "admin" | "warehouse" — для проверки прав.

        Returns:
            Order — обновлённый заказ.

        Raises:
            ValueError: Некорректный статус или переход.
            PermissionError: Нет прав на этот переход.
        """
        if new_status not in STATUS_IDS:
            raise ValueError(f"Некорректный статус: {new_status}")

        order = await self._crm.get_order(order_id)
        old_status = order.status or ""

        # Проверка прав для работника склада
        if role == "warehouse":
            if (old_status, new_status) not in _WAREHOUSE_TRANSITIONS:
                raise PermissionError(
                    f"Работник склада не может: {old_status} → {new_status}"
                )

        await self._crm.update_order_status(
            order_id, new_status,
            from_status=old_status,
            comment=comment,
        )

        updated = await self._crm.get_order(order_id)

        logger.info(
            "Статус: %s | %s → %s%s",
            updated.number, old_status or "(пусто)", new_status,
            f" ({comment})" if comment else "",
        )

        if self._notifier:
            await self._notifier.on_status_changed(updated, old_status, new_status)

        # Событие для SSE, Redis, подписчиков
        await events.emit("order.status_changed", {
            "order_id": updated.id, "number": updated.number,
            "old_status": old_status, "new_status": new_status,
        })

        return updated

    # ------------------------------------------------------------------
    # Чтение
    # ------------------------------------------------------------------

    async def get_orders(
        self,
        *,
        client_id: Optional[int] = None,
        status: Optional[str] = None,
    ) -> list[Order]:
        """Получить список заказов."""
        return await self._crm.get_orders(client_id=client_id, status=status)

    async def get_order(self, order_id: int) -> Order:
        """Получить заказ по ID."""
        return await self._crm.get_order(order_id)

    async def get_order_items(self, order_id: int) -> list[OrderItem]:
        """Получить позиции заказа."""
        return await self._crm.get_order_items(order_id)

    async def get_order_items_bulk(self) -> list[OrderItem]:
        """Получить все позиции всех заказов."""
        return await self._crm.get_order_items_bulk()

    # ------------------------------------------------------------------
    # Позиции
    # ------------------------------------------------------------------

    async def add_item(
        self, order_id: int, product_id: int, quantity: int, unit_price: float,
    ) -> int:
        """Добавить позицию в заказ. Возвращает item_id."""
        order = await self._crm.get_order(order_id)
        if order.status and order.status not in EDITABLE_STATUSES:
            raise ValueError(f"Заказ в статусе '{order.status}' нельзя редактировать")

        item_id = await self._crm.add_order_item(order_id, product_id, quantity, unit_price)
        await self._crm.recalculate_order_totals(order_id)
        return item_id

    async def update_item(
        self, order_id: int, item_id: int, **kwargs: Any,
    ) -> None:
        """Обновить позицию заказа (quantity, unit_price)."""
        order = await self._crm.get_order(order_id)
        if order.status and order.status not in EDITABLE_STATUSES:
            raise ValueError(f"Заказ в статусе '{order.status}' нельзя редактировать")

        await self._crm.update_order_item(item_id, **kwargs)
        await self._crm.recalculate_order_totals(order_id)

    async def delete_item(self, order_id: int, item_id: int) -> None:
        """Удалить позицию заказа."""
        order = await self._crm.get_order(order_id)
        if order.status and order.status not in EDITABLE_STATUSES:
            raise ValueError(f"Заказ в статусе '{order.status}' нельзя редактировать")

        await self._crm.delete_order_item(item_id)
        await self._crm.recalculate_order_totals(order_id)

    # ------------------------------------------------------------------
    # Обновление полей
    # ------------------------------------------------------------------

    async def update_order(self, order_id: int, **kwargs: Any) -> Order:
        """Обновить поля заказа (адрес, доставка, комментарий, трекинг)."""
        tracking = kwargs.get("tracking_number")
        if tracking and self._notifier:
            await self._crm.update_order(order_id, **kwargs)
            updated = await self._crm.get_order(order_id)
            await self._notifier.on_tracking_added(updated, tracking)
            return updated

        await self._crm.update_order(order_id, **kwargs)
        return await self._crm.get_order(order_id)
