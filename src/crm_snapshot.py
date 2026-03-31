"""Кэшированный снимок CRM — заказы с позициями, клиенты, товары.

Обновляется:
  - автоматически каждые N секунд (фоновая задача)
  - принудительно по команде администратора (/refresh_crm или кнопка)

Экономит API-запросы: 2 вызова (get_orders + get_order_items_bulk)
вместо N+1 отдельных запросов за позициями каждого заказа.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from typing import Callable, Awaitable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.integram_client import IntegramClient
    from src.models import Order, Client, Product

logger = logging.getLogger(__name__)

DEFAULT_REFRESH_INTERVAL = 300  # 5 минут
LOW_STOCK_THRESHOLD = 5          # алерт при остатке < N штук
_LOW_STOCK_DEBOUNCE = 86400      # не повторять алерт по одному товару чаще 1 раза в сутки


class CrmSnapshot:
    """Кэш данных CRM с позициями заказов.

    Использование:
        snapshot = CrmSnapshot(crm)
        asyncio.create_task(snapshot.run())   # фоновое обновление
        await snapshot.refresh()              # принудительное обновление
        order = snapshot.get_order(order_id)  # быстрый поиск
    """

    def __init__(
        self,
        crm: "IntegramClient",
        refresh_interval: int = DEFAULT_REFRESH_INTERVAL,
        alert_fn: "Optional[Callable[[str], Awaitable[None]]]" = None,
    ):
        self._crm = crm
        self._refresh_interval = refresh_interval
        self._alert_fn = alert_fn
        self._running = False
        self._low_stock_alerted: dict[int, float] = {}  # product_id → last alert monotonic

        self.orders: list["Order"] = []
        self.clients: list["Client"] = []
        self.products: list["Product"] = []
        self.updated_at: Optional[datetime] = None

    async def refresh(self) -> None:
        """Загрузить полный снимок: заказы + позиции + клиенты + товары."""
        logger.info("CRM snapshot: обновление...")

        # Заказы и позиции — 2 параллельных запроса
        results = await asyncio.gather(
            self._crm.get_orders(),
            self._crm.get_order_items_bulk(),
            return_exceptions=True,
        )
        orders, all_items = results

        if isinstance(orders, BaseException):
            logger.error("CRM snapshot: ошибка загрузки заказов: %s", orders)
            return

        # Группируем позиции по order_id → быстрый поиск O(1)
        items_by_order: dict[int, list] = {}
        if not isinstance(all_items, BaseException):
            for item in all_items:
                items_by_order.setdefault(item.order_id, []).append(item)
        else:
            logger.warning("CRM snapshot: позиции недоступны: %s", all_items)

        # Наполняем каждый заказ позициями
        for order in orders:
            order.items = items_by_order.get(order.id, [])

        self.orders = orders

        # Клиенты и товары параллельно (некритично — не останавливаем если упадут)
        try:
            clients, products = await asyncio.gather(
                self._crm.get_clients(),
                self._crm.get_products(),
                return_exceptions=True,
            )
            if not isinstance(clients, BaseException):
                self.clients = clients
            if not isinstance(products, BaseException):
                self.products = products
        except Exception as e:
            logger.warning("CRM snapshot: клиенты/товары недоступны: %s", e)

        self.updated_at = datetime.now()
        total_items = sum(len(o.items) for o in self.orders)
        logger.info(
            "CRM snapshot: готов — %d заказов, %d позиций, %d клиентов, %d товаров",
            len(self.orders), total_items, len(self.clients), len(self.products),
        )

        # Алерт при низком остатке (11.3)
        if self._alert_fn and self.products:
            await self._check_low_stock()

    async def _check_low_stock(self) -> None:
        """Отправить алерт пчеловоду если остаток товара < LOW_STOCK_THRESHOLD."""
        now = time.monotonic()
        for product in self.products:
            if product.stock is None:
                continue
            if product.stock >= LOW_STOCK_THRESHOLD:
                continue
            last = self._low_stock_alerted.get(product.id, 0)
            if now - last < _LOW_STOCK_DEBOUNCE:
                continue
            self._low_stock_alerted[product.id] = now
            try:
                await self._alert_fn(
                    f"⚠️ *Низкий остаток*: {product.name} — *{product.stock} шт.*"
                )
                logger.info(
                    "CrmSnapshot: алерт низкого остатка — %s (%d шт.)",
                    product.name, product.stock,
                )
            except Exception as _e:
                logger.warning("CrmSnapshot: ошибка алерта низкого остатка: %s", _e)

    def stop(self) -> None:
        """Остановить фоновую задачу."""
        self._running = False

    async def run(self) -> None:
        """Фоновая задача: обновлять снимок каждые N секунд."""
        self._running = True
        while self._running:
            try:
                await self.refresh()
            except Exception as e:
                logger.error("CRM snapshot: необработанная ошибка: %s", e)
            await asyncio.sleep(self._refresh_interval)

    # ------------------------------------------------------------------
    # Поиск
    # ------------------------------------------------------------------

    def get_order(self, order_id: int) -> Optional["Order"]:
        """Найти заказ по ID (с позициями)."""
        return next((o for o in self.orders if o.id == order_id), None)

    def get_order_by_number(self, number: str) -> Optional["Order"]:
        """Найти заказ по номеру."""
        return next((o for o in self.orders if (o.number or "") == number), None)

    def get_worker_queue(self) -> list["Order"]:
        """Заказы для очереди сборки (Новый/Подтверждён/В сборке)."""
        statuses = {"Новый", "Подтверждён", "В сборке"}
        return [o for o in self.orders if o.status in statuses]

    # ------------------------------------------------------------------
    # Метаинформация
    # ------------------------------------------------------------------

    @property
    def age_str(self) -> str:
        """Возраст снимка в читаемом виде."""
        if not self.updated_at:
            return "не загружен"
        delta = (datetime.now() - self.updated_at).total_seconds()
        if delta < 60:
            return f"{int(delta)} сек назад"
        if delta < 3600:
            return f"{int(delta / 60)} мин назад"
        return f"{int(delta / 3600)} ч назад"

    @property
    def is_ready(self) -> bool:
        return self.updated_at is not None
