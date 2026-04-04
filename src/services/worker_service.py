"""WorkerService — бизнес-логика работника склада (состояние, чеклисты, очередь).

Извлечено из src/agents/worker.py (WorkerStateManager).
Пока in-memory, SQLite-персистенция — отдельная задача.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from src.integram_client import IntegramClient
    from src.models import Order

logger = logging.getLogger(__name__)

# Статусы, которые показывает очередь
WORKER_QUEUE_STATUSES = {"Новый", "Подтверждён", "В сборке"}


class WorkerService:
    """Управление состоянием работников, чеклистами и очередью заказов.

    Жизненный цикл:
      - Работник берёт заказ → set_busy(worker_id)
      - Новый push-Gift приходит → receive(worker_id, gift) → DEFERRED если занят
      - Работник завершает сборку → set_idle(worker_id, deliver_fn)
      - Все отложенные Gift доставляются через deliver_fn
    """

    def __init__(self) -> None:
        self._states: dict[int, Literal["idle", "busy"]] = {}
        self._inboxes: dict[int, asyncio.Queue] = {}
        self._checklists: dict[int, dict[int, set[int]]] = {}

    # ------------------------------------------------------------------
    # Состояние работника
    # ------------------------------------------------------------------

    def is_busy(self, worker_id: int) -> bool:
        return self._states.get(worker_id) == "busy"

    def set_busy(self, worker_id: int) -> None:
        """Пометить работника как занятого сборкой."""
        self._states[worker_id] = "busy"
        logger.debug("WorkerService: worker %d → busy", worker_id)

    async def set_idle(
        self,
        worker_id: int,
        deliver_fn: "Callable | None" = None,
    ) -> None:
        """Пометить работника как свободного; доставить отложенные Gift."""
        self._states[worker_id] = "idle"
        logger.debug("WorkerService: worker %d → idle", worker_id)
        inbox = self._inboxes.get(worker_id)
        if inbox and not inbox.empty() and deliver_fn:
            while not inbox.empty():
                gift = inbox.get_nowait()
                try:
                    await deliver_fn(worker_id, gift)
                except Exception as _e:
                    logger.warning("WorkerService: ошибка доставки отложенного Gift: %s", _e)

    def receive(self, worker_id: int, gift=None) -> Literal["ACCEPTED", "DEFERRED"]:
        """Принять Gift для работника. DEFERRED если работник занят."""
        if self.is_busy(worker_id):
            inbox = self._inboxes.setdefault(worker_id, asyncio.Queue())
            inbox.put_nowait(gift)
            logger.debug("WorkerService: Gift для worker %d → DEFERRED", worker_id)
            return "DEFERRED"
        logger.debug("WorkerService: Gift для worker %d → ACCEPTED", worker_id)
        return "ACCEPTED"

    # ------------------------------------------------------------------
    # Чеклист (per-worker, per-order)
    # ------------------------------------------------------------------

    def toggle_item(self, worker_id: int, order_id: int, item_id: int) -> None:
        """Отметить/снять позицию в чеклисте."""
        order_checked = self._checklists.setdefault(worker_id, {}).setdefault(order_id, set())
        if item_id in order_checked:
            order_checked.discard(item_id)
        else:
            order_checked.add(item_id)

    def clear_checklist(self, worker_id: int, order_id: int) -> None:
        """Очистить чеклист заказа (после завершения сборки)."""
        self._checklists.get(worker_id, {}).pop(order_id, None)

    def get_checked(self, worker_id: int, order_id: int) -> set[int]:
        """Вернуть множество отмеченных item_id."""
        return self._checklists.get(worker_id, {}).get(order_id, set())

    def is_fully_checked(self, worker_id: int, order_id: int, items: list) -> bool:
        """Все ли позиции заказа отмечены."""
        if not items:
            return False
        checked = self.get_checked(worker_id, order_id)
        return {i.id for i in items}.issubset(checked)

    # ------------------------------------------------------------------
    # Очередь заказов
    # ------------------------------------------------------------------

    async def get_queue(self, crm: "IntegramClient") -> list["Order"]:
        """Заказы со статусом Новый/Подтверждён/В сборке."""
        all_orders = await crm.get_orders()
        return [o for o in all_orders if o.status in WORKER_QUEUE_STATUSES]
