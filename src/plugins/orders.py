"""OrdersPlugin — OrderService + NotificationService.

Зависимости: crm, agents

Публикует в контейнере:
  "order_service"        → OrderService
  "notification_service" → NotificationService
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Coroutine, Optional

from src.kernel.plugin import Plugin

if TYPE_CHECKING:
    from src.kernel.container import Container

logger = logging.getLogger(__name__)

# Тип callback-а для отправки Telegram-сообщений
SendTgFn = Optional[Callable[[int, str], Coroutine[Any, Any, bool]]]


class OrdersPlugin(Plugin):
    name = "orders"
    dependencies = ["crm", "agents"]

    def __init__(self, send_telegram: SendTgFn = None) -> None:
        """
        Args:
            send_telegram: async callback `(chat_id, text) → bool` для уведомлений.
                           Передаётся из bot.py где живёт aiogram Bot.
        """
        self._send_telegram = send_telegram

    async def setup(self, container: "Container") -> None:
        from src.config import BEEKEEPER_CHAT_ID, WORKER_CHAT_IDS
        from src.services.order_service import OrderService
        from src.services.notification_service import NotificationService

        crm = container.get("crm")
        logist = container.require("logist")

        notification_service: Optional[NotificationService] = None
        if crm and self._send_telegram:
            notification_service = NotificationService(
                send_telegram=self._send_telegram,
                beekeeper_chat_id=BEEKEEPER_CHAT_ID,
                worker_ids=WORKER_CHAT_IDS,
                get_client_tg_id=crm.get_client_telegram_id,
            )

        order_service: Optional[OrderService] = None
        if crm:
            order_service = OrderService(
                crm=crm,
                notifier=notification_service,
            )
            logist.set_order_service(order_service)

        container.set("order_service", order_service)
        container.set("notification_service", notification_service)

        logger.info(
            "OrdersPlugin: order_service=%s, notifications=%s.",
            bool(order_service),
            bool(notification_service),
        )
