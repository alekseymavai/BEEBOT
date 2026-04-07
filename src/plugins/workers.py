"""WorkersPlugin — WorkerService + WorkerAgent (режим работника склада).

Зависимости: crm

Публикует в контейнере:
  "worker_service" → WorkerService

Регистрирует:
  worker_router → aiogram Dispatcher
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from src.kernel.plugin import Plugin

if TYPE_CHECKING:
    from aiogram import Dispatcher
    from src.kernel.container import Container

logger = logging.getLogger(__name__)


class WorkersPlugin(Plugin):
    name = "workers"
    dependencies = ["crm"]

    def __init__(self, bot: Any = None) -> None:
        self._bot = bot
        self._worker_router = None

    async def setup(self, container: "Container") -> None:
        from src.config import WORKER_CHAT_IDS
        from src.services.worker_service import WorkerService

        crm = container.get("crm")
        worker_service = WorkerService()
        container.set("worker_service", worker_service)

        if crm and WORKER_CHAT_IDS:
            logger.info(
                "WorkersPlugin: режим работника включён (%d работников).",
                len(WORKER_CHAT_IDS),
            )
        else:
            logger.info("WorkersPlugin: работники не настроены.")

    def register_routers(self, dp: "Dispatcher") -> None:
        from src.config import WORKER_CHAT_IDS
        if not WORKER_CHAT_IDS or not self._bot:
            return

        # Импортируем контейнер через замыкание — к моменту регистрации
        # setup() уже вызван, crm и gift_broker в контейнере есть.
        from src.routers.worker import router as worker_router, setup_worker

        # gift_broker может быть ещё не зарегистрирован (зависит от порядка).
        # setup_worker принимает gift_broker=None — это нормально.
        self._worker_router = worker_router
        dp.include_router(worker_router)
        logger.debug("WorkersPlugin: worker_router подключён.")

    def configure_worker_router(self, container: "Container") -> None:
        """Вызывается после setup всех плагинов для финальной настройки."""
        from src.config import WORKER_CHAT_IDS
        if not WORKER_CHAT_IDS or not self._bot:
            return
        from src.routers.worker import setup_worker

        crm = container.get("crm")
        gift_broker = container.get("gift_broker")
        auth = container.get("auth")

        if crm:
            setup_worker(crm, self._bot, gift_broker=gift_broker, auth=auth)
            logger.info("WorkersPlugin: setup_worker выполнен.")
