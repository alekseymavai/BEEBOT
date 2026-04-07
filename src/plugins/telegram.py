"""TelegramPlugin — все aiogram Router-ы Telegram-бота.

Зависимости: crm, agents, orders, gift, workers

Регистрирует роутеры в Dispatcher (порядок критичен):
  1. admin_router         — /orders, /status, /track (приоритет)
  2. bot_admin_router     — /stats, /faq, /dev, /admin-mode
  3. inspect_router       — InspectFSM
  4. fsm_order_router     — OrderFSM
  5. worker_router        — очередь сборки (если WORKER_CHAT_IDS)
  6. fsm_edit_router      — редактирование заказа
  7. user_router          — ПОСЛЕДНИМ (StateFilter(None))

Не публикует ничего в контейнере.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from src.kernel.plugin import Plugin

if TYPE_CHECKING:
    from aiogram import Dispatcher
    from src.kernel.container import Container

logger = logging.getLogger(__name__)


class TelegramPlugin(Plugin):
    name = "telegram"
    dependencies = ["crm", "agents", "orders", "gift", "workers"]

    def __init__(self, bot: Any, alert_fn: Optional[Any] = None) -> None:
        self._bot = bot
        self._alert_fn = alert_fn
        self._container: Optional["Container"] = None

    async def setup(self, container: "Container") -> None:
        """Сохраняем контейнер для использования при register_routers."""
        self._container = container

        # Notifier для работников склада
        from src.config import WORKER_CHAT_IDS
        if WORKER_CHAT_IDS:
            from src.notifications import Notifier
            import src.notifications as _notif_module
            _notif_module._worker_notifier = Notifier(self._bot)

    def register_routers(self, dp: "Dispatcher") -> None:
        """Подключить все роутеры в правильном порядке."""
        assert self._container is not None, "setup() не вызван"
        c = self._container

        auth = c.get("auth")
        orchestrator = c.require("orchestrator")
        inspector = c.require("inspector")
        logist = c.require("logist")
        analyst = c.require("analyst")
        admin_chat_agent = c.require("admin_chat_agent")
        gift_broker = c.get("gift_broker")
        crm = c.get("crm")
        kb = c.require("kb")
        memory_svc = getattr(orchestrator, "_memory_svc", None)

        # 1. admin_router (приоритет)
        from src.admin import router as admin_router, setup_admin
        setup_admin(self._bot, crm=crm, kb=kb, memory=memory_svc, auth=auth)
        dp.include_router(admin_router)

        # 2. bot_admin_router
        from src.routers.bot_admin import router as bot_admin_router, setup_bot_admin
        setup_bot_admin(analyst, orchestrator, admin_chat_agent, inspector, self._bot, auth=auth)
        dp.include_router(bot_admin_router)

        # 3. inspect_router
        from src.routers.inspect import router as inspect_router, setup_inspect
        setup_inspect(inspector)
        dp.include_router(inspect_router)

        # 4. fsm_order_router
        from src.routers.fsm_order import router as fsm_order_router, setup_fsm_order
        setup_fsm_order(logist, self._bot)
        dp.include_router(fsm_order_router)

        # 5. worker_router (если работники настроены)
        from src.config import WORKER_CHAT_IDS
        if crm and WORKER_CHAT_IDS:
            from src.routers.worker import router as worker_router, setup_worker
            setup_worker(crm, self._bot, gift_broker=gift_broker, auth=auth)
            dp.include_router(worker_router)

        # 6. fsm_edit_router
        from src.routers.fsm_edit import router as fsm_edit_router, setup_fsm_edit
        setup_fsm_edit(logist, self._bot)
        dp.include_router(fsm_edit_router)

        # 7. user_router — ПОСЛЕДНИМ
        from src.routers.user import router as user_router, setup_user
        setup_user(orchestrator, admin_chat_agent, logist, gift_broker=gift_broker, auth=auth)
        dp.include_router(user_router)

        logger.info("TelegramPlugin: все роутеры подключены.")
