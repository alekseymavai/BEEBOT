"""CrmPlugin — CRM-клиент + AuthService.

Публикует в контейнере:
  "crm"   → IntegramClient (v1 или v2 в зависимости от INTEGRAM_V2)
  "auth"  → AuthService

Все остальные плагины берут CRM через container.get("crm").
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.kernel.plugin import Plugin

if TYPE_CHECKING:
    from src.kernel.container import Container

logger = logging.getLogger(__name__)


class CrmPlugin(Plugin):
    name = "crm"
    dependencies: list[str] = []

    async def setup(self, container: "Container") -> None:
        from src.config import ADMIN_IDS, WORKER_CHAT_IDS, BEEKEEPER_CHAT_ID
        from src.crm_factory import get_crm_client
        from src.services.auth_service import AuthService

        # --- AuthService (не требует CRM) ---
        auth = AuthService(
            admin_ids=ADMIN_IDS,
            worker_ids=WORKER_CHAT_IDS,
            beekeeper_id=BEEKEEPER_CHAT_ID,
        )
        container.set("auth", auth)

        # --- CRM ---
        try:
            crm = get_crm_client()
            await crm.authenticate()
            container.set("crm", crm)
            logger.info("CRM подключена.")
        except Exception as e:
            logger.warning("CRM недоступна: %s — работаем без CRM.", e)
            container.set("crm", None)

    async def teardown(self) -> None:
        # crm хранится только в контейнере — не ссылаемся на него здесь
        pass
