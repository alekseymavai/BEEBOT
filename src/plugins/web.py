"""WebPlugin — FastAPI веб-панель.

Зависимости: crm, orders, analytics, delivery, workers

Публикует в контейнере:
  "fastapi_app" → FastAPI приложение

Регистрирует API-роутеры через register_api().
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Optional

from src.kernel.plugin import Plugin

if TYPE_CHECKING:
    from aiogram import Dispatcher
    from fastapi import FastAPI
    from src.kernel.container import Container

logger = logging.getLogger(__name__)


class WebPlugin(Plugin):
    name = "web"
    dependencies = ["crm", "orders", "analytics", "delivery", "workers"]

    def __init__(self, bot: Any = None) -> None:
        self._bot = bot
        self._container: Optional["Container"] = None

    async def setup(self, container: "Container") -> None:
        from src.web.api import create_app

        self._container = container

        # Создать FastAPI приложение
        fastapi_app = create_app()
        container.set("fastapi_app", fastapi_app)

        # Инжектировать сервисы в веб-приложение
        from src.web.api import inject_services

        # Собираем совместимый Services-объект из контейнера
        svc = _ContainerServicesAdapter(container)
        inject_services(svc)

        logger.info("WebPlugin: FastAPI приложение создано.")

    def register_api(self, app: "FastAPI") -> None:
        """Роутеры уже подключены при inject_services — ничего не делаем."""


class _ContainerServicesAdapter:
    """Адаптер контейнера под старый интерфейс Services для inject_services().

    Позволяет передавать контейнер туда, где ожидается Services-датакласс.
    Убираем после полного перехода web/api.py на Container.
    """

    def __init__(self, container: "Container") -> None:
        self._c = container

    def __getattr__(self, name: str) -> Any:
        # Маппинг старых полей Services → новые ключи контейнера
        _aliases = {
            "crm": "crm",
            "auth": "auth",
            "order_service": "order_service",
            "notification_service": "notification_service",
            "analytics_service": "analytics_service",
            "consult_service": "consult_service",
            "worker_service": "worker_service",
            "delivery_service": "delivery_service",
            "dashboard_service": "dashboard_service",
            "orchestrator": "orchestrator",
            "analyst": "analyst",
            "admin_chat_agent": "admin_chat_agent",
            "inspector": "inspector",
            "logist": "logist",
            "kb": "kb",
            "gift_broker": "gift_broker",
            "crm_agent": "crm_agent",
            "anamnesis_cache": "anamnesis",
            "agent_specs": "agent_specs",
            "bg_manager": "bg_manager",
            "state_store": "state_store",
        }
        key = _aliases.get(name, name)
        return self._c.get(key)
