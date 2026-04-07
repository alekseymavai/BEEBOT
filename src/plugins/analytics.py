"""AnalyticsPlugin — AnalyticsService.

Зависимости: crm, agents

Публикует в контейнере:
  "analytics_service" → AnalyticsService
  "dashboard_service" → DashboardService
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.kernel.plugin import Plugin

if TYPE_CHECKING:
    from src.kernel.container import Container

logger = logging.getLogger(__name__)


class AnalyticsPlugin(Plugin):
    name = "analytics"
    dependencies = ["crm", "agents"]

    async def setup(self, container: "Container") -> None:
        from src.services.analytics_service import AnalyticsService
        from src.services.dashboard_service import DashboardService

        crm = container.get("crm")
        orchestrator = container.require("orchestrator")

        analytics_service = None
        dashboard_service = None

        if crm:
            analytics_service = AnalyticsService(
                crm=crm,
                groq_client=orchestrator._groq,
                groq_model=orchestrator._model,
            )
            dashboard_service = DashboardService(crm=crm)

        container.set("analytics_service", analytics_service)
        container.set("dashboard_service", dashboard_service)

        logger.info("AnalyticsPlugin: analytics=%s.", bool(analytics_service))
