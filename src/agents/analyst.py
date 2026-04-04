"""Агент «Аналитик» — тонкая обёртка над AnalyticsService.

Бизнес-логика (все format_*, classify, filter) — в src/services/analytics_service.py.
Агент сохраняет обратную совместимость внешнего API.
"""

from __future__ import annotations

import logging

from src.services.analytics_service import AnalyticsService

# Реэкспорт для обратной совместимости (импортируются в тестах)
from src.services.analytics_service import (  # noqa: F401
    format_orders_report,
    format_top_products_report,
    format_packaging_report,
    format_clients_report,
    format_delivery_report,
    format_sources_report,
    format_abc_report,
    format_seasonal_report,
    format_forecast_report,
    format_summary_report,
    keyword_classify as _keyword_classify,
    filter_by_period as _filter_by_period,
)

logger = logging.getLogger(__name__)


class AnalystAgent:
    """Агент-обёртка: делегирует AnalyticsService.

    Доступен только администратору (ADMIN_CHAT_ID).
    """

    def __init__(self, integram_client=None, groq_client=None, groq_model: str = ""):
        self._service = AnalyticsService(
            crm=integram_client,
            groq_client=groq_client,
            groq_model=groq_model,
        )

    def set_crm(self, crm) -> None:
        """Установить CRM-клиент (v1 или v2)."""
        self._service._crm = crm

    async def handle_query(self, query: str) -> str:
        return await self._service.handle_query(query)

    async def get_sales_summary(self, period: str = "all") -> dict:
        return await self._service.get_sales_summary(period)

    async def get_packaging_recommendations(self, period: str = "month") -> list[dict]:
        return await self._service.get_packaging_recommendations(period)
