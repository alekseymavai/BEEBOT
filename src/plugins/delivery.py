"""DeliveryPlugin — DeliveryService (СДЭК + Почта России).

Зависимости: (нет)

Публикует в контейнере:
  "delivery_service" → DeliveryService
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.kernel.plugin import Plugin

if TYPE_CHECKING:
    from src.kernel.container import Container

logger = logging.getLogger(__name__)


class DeliveryPlugin(Plugin):
    name = "delivery"
    dependencies: list[str] = []

    async def setup(self, container: "Container") -> None:
        from src.delivery.calculator import DeliveryCalculator
        from src.services.delivery_service import DeliveryService

        delivery_service = DeliveryService(calculator=DeliveryCalculator())
        container.set("delivery_service", delivery_service)
        logger.info("DeliveryPlugin: инициализирован.")
