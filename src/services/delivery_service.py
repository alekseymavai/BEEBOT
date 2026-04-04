"""DeliveryService — фасад для расчёта доставки и трекинга.

Тонкая обёртка вокруг delivery/calculator.py и delivery/tracker.py.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.delivery.calculator import DeliveryCalculator, DeliveryQuote
    from src.delivery.tracker import OrderTracker

logger = logging.getLogger(__name__)


class DeliveryService:
    """Единая точка доступа к расчёту доставки и трекингу."""

    def __init__(
        self,
        calculator: "DeliveryCalculator | None" = None,
        tracker: "OrderTracker | None" = None,
    ) -> None:
        self._calculator = calculator
        self._tracker = tracker

    async def calculate(
        self,
        to_city: str,
        weight_grams: int,
        from_city: str | None = None,
    ) -> list["DeliveryQuote"]:
        """Рассчитать варианты доставки."""
        if not self._calculator:
            raise RuntimeError("DeliveryCalculator не инициализирован")
        return await self._calculator.calculate(
            to_city=to_city,
            weight_grams=weight_grams,
            from_city=from_city,
        )

    @property
    def tracker(self) -> "OrderTracker | None":
        return self._tracker
