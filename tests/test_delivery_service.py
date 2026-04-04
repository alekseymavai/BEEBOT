"""Тесты DeliveryService — фасад доставки."""

import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from src.services.delivery_service import DeliveryService


class TestCalculate:
    @pytest.mark.asyncio
    async def test_delegates_to_calculator(self):
        calc = AsyncMock()
        quote = SimpleNamespace(provider="СДЭК", cost=350, days=3)
        calc.calculate.return_value = [quote]

        svc = DeliveryService(calculator=calc)
        result = await svc.calculate(to_city="Москва", weight_grams=500)
        assert len(result) == 1
        assert result[0].provider == "СДЭК"
        calc.calculate.assert_called_once_with(
            to_city="Москва", weight_grams=500, from_city=None,
        )

    @pytest.mark.asyncio
    async def test_no_calculator_raises(self):
        svc = DeliveryService()
        with pytest.raises(RuntimeError):
            await svc.calculate(to_city="Москва", weight_grams=500)


class TestTracker:
    def test_tracker_property(self):
        tracker = object()
        svc = DeliveryService(tracker=tracker)
        assert svc.tracker is tracker

    def test_tracker_none_by_default(self):
        svc = DeliveryService()
        assert svc.tracker is None
