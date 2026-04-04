"""Тесты модуля startup — единая точка создания сервисов."""

import pytest
from unittest.mock import AsyncMock

from src.startup import Services


class TestServices:
    """Тесты контейнера Services."""

    def test_services_defaults(self):
        svc = Services()
        assert svc.crm is None
        assert svc.auth is None
        assert svc.order_service is None
        assert svc.analytics_service is None
        assert svc.consult_service is None
        assert svc.worker_service is None
        assert svc.delivery_service is None
        assert svc.orchestrator is None

    @pytest.mark.asyncio
    async def test_close_idempotent(self):
        svc = Services()
        await svc.close()
        await svc.close()  # Повторный вызов не падает

    @pytest.mark.asyncio
    async def test_close_with_crm(self):
        crm = AsyncMock()
        svc = Services(crm=crm)
        await svc.close()
        crm.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_close_with_bg_manager(self):
        bg = AsyncMock()
        svc = Services(bg_manager=bg)
        await svc.close()
        bg.stop_all.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_close_handles_crm_error(self):
        crm = AsyncMock()
        crm.close.side_effect = Exception("connection lost")
        svc = Services(crm=crm)
        # Не должен падать
        await svc.close()
