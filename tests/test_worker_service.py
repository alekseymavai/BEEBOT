"""Тесты WorkerService — состояние работника, чеклисты, очередь."""

import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from src.services.worker_service import WorkerService


def _make_item(id=1):
    return SimpleNamespace(id=id)


class TestWorkerState:
    def test_initially_not_busy(self):
        svc = WorkerService()
        assert svc.is_busy(100) is False

    def test_set_busy(self):
        svc = WorkerService()
        svc.set_busy(100)
        assert svc.is_busy(100) is True

    @pytest.mark.asyncio
    async def test_set_idle(self):
        svc = WorkerService()
        svc.set_busy(100)
        await svc.set_idle(100)
        assert svc.is_busy(100) is False

    def test_receive_accepted_when_idle(self):
        svc = WorkerService()
        assert svc.receive(100, gift="test") == "ACCEPTED"

    def test_receive_deferred_when_busy(self):
        svc = WorkerService()
        svc.set_busy(100)
        assert svc.receive(100, gift="test") == "DEFERRED"

    @pytest.mark.asyncio
    async def test_deferred_gifts_delivered_on_idle(self):
        svc = WorkerService()
        svc.set_busy(100)
        svc.receive(100, gift="gift1")
        svc.receive(100, gift="gift2")

        delivered = []
        async def deliver_fn(worker_id, gift):
            delivered.append(gift)

        await svc.set_idle(100, deliver_fn=deliver_fn)
        assert delivered == ["gift1", "gift2"]


class TestChecklist:
    def test_toggle_item(self):
        svc = WorkerService()
        svc.toggle_item(100, 1, 10)
        assert 10 in svc.get_checked(100, 1)
        svc.toggle_item(100, 1, 10)
        assert 10 not in svc.get_checked(100, 1)

    def test_clear_checklist(self):
        svc = WorkerService()
        svc.toggle_item(100, 1, 10)
        svc.clear_checklist(100, 1)
        assert svc.get_checked(100, 1) == set()

    def test_is_fully_checked(self):
        svc = WorkerService()
        items = [_make_item(id=10), _make_item(id=20)]
        assert svc.is_fully_checked(100, 1, items) is False
        svc.toggle_item(100, 1, 10)
        svc.toggle_item(100, 1, 20)
        assert svc.is_fully_checked(100, 1, items) is True

    def test_empty_items_not_fully_checked(self):
        svc = WorkerService()
        assert svc.is_fully_checked(100, 1, []) is False


class TestGetQueue:
    @pytest.mark.asyncio
    async def test_filters_by_status(self):
        crm = AsyncMock()
        crm.get_orders.return_value = [
            SimpleNamespace(status="Новый"),
            SimpleNamespace(status="Отправлен"),
            SimpleNamespace(status="В сборке"),
        ]
        svc = WorkerService()
        queue = await svc.get_queue(crm)
        assert len(queue) == 2
        assert all(o.status in ("Новый", "В сборке") for o in queue)
