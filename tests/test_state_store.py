"""Тесты StateStore — хранение состояния (in-memory fallback)."""

import pytest

from src.services.state_store import StateStore


class TestStateStoreInMemory:
    """Тесты без Redis (in-memory fallback)."""

    @pytest.mark.asyncio
    async def test_user_style(self):
        store = StateStore()
        assert await store.get_user_style(1) is None
        await store.set_user_style(1, "founder")
        assert await store.get_user_style(1) == "founder"

    @pytest.mark.asyncio
    async def test_admin_mode(self):
        store = StateStore()
        assert not await store.is_admin_mode(1)
        await store.set_admin_mode(1, True)
        assert await store.is_admin_mode(1)
        await store.set_admin_mode(1, False)
        assert not await store.is_admin_mode(1)

    @pytest.mark.asyncio
    async def test_admin_view(self):
        store = StateStore()
        assert await store.get_admin_view(1) == "admin"
        await store.set_admin_view(1, "worker")
        assert await store.get_admin_view(1) == "worker"

    @pytest.mark.asyncio
    async def test_has_redis_false(self):
        store = StateStore()
        assert not store.has_redis

    @pytest.mark.asyncio
    async def test_close_no_redis(self):
        store = StateStore()
        await store.close()  # Не падает
