"""Тесты EventEmitter — публикация бизнес-событий."""

import pytest

from src.services.event_emitter import EventEmitter


class TestEventEmitter:
    @pytest.mark.asyncio
    async def test_emit_calls_listener(self):
        emitter = EventEmitter()
        received = []

        async def handler(event_type, data):
            received.append((event_type, data))

        emitter.on("order.created", handler)
        await emitter.emit("order.created", {"id": 1})
        assert len(received) == 1
        assert received[0] == ("order.created", {"id": 1})

    @pytest.mark.asyncio
    async def test_wildcard_listener(self):
        emitter = EventEmitter()
        received = []

        async def handler(event_type, data):
            received.append(event_type)

        emitter.on("*", handler)
        await emitter.emit("order.created", {})
        await emitter.emit("order.status_changed", {})
        assert received == ["order.created", "order.status_changed"]

    @pytest.mark.asyncio
    async def test_no_listeners(self):
        emitter = EventEmitter()
        # Не падает без подписчиков
        await emitter.emit("unknown.event", {"data": True})

    @pytest.mark.asyncio
    async def test_listener_error_does_not_crash(self):
        emitter = EventEmitter()

        async def bad_handler(event_type, data):
            raise RuntimeError("oops")

        emitter.on("test", bad_handler)
        # Не падает
        await emitter.emit("test", {})
