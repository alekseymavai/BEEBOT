"""EventEmitter — публикация бизнес-событий в шину.

Сервисы вызывают EventEmitter после записи данных.
Подписчики (SSE, Telegram-нотификатор) реагируют асинхронно.

Best practice: Cosmic Python Ch.8 — «Events and the Message Bus».
Паттерн: write → emit event → subscribers react.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Coroutine

logger = logging.getLogger(__name__)

# Тип callback-подписчика
EventCallback = Callable[[str, dict[str, Any]], Coroutine[Any, Any, None]]


class EventEmitter:
    """Лёгкая шина событий (in-process + опциональный Redis).

    В отличие от EventBus (Redis Streams), работает без Redis.
    Если Redis EventBus подключён — дублирует события туда.
    """

    def __init__(self):
        self._listeners: dict[str, list[EventCallback]] = {}
        self._redis_bus = None  # Optional EventBus

    def set_redis_bus(self, bus) -> None:
        """Подключить Redis EventBus для кросс-процессного обмена."""
        self._redis_bus = bus

    def on(self, event_type: str, callback: EventCallback) -> None:
        """Подписаться на событие."""
        self._listeners.setdefault(event_type, []).append(callback)

    async def emit(self, event_type: str, data: dict[str, Any]) -> None:
        """Опубликовать событие.

        1. Вызвать in-process подписчиков (SSE, Telegram).
        2. Опубликовать в Redis Streams (если подключён).
        """
        # In-process listeners
        for cb in self._listeners.get(event_type, []):
            try:
                await cb(event_type, data)
            except Exception as e:
                logger.warning("EventEmitter: ошибка обработчика %s: %s", event_type, e)

        # Wildcard listeners (*)
        for cb in self._listeners.get("*", []):
            try:
                await cb(event_type, data)
            except Exception as e:
                logger.warning("EventEmitter: ошибка wildcard для %s: %s", event_type, e)

        # Redis EventBus (кросс-процесс)
        if self._redis_bus and self._redis_bus.connected:
            try:
                from src.bus import make_event
                event = make_event(event_type, data)
                await self._redis_bus.publish("stream:events", event)
            except Exception as e:
                logger.warning("EventEmitter: ошибка Redis publish %s: %s", event_type, e)


# Singleton — создаётся один раз, используется всеми сервисами
events = EventEmitter()
