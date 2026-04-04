"""StateStore — хранение состояния в Redis вместо in-memory dict.

Заменяет module-level globals из src/routers/_state.py:
- _user_styles (голос улья)
- _admin_mode_users
- _admin_view_mode

Данные переживают рестарт бота. Оба процесса (бот и бэкенд)
видят одно состояние.

Best practice: redis.io — «Use Redis hashes for structured objects,
strings with TTL for cache entries.»

Anti-pattern avoided: «In-memory FSM state that is lost on restart.»
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Префиксы ключей в Redis
_PREFIX = "beebot:"
_USER_STYLES_KEY = f"{_PREFIX}user_styles"
_ADMIN_MODE_KEY = f"{_PREFIX}admin_mode"
_ADMIN_VIEW_KEY = f"{_PREFIX}admin_view"
_WORKER_CHECKLIST_KEY = f"{_PREFIX}worker:checklist"


class StateStore:
    """Хранение состояния в Redis с fallback на in-memory dict.

    Если Redis недоступен — работает как обычный dict (backward compat).
    """

    def __init__(self, redis=None):
        self._redis = redis
        # In-memory fallback
        self._user_styles: dict[int, str] = {}
        self._admin_mode_users: set[int] = set()
        self._admin_view_mode: dict[int, str] = {}

    @property
    def has_redis(self) -> bool:
        return self._redis is not None

    # ------------------------------------------------------------------
    # User Styles (Голос Улья)
    # ------------------------------------------------------------------

    async def get_user_style(self, user_id: int) -> Optional[str]:
        if self._redis:
            val = await self._redis.hget(_USER_STYLES_KEY, str(user_id))
            return val
        return self._user_styles.get(user_id)

    async def set_user_style(self, user_id: int, style: str) -> None:
        if self._redis:
            await self._redis.hset(_USER_STYLES_KEY, str(user_id), style)
        else:
            self._user_styles[user_id] = style

    # ------------------------------------------------------------------
    # Admin Mode
    # ------------------------------------------------------------------

    async def is_admin_mode(self, user_id: int) -> bool:
        if self._redis:
            return bool(await self._redis.sismember(_ADMIN_MODE_KEY, str(user_id)))
        return user_id in self._admin_mode_users

    async def set_admin_mode(self, user_id: int, enabled: bool) -> None:
        if self._redis:
            if enabled:
                await self._redis.sadd(_ADMIN_MODE_KEY, str(user_id))
            else:
                await self._redis.srem(_ADMIN_MODE_KEY, str(user_id))
        else:
            if enabled:
                self._admin_mode_users.add(user_id)
            else:
                self._admin_mode_users.discard(user_id)

    # ------------------------------------------------------------------
    # Admin View Mode
    # ------------------------------------------------------------------

    async def get_admin_view(self, user_id: int) -> str:
        if self._redis:
            val = await self._redis.hget(_ADMIN_VIEW_KEY, str(user_id))
            return val or "admin"
        return self._admin_view_mode.get(user_id, "admin")

    async def set_admin_view(self, user_id: int, view: str) -> None:
        if self._redis:
            await self._redis.hset(_ADMIN_VIEW_KEY, str(user_id), view)
        else:
            self._admin_view_mode[user_id] = view

    # ------------------------------------------------------------------
    # Worker Checklists (переживают рестарт!)
    # ------------------------------------------------------------------

    async def get_checked_items(self, worker_id: int, order_id: int) -> set[int]:
        if self._redis:
            key = f"{_WORKER_CHECKLIST_KEY}:{worker_id}:{order_id}"
            items = await self._redis.smembers(key)
            return {int(i) for i in items}
        return set()

    async def toggle_checked_item(self, worker_id: int, order_id: int, item_id: int) -> None:
        if self._redis:
            key = f"{_WORKER_CHECKLIST_KEY}:{worker_id}:{order_id}"
            if await self._redis.sismember(key, str(item_id)):
                await self._redis.srem(key, str(item_id))
            else:
                await self._redis.sadd(key, str(item_id))

    async def clear_checklist(self, worker_id: int, order_id: int) -> None:
        if self._redis:
            key = f"{_WORKER_CHECKLIST_KEY}:{worker_id}:{order_id}"
            await self._redis.delete(key)

    # ------------------------------------------------------------------
    # Подключение Redis
    # ------------------------------------------------------------------

    async def connect_redis(self, redis_url: str) -> None:
        """Подключиться к Redis."""
        try:
            import redis.asyncio as aioredis
            self._redis = aioredis.from_url(
                redis_url, decode_responses=True,
                socket_connect_timeout=5,
            )
            await self._redis.ping()
            logger.info("StateStore: Redis подключён (%s)", redis_url)
        except Exception as e:
            logger.warning("StateStore: Redis недоступен, fallback на in-memory: %s", e)
            self._redis = None

    async def close(self) -> None:
        if self._redis:
            await self._redis.aclose()
            self._redis = None
