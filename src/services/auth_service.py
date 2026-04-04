"""AuthService — единый источник проверки ролей пользователей.

Консолидирует 6 копий _is_admin() / _is_worker() в один модуль.
НЕ покрывает DEVBOT (отдельная система с DEVBOT_ADMIN_CHAT_ID).
"""

from __future__ import annotations


class AuthService:
    """Проверка ролей: администратор, работник склада, пчеловод."""

    def __init__(
        self,
        *,
        admin_ids: frozenset[int],
        worker_ids: frozenset[int],
        beekeeper_id: int | None = None,
    ) -> None:
        self._admin_ids = admin_ids
        self._worker_ids = worker_ids
        self._beekeeper_id = beekeeper_id

    def is_admin(self, user_id: int) -> bool:
        """Пользователь — администратор (ADMIN_IDS)."""
        return bool(self._admin_ids and user_id in self._admin_ids)

    def is_worker(self, user_id: int) -> bool:
        """Пользователь — работник склада (WORKER_CHAT_IDS)."""
        return bool(self._worker_ids and user_id in self._worker_ids)

    def is_admin_or_worker(self, user_id: int) -> bool:
        """Пользователь — администратор или работник."""
        return self.is_admin(user_id) or self.is_worker(user_id)

    def is_beekeeper(self, user_id: int) -> bool:
        """Пользователь — пчеловод (BEEKEEPER_CHAT_ID)."""
        return self._beekeeper_id is not None and user_id == self._beekeeper_id

    def is_admin_legacy(self, user_id: int) -> bool:
        """Совместимость с admin.py: ADMIN_CHAT_ID или BEEKEEPER_CHAT_ID."""
        if self.is_admin(user_id):
            return True
        return self.is_beekeeper(user_id)
