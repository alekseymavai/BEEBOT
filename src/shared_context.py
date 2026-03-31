"""SharedContext — рабочая память пользователя с TTL.

Единый per-user контейнер, заменяющий разрозненные _histories и
случайные dict-переменные. Передаётся агентам вместо разрозненных параметров.

Жизненный цикл:
  1. Пользователь пишет — SharedContextStore.get(user_id) возвращает/создаёт контекст
  2. Оркестратор пишет историю диалога через ctx.append_history()
  3. GiftBroker читает ctx.dialog_history перед отправкой Gift
  4. По истечении TTL (30 мин) контекст вытесняется при следующем evict_stale()
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

_DIALOG_TTL = 30 * 60  # 30 минут
_MAX_HISTORY = 5        # максимум пар вопрос-ответ


@dataclass
class UserContext:
    """Рабочая память одного пользователя на время сессии."""

    user_id: int
    dialog_history: list[dict] = field(default_factory=list)    # последние 5 пар
    active_order: dict | None = None                             # текущий заказ в FSM
    health_facts: list[str] = field(default_factory=list)       # из SQLite + Integram
    interests: list[str] = field(default_factory=list)          # упомянутые продукты
    last_products_hint: list[str] = field(default_factory=list) # из онтологии
    checklist: dict = field(default_factory=dict)               # {order_id: set[item_id]} для WorkerAgent
    updated_at: float = field(default_factory=time.monotonic)

    def is_fresh(self, ttl: float = _DIALOG_TTL) -> bool:
        return (time.monotonic() - self.updated_at) < ttl

    def touch(self) -> None:
        self.updated_at = time.monotonic()

    def append_history(self, query: str, response: str) -> None:
        """Добавить пару вопрос-ответ; автоматически обрезает до _MAX_HISTORY пар."""
        self.dialog_history.append({"role": "user", "content": query})
        self.dialog_history.append({"role": "assistant", "content": response})
        max_msgs = _MAX_HISTORY * 2
        if len(self.dialog_history) > max_msgs:
            self.dialog_history = self.dialog_history[-max_msgs:]
        self.touch()

    def get_history(self) -> list[dict]:
        return list(self.dialog_history)


class SharedContextStore:
    """Хранилище UserContext по user_id с автоматическим TTL-вытеснением."""

    def __init__(self, ttl: float = _DIALOG_TTL) -> None:
        self._ttl = ttl
        self._store: dict[int, UserContext] = {}

    def get(self, user_id: int) -> UserContext:
        """Вернуть или создать UserContext для пользователя."""
        ctx = self._store.get(user_id)
        if ctx is None or not ctx.is_fresh(self._ttl):
            ctx = UserContext(user_id=user_id)
            self._store[user_id] = ctx
        return ctx

    def evict_stale(self) -> int:
        """Удалить просроченные контексты. Возвращает число удалённых."""
        stale = [uid for uid, ctx in self._store.items() if not ctx.is_fresh(self._ttl)]
        for uid in stale:
            del self._store[uid]
        return len(stale)

    def __contains__(self, user_id: int) -> bool:
        return user_id in self._store and self._store[user_id].is_fresh(self._ttl)

    def __len__(self) -> int:
        return sum(1 for ctx in self._store.values() if ctx.is_fresh(self._ttl))
