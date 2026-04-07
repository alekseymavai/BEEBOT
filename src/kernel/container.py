"""DI-контейнер BEEBOT.

Централизованное хранилище сервисов приложения.
Плагины публикуют созданные объекты через set(), получают зависимости через get().

Заменяет Services-датакласс из startup.py: вместо фиксированных полей —
динамический словарь с типизированным доступом.

Пример:
    container.set("crm", crm_client)
    container.set("kb", knowledge_base)

    crm = container.get("crm")                      # Any
    crm = container.get("crm", IntegramClient)       # типизированный
    kb  = container.require("kb", KnowledgeBase)     # Exception если нет
"""

from __future__ import annotations

import logging
from typing import Any, Type, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class Container:
    """Простой DI-контейнер с именованными сервисами."""

    def __init__(self) -> None:
        self._services: dict[str, Any] = {}

    # --- Write ---

    def set(self, name: str, service: Any) -> None:
        """Зарегистрировать сервис под именем.

        Если сервис с таким именем уже зарегистрирован — перезаписывает
        и логирует предупреждение (обычно это ошибка конфигурации).
        """
        if name in self._services:
            logger.warning(
                "Container: сервис '%s' уже зарегистрирован — перезапись.", name
            )
        self._services[name] = service

    # --- Read ---

    def get(self, name: str, type_: Type[T] | None = None) -> T | None:  # type: ignore[return]
        """Получить сервис по имени. Возвращает None если не зарегистрирован."""
        return self._services.get(name)  # type: ignore[return-value]

    def require(self, name: str, type_: Type[T] | None = None) -> T:
        """Получить сервис или бросить RuntimeError если не зарегистрирован."""
        service = self._services.get(name)
        if service is None:
            registered = list(self._services.keys())
            raise RuntimeError(
                f"Container: сервис '{name}' не зарегистрирован. "
                f"Зарегистрированы: {registered}"
            )
        return service  # type: ignore[return-value]

    def has(self, name: str) -> bool:
        """Проверить наличие сервиса."""
        return name in self._services

    # --- Introspection ---

    def keys(self) -> list[str]:
        """Список имён всех зарегистрированных сервисов."""
        return list(self._services.keys())

    def __repr__(self) -> str:
        keys = list(self._services.keys())
        return f"Container(services={keys})"
