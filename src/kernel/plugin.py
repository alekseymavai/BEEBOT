"""Базовый класс плагина микроядерной архитектуры BEEBOT.

Каждый плагин — автономный модуль с объявленными зависимостями.
Ядро (BeeBotApp) выстраивает граф зависимостей и инициализирует плагины
в правильном порядке.

Lifecycle плагина:
  setup(container)        → инициализация, регистрация сервисов в контейнере
  register_routers(dp)    → подключение aiogram Router-ов к Dispatcher
  register_api(app)       → подключение FastAPI роутеров
  get_bg_tasks()          → список фоновых задач для BackgroundTaskManager
  teardown()              → graceful shutdown
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Coroutine

if TYPE_CHECKING:
    from aiogram import Dispatcher
    from fastapi import FastAPI
    from src.kernel.container import Container


@dataclass
class BgTask:
    """Описание фоновой задачи для BackgroundTaskManager."""

    name: str
    coro_fn: Callable[[], Coroutine[Any, Any, None]]


class Plugin(ABC):
    """Абстрактный плагин BEEBOT.

    Подкласс объявляет:
      - name: str — уникальное имя плагина (используется как ключ в реестре)
      - dependencies: list[str] — имена плагинов, которые должны быть
        инициализированы раньше этого.

    Ядро вызывает методы в порядке:
      setup → register_routers → register_api → get_bg_tasks → (работа) → teardown
    """

    # Имя плагина — должно быть уникальным
    name: str = ""

    # Имена плагинов-зависимостей (должны быть инициализированы раньше)
    dependencies: list[str] = field(default_factory=list)

    @abstractmethod
    async def setup(self, container: "Container") -> None:
        """Инициализировать плагин и зарегистрировать сервисы в контейнере.

        Здесь создаются все сервисы плагина и публикуются через container.set().
        Зависимости читаются через container.get().
        """

    def register_routers(self, dp: "Dispatcher") -> None:
        """Подключить aiogram Router-ы к Dispatcher.

        Переопределить если плагин обрабатывает Telegram-апдейты.
        Порядок роутеров задаётся внутри плагина.
        """

    def register_api(self, app: "FastAPI") -> None:
        """Подключить FastAPI APIRouter-ы к приложению.

        Переопределить если плагин предоставляет HTTP-эндпоинты.
        """

    def get_bg_tasks(self) -> list[BgTask]:
        """Вернуть список фоновых задач.

        Ядро передаёт их в BackgroundTaskManager.start().
        По умолчанию — пустой список.
        """
        return []

    async def teardown(self) -> None:
        """Graceful shutdown ресурсов плагина.

        Вызывается при остановке приложения в обратном порядке инициализации.
        """
