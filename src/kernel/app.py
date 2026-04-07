"""BeeBotApp — ядро микроядерной архитектуры BEEBOT.

Ядро отвечает за:
  1. Регистрацию плагинов
  2. Топологическую сортировку по зависимостям
  3. Последовательную инициализацию (setup)
  4. Подключение роутеров к Dispatcher и FastAPI
  5. Запуск фоновых задач
  6. Graceful shutdown в обратном порядке

Плагины — автономные модули. Ядро о конкретных плагинах не знает.

Использование:
    app = BeeBotApp(bot, dp)
    app.register(CrmPlugin())
    app.register(KnowledgePlugin())
    app.register(AgentsPlugin())
    ...
    await app.start(fastapi_app, bg_manager)
    # при остановке:
    await app.stop()
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Coroutine, Optional

from src.kernel.container import Container
from src.kernel.plugin import Plugin

if TYPE_CHECKING:
    from aiogram import Bot, Dispatcher
    from fastapi import FastAPI

logger = logging.getLogger(__name__)

AlertFn = Callable[[str], Coroutine[Any, Any, None]]


def _topo_sort(plugins: list[Plugin]) -> list[Plugin]:
    """Топологическая сортировка плагинов по зависимостям (Kahn's algorithm).

    Raises:
        ValueError: если обнаружен циклический граф зависимостей.
        ValueError: если зависимость объявлена, но плагин не зарегистрирован.
    """
    by_name: dict[str, Plugin] = {p.name: p for p in plugins}

    # Проверяем что все зависимости зарегистрированы
    for p in plugins:
        for dep in getattr(p, "dependencies", []):
            if dep not in by_name:
                raise ValueError(
                    f"Плагин '{p.name}' зависит от '{dep}', "
                    f"но такой плагин не зарегистрирован."
                )

    # in-degree (сколько зависимостей не удовлетворено)
    in_degree: dict[str, int] = {p.name: 0 for p in plugins}
    dependents: dict[str, list[str]] = {p.name: [] for p in plugins}

    for p in plugins:
        for dep in getattr(p, "dependencies", []):
            in_degree[p.name] += 1
            dependents[dep].append(p.name)

    queue = [name for name, deg in in_degree.items() if deg == 0]
    result: list[Plugin] = []

    while queue:
        name = queue.pop(0)
        result.append(by_name[name])
        for dependent in dependents[name]:
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)

    if len(result) != len(plugins):
        remaining = [name for name, deg in in_degree.items() if deg > 0]
        raise ValueError(
            f"Циклические зависимости между плагинами: {remaining}"
        )

    return result


class BeeBotApp:
    """Ядро BEEBOT — регистрирует плагины и управляет их lifecycle."""

    def __init__(self, bot: "Bot", dp: "Dispatcher") -> None:
        self.bot = bot
        self.dp = dp
        self.container = Container()
        self._plugins: list[Plugin] = []
        self._sorted: list[Plugin] = []  # порядок инициализации

    # ------------------------------------------------------------------ #
    # Регистрация                                                          #
    # ------------------------------------------------------------------ #

    def register(self, plugin: Plugin) -> "BeeBotApp":
        """Зарегистрировать плагин. Возвращает self для chaining."""
        if not plugin.name:
            raise ValueError(f"Плагин {type(plugin).__name__} не задал name.")
        existing = [p.name for p in self._plugins]
        if plugin.name in existing:
            raise ValueError(f"Плагин '{plugin.name}' уже зарегистрирован.")
        self._plugins.append(plugin)
        logger.debug("Плагин зарегистрирован: %s", plugin.name)
        return self

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    async def start(self) -> None:
        """Инициализировать все плагины и подключить роутеры.

        Порядок:
          1. setup() — создание сервисов (по графу зависимостей)
          2. register_routers() — подключение aiogram Router-ов
          3. register_api() — подключение FastAPI APIRouter-ов
          4. get_bg_tasks() → bg_manager.start() — запуск фоновых задач
             (bg_manager берётся из контейнера после setup)
        """
        # Топологическая сортировка
        self._sorted = _topo_sort(self._plugins)
        names = [p.name for p in self._sorted]
        logger.info("Порядок инициализации плагинов: %s", names)

        # 1. setup — создание сервисов
        for plugin in self._sorted:
            logger.info("  → setup: %s", plugin.name)
            await plugin.setup(self.container)

        # 2. register_routers — подключение Telegram-роутеров
        for plugin in self._sorted:
            plugin.register_routers(self.dp)

        # 3. register_api — подключение FastAPI-роутеров
        fastapi_app = self.container.get("fastapi_app")
        if fastapi_app is not None:
            for plugin in self._sorted:
                plugin.register_api(fastapi_app)

        # 4. get_bg_tasks — запуск фоновых задач через bg_manager из контейнера
        bg_manager = self.container.get("bg_manager")
        if bg_manager is not None:
            for plugin in self._sorted:
                for task in plugin.get_bg_tasks():
                    await bg_manager.start(task.name, task.coro_fn)
                    logger.info("  BG task запущена: %s", task.name)

        logger.info(
            "BeeBotApp запущен. Плагинов: %d. Сервисов: %s",
            len(self._sorted),
            self.container.keys(),
        )

    async def stop(self) -> None:
        """Graceful shutdown — teardown в обратном порядке инициализации."""
        logger.info("BeeBotApp останавливается...")
        for plugin in reversed(self._sorted):
            try:
                logger.info("  ← teardown: %s", plugin.name)
                await plugin.teardown()
            except Exception as e:
                logger.warning("Ошибка teardown плагина '%s': %s", plugin.name, e)
        logger.info("BeeBotApp остановлен.")

    # ------------------------------------------------------------------ #
    # Удобный доступ к контейнеру                                          #
    # ------------------------------------------------------------------ #

    def get(self, name: str) -> Any:
        """Получить сервис из контейнера (shortcut)."""
        return self.container.get(name)

    def require(self, name: str) -> Any:
        """Получить сервис или бросить RuntimeError (shortcut)."""
        return self.container.require(name)
