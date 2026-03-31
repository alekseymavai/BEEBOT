"""Мониторинг SSH-туннеля hive:8990 → Groq API.

Запускается как фоновая asyncio-задача, проверяет порт 8990 каждые 60 секунд.
При смене состояния (up→down или down→up) вызывает alert_fn.

Graceful: если порт 8990 не слушается (dev-машина без туннеля),
считает туннель здоровым (режим разработки).
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Optional

logger = logging.getLogger(__name__)

_TUNNEL_HOST = "127.0.0.1"
_TUNNEL_PORT = 8990
_CHECK_INTERVAL = 60          # секунд между проверками
_CONNECT_TIMEOUT = 5.0        # таймаут TCP-коннекта


class TunnelMonitor:
    """Фоновый монитор TCP-доступности прокси-туннеля на localhost:8990.

    Attributes:
        is_healthy: текущее состояние туннеля (True = доступен).
    """

    def __init__(
        self,
        alert_fn: Optional[Callable[[str], Awaitable[None]]] = None,
        host: str = _TUNNEL_HOST,
        port: int = _TUNNEL_PORT,
        interval: int = _CHECK_INTERVAL,
    ) -> None:
        self._host = host
        self._port = port
        self._interval = interval
        self._alert_fn = alert_fn
        # None = ещё не проверялось (первый чек не считается изменением состояния)
        self._healthy: Optional[bool] = None
        self._dev_mode: bool = False   # True если порт не слушается вообще

    @property
    def is_healthy(self) -> bool:
        """Возвращает True если туннель доступен (или режим разработки)."""
        if self._dev_mode:
            return True
        # До первой проверки считаем здоровым
        return self._healthy if self._healthy is not None else True

    async def check_once(self) -> bool:
        """Проверить TCP-соединение с портом туннеля.

        Returns:
            True — порт доступен.
            True — порт не слушается вообще (dev mode, graceful).
            False — порт слушается, но соединение закрылось / таймаут.
        """
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(self._host, self._port),
                timeout=_CONNECT_TIMEOUT,
            )
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
            return True
        except ConnectionRefusedError:
            # Порт не слушается — это dev-машина без туннеля
            logger.debug("TunnelMonitor: порт %d не слушается — dev-режим", self._port)
            self._dev_mode = True
            return True
        except (TimeoutError, asyncio.TimeoutError, OSError) as exc:
            logger.debug("TunnelMonitor: соединение не удалось: %s", exc)
            return False

    async def _send_alert(self, text: str) -> None:
        if self._alert_fn is None:
            return
        try:
            await self._alert_fn(text)
        except Exception as exc:
            logger.warning("TunnelMonitor: не удалось отправить алерт: %s", exc)

    async def run(self) -> None:
        """Бесконечная фоновая задача — проверяет туннель каждые _interval секунд."""
        logger.info("TunnelMonitor запущен (порт %d, интервал %ds)", self._port, self._interval)
        while True:
            try:
                healthy = await self.check_once()
                prev = self._healthy

                if prev is None:
                    # Первая проверка — инициализируем состояние без алерта
                    self._healthy = healthy
                    if not healthy:
                        logger.warning("TunnelMonitor: туннель недоступен при старте")
                elif healthy != prev:
                    # Смена состояния — алертим
                    self._healthy = healthy
                    if not healthy:
                        msg = "⚠️ Groq-туннель недоступен — бот работает в режиме FAQ"
                        logger.warning("TunnelMonitor: туннель DOWN")
                        await self._send_alert(msg)
                    else:
                        msg = "✅ Groq-туннель восстановлен — бот работает в штатном режиме"
                        logger.info("TunnelMonitor: туннель UP (восстановлен)")
                        await self._send_alert(msg)
                # else: состояние не изменилось — молчим
            except Exception as exc:
                logger.exception("TunnelMonitor: неожиданная ошибка при проверке: %s", exc)

            await asyncio.sleep(self._interval)
