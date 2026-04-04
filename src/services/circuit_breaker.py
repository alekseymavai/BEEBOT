"""CircuitBreaker — защита от каскадных сбоев при недоступности CRM.

Best practice: Microsoft Azure — Circuit Breaker Pattern.
Три состояния: CLOSED (нормальная работа), OPEN (отказ, fallback),
HALF_OPEN (пробный запрос после паузы).

Использование:
    breaker = CircuitBreaker(threshold=5, timeout=30)

    async def call_crm():
        if not breaker.allow_request():
            return cached_data  # fallback
        try:
            result = await crm.get_orders()
            breaker.record_success()
            return result
        except Exception as e:
            breaker.record_failure()
            raise
"""

from __future__ import annotations

import logging
import time
from enum import Enum

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED = "closed"          # Нормальная работа
    OPEN = "open"              # Отказ, все запросы блокируются
    HALF_OPEN = "half_open"    # Пробный запрос


class CircuitBreaker:
    """Circuit Breaker для внешних сервисов (CRM, LLM, Delivery API)."""

    def __init__(
        self,
        name: str = "default",
        threshold: int = 5,
        timeout: float = 30.0,
    ):
        """
        Args:
            name: имя (для логов).
            threshold: количество ошибок до открытия.
            timeout: секунды до перехода в half_open.
        """
        self.name = name
        self.threshold = threshold
        self.timeout = timeout

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: float = 0
        self._success_count = 0

    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN:
            # Проверяем, не пора ли перейти в half_open
            if time.monotonic() - self._last_failure_time >= self.timeout:
                self._state = CircuitState.HALF_OPEN
                logger.info("CircuitBreaker[%s]: OPEN → HALF_OPEN", self.name)
        return self._state

    def allow_request(self) -> bool:
        """Разрешён ли запрос?"""
        state = self.state
        if state == CircuitState.CLOSED:
            return True
        if state == CircuitState.HALF_OPEN:
            return True  # Один пробный запрос
        return False  # OPEN — все блокируются

    def record_success(self) -> None:
        """Успешный запрос."""
        if self._state == CircuitState.HALF_OPEN:
            logger.info("CircuitBreaker[%s]: HALF_OPEN → CLOSED", self.name)
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count += 1

    def record_failure(self) -> None:
        """Неуспешный запрос."""
        self._failure_count += 1
        self._last_failure_time = time.monotonic()

        if self._failure_count >= self.threshold:
            old = self._state
            self._state = CircuitState.OPEN
            if old != CircuitState.OPEN:
                logger.warning(
                    "CircuitBreaker[%s]: %s → OPEN (failures=%d)",
                    self.name, old.value, self._failure_count,
                )

    def status(self) -> dict:
        """Для /api/health."""
        return {
            "state": self.state.value,
            "failures": self._failure_count,
            "threshold": self.threshold,
            "successes": self._success_count,
        }
