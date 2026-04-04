"""Тесты CircuitBreaker — защита от каскадных сбоев."""

from src.services.circuit_breaker import CircuitBreaker, CircuitState


class TestCircuitBreaker:
    def test_initial_state_closed(self):
        cb = CircuitBreaker(threshold=3)
        assert cb.state == CircuitState.CLOSED
        assert cb.allow_request()

    def test_opens_after_threshold(self):
        cb = CircuitBreaker(threshold=3, timeout=60)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert not cb.allow_request()

    def test_success_resets(self):
        cb = CircuitBreaker(threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        assert cb.state == CircuitState.CLOSED
        assert cb._failure_count == 0

    def test_half_open_after_timeout(self):
        cb = CircuitBreaker(threshold=1, timeout=0)
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        # timeout=0 → сразу переход в half_open
        assert cb.state == CircuitState.HALF_OPEN
        assert cb.allow_request()

    def test_half_open_to_closed_on_success(self):
        cb = CircuitBreaker(threshold=1, timeout=0)
        cb.record_failure()
        _ = cb.state  # Trigger HALF_OPEN
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_status(self):
        cb = CircuitBreaker(name="test", threshold=5)
        cb.record_failure()
        cb.record_success()
        status = cb.status()
        assert status["state"] == "closed"
        assert status["successes"] == 1
        assert status["threshold"] == 5
