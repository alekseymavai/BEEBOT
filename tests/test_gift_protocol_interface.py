"""Tests for GiftBroker.suggest_interface() and set_interface_mode() — Phase 11.1."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.gift_protocol import GiftBroker
from src.shared_context import SharedContextStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_broker() -> GiftBroker:
    ctx_store = SharedContextStore()
    with (
        patch("src.orchestrator.Groq"),
        patch("src.orchestrator.BeebotAgent"),
        patch("src.orchestrator.LogistAgent"),
        patch("src.orchestrator.AnalystAgent"),
    ):
        from src.orchestrator import Orchestrator
        orch = Orchestrator()
    broker = GiftBroker(
        orchestrator=orch,
        context_store=ctx_store,
        anamnesis=MagicMock(),
        crm_agent=None,
    )
    return broker


# ---------------------------------------------------------------------------
# suggest_interface
# ---------------------------------------------------------------------------

class TestSuggestInterface:
    def test_non_worker_always_gets_client(self):
        broker = _make_broker()
        assert broker.suggest_interface(1001, is_worker=False) == "client"

    def test_worker_with_default_mode_gets_worker(self):
        broker = _make_broker()
        assert broker.suggest_interface(1002, is_worker=True) == "worker"

    def test_worker_who_switched_to_client_gets_client(self):
        broker = _make_broker()
        broker.set_interface_mode(1003, "client")
        assert broker.suggest_interface(1003, is_worker=True) == "client"

    def test_worker_after_reset_gets_worker(self):
        broker = _make_broker()
        broker.set_interface_mode(1004, "client")
        broker.set_interface_mode(1004, "default")
        assert broker.suggest_interface(1004, is_worker=True) == "worker"

    def test_unknown_user_worker_gets_worker_by_default(self):
        """User who has never interacted defaults to worker interface."""
        broker = _make_broker()
        assert broker.suggest_interface(9999, is_worker=True) == "worker"


# ---------------------------------------------------------------------------
# set_interface_mode
# ---------------------------------------------------------------------------

class TestSetInterfaceMode:
    def test_set_client_mode_stored_in_context(self):
        broker = _make_broker()
        broker.set_interface_mode(2001, "client")
        ctx = broker._ctx.get(2001)
        assert ctx.interface_mode == "client"

    def test_set_default_mode_resets_context(self):
        broker = _make_broker()
        broker.set_interface_mode(2002, "client")
        broker.set_interface_mode(2002, "default")
        ctx = broker._ctx.get(2002)
        assert ctx.interface_mode == "default"

    def test_mode_change_touches_context(self):
        """set_interface_mode should update updated_at."""
        import time
        broker = _make_broker()
        before = time.monotonic()
        broker.set_interface_mode(2003, "client")
        ctx = broker._ctx.get(2003)
        assert ctx.updated_at >= before
