"""Unit tests for src/orchestrator.py — intent classification and routing."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.orchestrator import (
    Orchestrator,
    _classify_intent,
    _DIALOG_TTL_SECONDS,
    OrchestratorState,
)


# ---------------------------------------------------------------------------
# _classify_intent
# ---------------------------------------------------------------------------

class TestClassifyIntent:
    """Tests for the _classify_intent helper."""

    def _make_groq_response(self, content: str):
        response = MagicMock()
        response.choices[0].message.content = content
        return response

    def test_returns_consult_for_product_question(self):
        client = MagicMock()
        client.chat.completions.create.return_value = self._make_groq_response("consult")
        result = _classify_intent(client, "llama-3.3-70b-versatile", "Как принимать прополис?")
        assert result == "consult"

    def test_returns_order_for_buy_intent(self):
        client = MagicMock()
        client.chat.completions.create.return_value = self._make_groq_response("order")
        result = _classify_intent(client, "llama-3.3-70b-versatile", "Хочу купить пергу")
        assert result == "order"

    def test_returns_track_for_shipping_question(self):
        """'track' is a valid intent; 'delivery' is not — it falls back to consult."""
        client = MagicMock()
        client.chat.completions.create.return_value = self._make_groq_response("track")
        result = _classify_intent(client, "llama-3.3-70b-versatile", "Где мой трек-номер?")
        assert result == "track"

    def test_returns_stats_for_analytics_question(self):
        client = MagicMock()
        client.chat.completions.create.return_value = self._make_groq_response("stats")
        result = _classify_intent(client, "llama-3.3-70b-versatile", "Покажи статистику продаж")
        assert result == "stats"

    def test_fallback_to_consult_on_unknown_intent(self):
        """Unknown response from LLM should fall back to consult."""
        client = MagicMock()
        client.chat.completions.create.return_value = self._make_groq_response("unknown_value")
        result = _classify_intent(client, "llama-3.3-70b-versatile", "что-то непонятное")
        assert result == "consult"

    def test_fallback_to_consult_on_api_error(self):
        """API error during classification should fall back to consult."""
        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("Network error")
        result = _classify_intent(client, "llama-3.3-70b-versatile", "какой-то вопрос")
        assert result == "consult"

    def test_strips_whitespace_from_response(self):
        """LLM response may include leading/trailing whitespace."""
        client = MagicMock()
        client.chat.completions.create.return_value = self._make_groq_response("  order  ")
        result = _classify_intent(client, "llama-3.3-70b-versatile", "Хочу заказать")
        assert result == "order"


# ---------------------------------------------------------------------------
# Orchestrator.route — routing logic
# ---------------------------------------------------------------------------

class TestOrchestratorRouting:
    """Tests for Orchestrator.route() and agent dispatch."""

    def _make_orchestrator(self):
        """Create Orchestrator with mocked dependencies."""
        with (
            patch("src.orchestrator.Groq"),
            patch("src.orchestrator.BeebotAgent"),
            patch("src.orchestrator.LogistAgent"),
            patch("src.orchestrator.AnalystAgent"),
        ):
            orch = Orchestrator()
        return orch

    @pytest.mark.asyncio
    async def test_consult_intent_routes_to_beebot(self):
        """consult intent should invoke BeebotAgent.answer."""
        orch = self._make_orchestrator()
        orch._beebot.answer = MagicMock(return_value=("Ответ о прополисе", [{"source": "pdf:X"}]))

        with patch("src.orchestrator._classify_intent", return_value="consult"):
            response, chunks = await orch.route(1001, "Как принимать прополис?")

        orch._beebot.answer.assert_called_once_with(
            "Как принимать прополис?",
            history=[],
            style=None,
            memory_facts=None,
            advice_text=None,
            user_name=None,
            system_prompt_override=None,
        )
        assert response == "Ответ о прополисе"
        assert chunks == [{"source": "pdf:X"}]

    @pytest.mark.asyncio
    async def test_order_intent_routes_to_logist(self):
        """order intent should route to logist node (returns empty — FSM handled in bot.py)."""
        orch = self._make_orchestrator()

        with patch("src.orchestrator._classify_intent", return_value="order"):
            response, chunks = await orch.route(1002, "Хочу купить пергу")

        assert chunks == []
        assert response == ""

    @pytest.mark.asyncio
    async def test_track_intent_routes_to_passthrough(self):
        """track intent should route to passthrough node (handled in bot.py)."""
        orch = self._make_orchestrator()

        with patch("src.orchestrator._classify_intent", return_value="track"):
            response, chunks = await orch.route(1003, "Где мой заказ?")

        assert chunks == []
        assert response == ""

    @pytest.mark.asyncio
    async def test_stats_intent_routes_to_analyst(self):
        """stats intent should invoke AnalystAgent."""
        orch = self._make_orchestrator()
        orch._analyst.get_sales_summary = AsyncMock(side_effect=NotImplementedError)

        with patch("src.orchestrator._classify_intent", return_value="stats"):
            response, chunks = await orch.route(1004, "Статистика продаж")

        assert chunks == []
        assert len(response) > 0

    @pytest.mark.asyncio
    async def test_classification_error_falls_back_to_beebot(self):
        """If intent classification fails, route to BEEBOT (fallback consult)."""
        orch = self._make_orchestrator()
        orch._beebot.answer = MagicMock(return_value=("Fallback ответ", []))

        # _classify_intent raises internally but returns "consult" as fallback
        with patch("src.orchestrator._classify_intent", return_value="consult"):
            response, chunks = await orch.route(1005, "непонятный запрос")

        orch._beebot.answer.assert_called_once()
        assert response == "Fallback ответ"


# ---------------------------------------------------------------------------
# Dialog state management
# ---------------------------------------------------------------------------

class TestDialogState:
    """Tests for in-memory dialog state with TTL."""

    def _make_orchestrator(self):
        with (
            patch("src.orchestrator.Groq"),
            patch("src.orchestrator.BeebotAgent"),
            patch("src.orchestrator.LogistAgent"),
            patch("src.orchestrator.AnalystAgent"),
        ):
            return Orchestrator()

    @pytest.mark.asyncio
    async def test_stores_dialog_state_after_route(self):
        """Dialog state should be stored after routing."""
        orch = self._make_orchestrator()
        orch._beebot.answer = MagicMock(return_value=("ответ", []))

        with patch("src.orchestrator._classify_intent", return_value="consult"):
            await orch.route(2001, "вопрос")

        assert 2001 in orch._dialog_states
        assert orch._dialog_states[2001]["intent"] == "consult"

    @pytest.mark.asyncio
    async def test_get_intent_returns_stored_intent(self):
        """get_intent should return last intent for user."""
        orch = self._make_orchestrator()
        orch._logist.collect_shipping_info = AsyncMock(side_effect=NotImplementedError)

        with patch("src.orchestrator._classify_intent", return_value="order"):
            await orch.route(2002, "купить пергу")

        assert orch.get_intent(2002) == "order"

    def test_get_intent_returns_none_for_unknown_user(self):
        """get_intent should return None for a user with no dialog state."""
        orch = self._make_orchestrator()
        assert orch.get_intent(9999) is None

    def test_get_intent_returns_none_for_expired_state(self):
        """Expired dialog states should not be returned."""
        orch = self._make_orchestrator()
        # Inject a stale state manually
        orch._dialog_states[3001] = {
            "user_id": 3001,
            "query": "вопрос",
            "intent": "consult",
            "response": "ответ",
            "chunks": [],
            "updated_at": time.monotonic() - (_DIALOG_TTL_SECONDS + 1),
        }
        assert orch.get_intent(3001) is None

    @pytest.mark.asyncio
    async def test_evicts_stale_states_on_route(self):
        """Stale dialog states should be evicted when a new route is processed."""
        orch = self._make_orchestrator()
        orch._beebot.answer = MagicMock(return_value=("ответ", []))

        # Inject stale state for another user
        orch._dialog_states[8888] = {
            "user_id": 8888,
            "query": "старый вопрос",
            "intent": "consult",
            "response": "старый ответ",
            "chunks": [],
            "updated_at": time.monotonic() - (_DIALOG_TTL_SECONDS + 1),
        }

        with patch("src.orchestrator._classify_intent", return_value="consult"):
            await orch.route(2003, "новый вопрос")

        assert 8888 not in orch._dialog_states
        assert 2003 in orch._dialog_states


# ---------------------------------------------------------------------------
# Zero regression: BEEBOT still works as before
# ---------------------------------------------------------------------------

class TestZeroRegression:
    """Ensure BEEBOT consult path returns same data structure as before."""

    @pytest.mark.asyncio
    async def test_beebot_response_is_string(self):
        with (
            patch("src.orchestrator.Groq"),
            patch("src.orchestrator.BeebotAgent"),
            patch("src.orchestrator.LogistAgent"),
            patch("src.orchestrator.AnalystAgent"),
        ):
            orch = Orchestrator()

        orch._beebot.answer = MagicMock(
            return_value=("Прополис принимают по 20 капель.", [{"source": "pdf:Прополис", "score": 0.9}])
        )

        with patch("src.orchestrator._classify_intent", return_value="consult"):
            response, chunks = await orch.route(4001, "как принимать прополис")

        assert isinstance(response, str)
        assert isinstance(chunks, list)
        assert len(chunks) == 1
        assert chunks[0]["source"] == "pdf:Прополис"
