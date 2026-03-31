"""Unit tests for src/agent_bus.py — AgentBus client."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent_bus import AgentBusClient, BEEBOT_AGENT_ID, create_agent_bus_client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_client(url="http://localhost:8081", kb=None, crm=None) -> AgentBusClient:
    return AgentBusClient(url, kb=kb, crm=crm)


def _make_kb(results=None):
    kb = MagicMock()
    kb.search.return_value = results or [
        {"text": "Прополис принимают по 20 капель.", "source": "pdf:Прополис", "score": 0.9},
    ]
    return kb


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

class TestRegister:
    @pytest.mark.asyncio
    async def test_register_success(self):
        client = _make_client()
        with patch("httpx.AsyncClient") as MockClient:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"ok": True, "agent": {"agentId": BEEBOT_AGENT_ID}}
            MockClient.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_resp)

            result = await client.register()

        assert result is True
        assert client._registered is True

    @pytest.mark.asyncio
    async def test_register_failure_returns_false(self):
        client = _make_client()
        with patch("httpx.AsyncClient") as MockClient:
            mock_resp = MagicMock()
            mock_resp.status_code = 500
            mock_resp.json.return_value = {"error": "Server error"}
            mock_resp.text = "Server error"
            MockClient.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_resp)

            result = await client.register()

        assert result is False
        assert client._registered is False

    @pytest.mark.asyncio
    async def test_register_network_error_returns_false(self):
        client = _make_client()
        with patch("httpx.AsyncClient") as MockClient:
            MockClient.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=Exception("Connection refused")
            )
            result = await client.register()

        assert result is False

    @pytest.mark.asyncio
    async def test_register_sends_correct_agent_id(self):
        client = _make_client()
        posted_data = {}

        async def fake_post(url, json=None):
            posted_data.update(json or {})
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = {"ok": True}
            return resp

        with patch("httpx.AsyncClient") as MockClient:
            MockClient.return_value.__aenter__.return_value.post = AsyncMock(side_effect=fake_post)
            await client.register()

        assert posted_data.get("agentId") == BEEBOT_AGENT_ID
        assert "capabilities" in posted_data


# ---------------------------------------------------------------------------
# Heartbeat
# ---------------------------------------------------------------------------

class TestHeartbeat:
    @pytest.mark.asyncio
    async def test_heartbeat_success(self):
        client = _make_client()
        with patch("httpx.AsyncClient") as MockClient:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            MockClient.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_resp)

            result = await client.heartbeat()

        assert result is True

    @pytest.mark.asyncio
    async def test_heartbeat_failure_returns_false(self):
        client = _make_client()
        with patch("httpx.AsyncClient") as MockClient:
            MockClient.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=Exception("timeout")
            )
            result = await client.heartbeat()

        assert result is False


# ---------------------------------------------------------------------------
# Tool: kb_search
# ---------------------------------------------------------------------------

class TestToolKbSearch:
    @pytest.mark.asyncio
    async def test_kb_search_returns_results(self):
        kb = _make_kb()
        client = _make_client(kb=kb)

        result = await client._tool_kb_search({"query": "как принимать прополис"})

        assert result["ok"] is True
        assert "results" in result
        assert len(result["results"]) > 0
        assert "text" in result["results"][0]
        assert "source" in result["results"][0]
        kb.search.assert_called_once_with("как принимать прополис", top_k=5)

    @pytest.mark.asyncio
    async def test_kb_search_error_when_no_query(self):
        client = _make_client(kb=_make_kb())

        result = await client._tool_kb_search({})

        assert "error" in result

    @pytest.mark.asyncio
    async def test_kb_search_error_when_no_kb(self):
        client = _make_client()  # no kb

        result = await client._tool_kb_search({"query": "прополис"})

        assert "error" in result

    @pytest.mark.asyncio
    async def test_kb_search_respects_top_k(self):
        kb = _make_kb()
        client = _make_client(kb=kb)

        await client._tool_kb_search({"query": "прополис", "top_k": 3})

        kb.search.assert_called_once_with("прополис", top_k=3)

    @pytest.mark.asyncio
    async def test_kb_search_truncates_long_chunks(self):
        kb = MagicMock()
        kb.search.return_value = [
            {"text": "А" * 1000, "source": "pdf:test", "score": 0.8},
        ]
        client = _make_client(kb=kb)

        result = await client._tool_kb_search({"query": "тест"})

        assert len(result["results"][0]["text"]) <= 500


# ---------------------------------------------------------------------------
# Tool: order_status
# ---------------------------------------------------------------------------

class TestToolOrderStatus:
    @pytest.mark.asyncio
    async def test_order_status_found(self):
        order = MagicMock()
        order.id = 42
        order.number = "БЦ-1234"
        order.status = "Отправлен"
        order.client_name = "Иван Иванов"
        order.total = 1500

        crm = MagicMock()
        crm.get_orders = AsyncMock(return_value=[order])
        client = _make_client(crm=crm)

        result = await client._tool_order_status({"order_number": "БЦ-1234"})

        assert result["ok"] is True
        assert result["order"]["status"] == "Отправлен"

    @pytest.mark.asyncio
    async def test_order_status_not_found(self):
        crm = MagicMock()
        crm.get_orders = AsyncMock(return_value=[])
        client = _make_client(crm=crm)

        result = await client._tool_order_status({"order_number": "НЕТ-999"})

        assert result["ok"] is False
        assert "не найден" in result["message"]

    @pytest.mark.asyncio
    async def test_order_status_no_number_returns_error(self):
        client = _make_client(crm=MagicMock())

        result = await client._tool_order_status({})

        assert "error" in result

    @pytest.mark.asyncio
    async def test_order_status_no_crm_returns_error(self):
        client = _make_client()

        result = await client._tool_order_status({"order_number": "БЦ-1234"})

        assert "error" in result


# ---------------------------------------------------------------------------
# Message dispatch
# ---------------------------------------------------------------------------

class TestHandleMessage:
    @pytest.mark.asyncio
    async def test_dispatch_kb_search(self):
        kb = _make_kb()
        client = _make_client(kb=kb)
        client.respond = AsyncMock()

        msg = {
            "id": "corr-123",
            "correlationId": "corr-123",
            "from": "dronedoc",
            "payload": {"tool": "kb_search", "query": "прополис"},
        }
        await client._handle_message(msg)

        client.respond.assert_called_once()
        args = client.respond.call_args[0]
        assert args[0] == "corr-123"
        assert args[1]["ok"] is True

    @pytest.mark.asyncio
    async def test_dispatch_order_status(self):
        order = MagicMock()
        order.id = 1
        order.number = "Т-100"
        order.status = "Новый"
        order.client_name = "Тест"
        order.total = 500

        crm = MagicMock()
        crm.get_orders = AsyncMock(return_value=[order])
        client = _make_client(crm=crm)
        client.respond = AsyncMock()

        msg = {
            "correlationId": "corr-456",
            "from": "test",
            "payload": {"tool": "order_status", "order_number": "Т-100"},
        }
        await client._handle_message(msg)

        client.respond.assert_called_once()

    @pytest.mark.asyncio
    async def test_dispatch_unknown_tool_responds_error(self):
        client = _make_client()
        client.respond = AsyncMock()

        msg = {
            "correlationId": "corr-789",
            "from": "test",
            "payload": {"tool": "nonexistent_tool"},
        }
        await client._handle_message(msg)

        client.respond.assert_called_once()
        result = client.respond.call_args[0][1]
        assert "error" in result

    @pytest.mark.asyncio
    async def test_no_respond_when_no_correlation_id(self):
        client = _make_client()
        client.respond = AsyncMock()

        msg = {"from": "test", "payload": {"tool": "kb_search", "query": "тест"}}
        await client._handle_message(msg)

        client.respond.assert_not_called()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class TestCreateAgentBusClient:
    def test_returns_none_when_url_not_configured(self):
        with patch("src.agent_bus.AGENT_BUS_URL", None):
            result = create_agent_bus_client()
        assert result is None

    def test_returns_client_when_url_configured(self):
        with patch("src.agent_bus.AGENT_BUS_URL", "http://localhost:8081"):
            result = create_agent_bus_client()
        assert isinstance(result, AgentBusClient)
        assert result._url == "http://localhost:8081"

    def test_strips_trailing_slash_from_url(self):
        with patch("src.agent_bus.AGENT_BUS_URL", "http://localhost:8081/"):
            result = create_agent_bus_client()
        assert result._url == "http://localhost:8081"
