"""AgentBus клиент — регистрация BEEBOT в dronedoc2026 AgentBus.

Протокол AgentBus (dronedoc2026, порт 8081):
  POST /api/agent-bus/register   — регистрация агента
  POST /api/agent-bus/heartbeat  — heartbeat каждые 30 сек
  GET  /api/agent-bus/inbox/:id  — получить входящие сообщения
  POST /api/agent-bus/respond    — ответить на запрос

Экспортируемые инструменты BEEBOT:
  kb_search    — семантический поиск по базе знаний пасеки
  order_status — статус заказа по номеру (через CRM)
  ask          — задать вопрос консультанту (полный LLM-ответ)

Работает с graceful fallback: если AgentBus недоступен — тихо отключается.
Активируется через AGENT_BUS_URL в .env.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional, TYPE_CHECKING

import httpx

from src.config import AGENT_BUS_URL

if TYPE_CHECKING:
    from src.knowledge_base import KnowledgeBase
    from src.integram_client import IntegramClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------

BEEBOT_AGENT_ID = "beebot"
_HEARTBEAT_INTERVAL = 30   # секунды
_INBOX_POLL_INTERVAL = 10  # секунды

# Спецификация агента (Agent Card)
_BEEBOT_CAPABILITIES = ["kb_search", "order_status", "ask"]

_BEEBOT_REGISTRATION = {
    "agentId": BEEBOT_AGENT_ID,
    "name": "BEEBOT — Консультант пасеки",
    "type": "assistant",
    "capabilities": _BEEBOT_CAPABILITIES,
    "transport": "rest",
    "meta": {
        "description": "AI-помощник Александра Дмитрова: продукты пчеловодства, база знаний, заказы",
        "tools": {
            "kb_search": "Семантический поиск по базе знаний пасеки (PDF + YouTube + Q&A)",
            "order_status": "Статус заказа по номеру (например, 'БЦ-1234')",
            "ask": "Задать вопрос консультанту и получить полный LLM-ответ",
        },
    },
}


# ---------------------------------------------------------------------------
# AgentBusClient
# ---------------------------------------------------------------------------

class AgentBusClient:
    """Клиент AgentBus для BEEBOT.

    Регистрирует агента, отправляет heartbeat и обрабатывает входящие запросы
    от других агентов dronedoc2026.
    """

    def __init__(
        self,
        bus_url: str,
        kb: Optional["KnowledgeBase"] = None,
        crm: Optional["IntegramClient"] = None,
    ) -> None:
        self._url = bus_url.rstrip("/")
        self._kb = kb
        self._crm = crm
        self._registered = False
        self._running = False

    # ------------------------------------------------------------------
    # Registration & Heartbeat
    # ------------------------------------------------------------------

    async def register(self) -> bool:
        """Зарегистрировать BEEBOT в AgentBus. Возвращает True при успехе."""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.post(
                    f"{self._url}/api/agent-bus/register",
                    json=_BEEBOT_REGISTRATION,
                )
                if r.status_code == 200 and r.json().get("ok"):
                    self._registered = True
                    logger.info(
                        "AgentBus: BEEBOT зарегистрирован (%s, возможности: %s)",
                        self._url, ", ".join(_BEEBOT_CAPABILITIES),
                    )
                    return True
                logger.warning("AgentBus: регистрация не удалась: %s", r.text[:200])
                return False
        except Exception as e:
            logger.warning("AgentBus: не удалось зарегистрироваться в %s: %s", self._url, e)
            return False

    async def heartbeat(self) -> bool:
        """Отправить heartbeat. Возвращает True при успехе."""
        try:
            async with httpx.AsyncClient(timeout=3) as client:
                r = await client.post(
                    f"{self._url}/api/agent-bus/heartbeat",
                    json={"agentId": BEEBOT_AGENT_ID},
                )
                return r.status_code == 200
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Inbox polling
    # ------------------------------------------------------------------

    async def poll_inbox(self) -> list[dict]:
        """Получить и пометить прочитанными входящие сообщения."""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(
                    f"{self._url}/api/agent-bus/inbox/{BEEBOT_AGENT_ID}",
                    params={"markRead": "true", "limit": "20"},
                )
                if r.status_code == 200:
                    return r.json().get("messages", [])
        except Exception as _e:
            logger.debug("AgentBus: ошибка чтения inbox: %s", _e)
        return []

    async def respond(self, correlation_id: str, payload: dict) -> None:
        """Ответить на запрос через AgentBus."""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                await client.post(
                    f"{self._url}/api/agent-bus/respond",
                    json={
                        "from": BEEBOT_AGENT_ID,
                        "correlationId": correlation_id,
                        "payload": payload,
                    },
                )
        except Exception as _e:
            logger.warning("AgentBus: ошибка отправки ответа: %s", _e)

    # ------------------------------------------------------------------
    # Tool dispatch
    # ------------------------------------------------------------------

    async def _handle_message(self, msg: dict) -> None:
        """Обработать входящее сообщение и ответить если нужно."""
        payload = msg.get("payload", {})
        tool = payload.get("tool") or payload.get("type")
        correlation_id = msg.get("correlationId") or msg.get("id")

        logger.debug(
            "AgentBus: входящее сообщение tool=%s from=%s",
            tool, msg.get("from", "?"),
        )

        result: dict[str, Any]
        if tool == "kb_search":
            result = await self._tool_kb_search(payload)
        elif tool == "order_status":
            result = await self._tool_order_status(payload)
        elif tool == "ask":
            result = await self._tool_ask(payload)
        else:
            logger.debug("AgentBus: неизвестный tool=%s", tool)
            result = {"error": f"Неизвестный инструмент: {tool!r}"}

        if correlation_id:
            await self.respond(correlation_id, result)

    async def _tool_kb_search(self, payload: dict) -> dict:
        """kb_search: семантический поиск по базе знаний."""
        query = payload.get("query", "")
        if not query:
            return {"error": "query обязателен"}
        if not self._kb:
            return {"error": "База знаний недоступна"}
        try:
            chunks = self._kb.search(query, top_k=payload.get("top_k", 5))
            return {
                "ok": True,
                "query": query,
                "results": [
                    {"text": c["text"][:500], "source": c["source"], "score": round(c["score"], 3)}
                    for c in chunks
                ],
            }
        except Exception as e:
            return {"error": str(e)}

    async def _tool_order_status(self, payload: dict) -> dict:
        """order_status: статус заказа по номеру."""
        order_number = payload.get("order_number") or payload.get("number")
        if not order_number:
            return {"error": "order_number обязателен"}
        if not self._crm:
            return {"error": "CRM недоступна"}
        try:
            orders = await self._crm.get_orders()
            order = next((o for o in orders if o.number == str(order_number)), None)
            if not order:
                return {"ok": False, "message": f"Заказ {order_number!r} не найден"}
            return {
                "ok": True,
                "order": {
                    "id": order.id,
                    "number": order.number,
                    "status": order.status,
                    "client": order.client_name,
                    "total": order.total,
                },
            }
        except Exception as e:
            return {"error": str(e)}

    async def _tool_ask(self, payload: dict) -> dict:
        """ask: задать вопрос консультанту (только KB-поиск, без LLM)."""
        query = payload.get("query", "")
        if not query:
            return {"error": "query обязателен"}
        # В безголовом режиме возвращаем топ-3 чанка без LLM
        if not self._kb:
            return {"error": "База знаний недоступна"}
        try:
            chunks = self._kb.search(query, top_k=3)
            context = "\n---\n".join(c["text"] for c in chunks)
            return {"ok": True, "query": query, "context": context[:2000]}
        except Exception as e:
            return {"error": str(e)}

    # ------------------------------------------------------------------
    # Background tasks
    # ------------------------------------------------------------------

    async def run_heartbeat(self) -> None:
        """Фоновая задача: heartbeat каждые 30 сек."""
        while self._running:
            await asyncio.sleep(_HEARTBEAT_INTERVAL)
            if not await self.heartbeat():
                # Потеря соединения — попробовать перерегистрироваться
                logger.info("AgentBus: heartbeat failed, пробую перерегистрацию...")
                self._registered = await self.register()

    async def run_inbox(self) -> None:
        """Фоновая задача: опрос inbox каждые 10 сек."""
        while self._running:
            await asyncio.sleep(_INBOX_POLL_INTERVAL)
            if not self._registered:
                continue
            messages = await self.poll_inbox()
            for msg in messages:
                try:
                    await self._handle_message(msg)
                except Exception as e:
                    logger.warning("AgentBus: ошибка обработки сообщения: %s", e)

    async def start(self, kb: Optional["KnowledgeBase"] = None, crm: Optional["IntegramClient"] = None) -> None:
        """Запустить фоновые задачи: регистрация + heartbeat + inbox.

        Graceful: если регистрация не удалась — задачи не запускаются.
        """
        if kb:
            self._kb = kb
        if crm:
            self._crm = crm

        registered = await self.register()
        if not registered:
            logger.info("AgentBus: регистрация не удалась — работаем без AgentBus.")
            return

        self._running = True
        asyncio.create_task(self.run_heartbeat())
        asyncio.create_task(self.run_inbox())
        logger.info("AgentBus: клиент запущен (heartbeat=%ds, inbox=%ds)", _HEARTBEAT_INTERVAL, _INBOX_POLL_INTERVAL)

    def stop(self) -> None:
        self._running = False


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_agent_bus_client() -> Optional[AgentBusClient]:
    """Создать клиент AgentBus если AGENT_BUS_URL настроен."""
    if not AGENT_BUS_URL:
        return None
    return AgentBusClient(AGENT_BUS_URL)
