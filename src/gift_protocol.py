"""Gift Protocol — архитектурный протокол обмена ценностями между агентами.

Gift (A3 Gift Ontology) — типизированный контейнер для передачи:
  - намерения (telos)
  - содержания (content)
  - контекста (context: SharedContext + anamnesis)
  - свободы агента (freedom: ACCEPTED | DEFERRED | DECLINED)

GiftBroker — надстройка над Orchestrator:
  1. Обогащает запрос контекстом из SharedContextStore + AnamnesisCache
  2. Делегирует маршрутизацию в Orchestrator (LangGraph остаётся внутри)
  3. Обновляет SharedContext историей диалога после ответа
  4. Логирует каждый Gift для аудита

Принципы (из dronedoc2026):
  - Минимально необходимый доступ: CrmAgent — единственный кто видит CRM
  - Прозрачность: каждый Gift логируется с намерением и результатом
  - Свобода агента (A5): DEFERRED — корректный ответ, не ошибка
"""
from __future__ import annotations

import logging
import time
from typing import Literal, TYPE_CHECKING

from typing_extensions import TypedDict

if TYPE_CHECKING:
    from src.orchestrator import Orchestrator
    from src.shared_context import SharedContextStore
    from src.anamnesis import AnamnesisCache
    from src.crm_agent import CrmAgent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Gift TypedDict
# ---------------------------------------------------------------------------

Freedom = Literal["ACCEPTED", "DEFERRED", "DECLINED"]


class Gift(TypedDict, total=False):
    """Единица обмена ценностями между агентами (A3 Gift Ontology)."""
    giver: str           # источник: "user:<id>" | "admin" | "worker:<id>"
    receiver: str        # получатель: "beebot" | "logist" | "analyst" | "system"
    content: dict        # содержание дара (query / response / chunks)
    context: dict        # обогащённый контекст из SharedContext
    telos: str           # зачем этот дар (intent)
    anamnesis: list[dict]  # прошлые значимые взаимодействия (A3)
    freedom: Freedom     # ACCEPTED | DEFERRED | DECLINED
    timestamp: float     # unix-время создания


# ---------------------------------------------------------------------------
# GiftBroker
# ---------------------------------------------------------------------------

_INTENT_TO_AGENT: dict[str, str] = {
    "consult": "beebot",
    "order": "logist",
    "edit": "logist",
    "track": "logist",
    "stats": "analyst",
    "greeting": "beebot",
}


class GiftBroker:
    """Знает SharedContext. Матчит потребности. Доставляет дары.

    Надстройка над Orchestrator — не замена. Orchestrator остаётся
    внутренним механизмом маршрутизации через LangGraph.
    """

    def __init__(
        self,
        orchestrator: "Orchestrator",
        context_store: "SharedContextStore",
        anamnesis: "AnamnesisCache",
        crm_agent: "CrmAgent | None" = None,
    ) -> None:
        self._orchestrator = orchestrator
        self._ctx = context_store
        self._anamnesis = anamnesis
        self._crm = crm_agent

    async def send(
        self,
        user_id: int,
        query: str,
        *,
        style: str | None = None,
        user_name: str | None = None,
    ) -> tuple[str, list[dict]]:
        """Отправить Gift пользователя в систему агентов.

        Обогащает запрос контекстом из SharedContextStore + AnamnesisCache,
        делегирует Orchestrator, обновляет SharedContext после ответа.

        Returns:
            (response_text, chunks) — идентично Orchestrator.route().
        """
        user_ctx = self._ctx.get(user_id)

        # Собрать анамнез (прошлые взаимодействия пользователя)
        anamnesis = await self._anamnesis.get(user_id, self._crm)

        gift: Gift = {
            "giver": f"user:{user_id}",
            "receiver": "beebot",  # уточнится после classify
            "content": {"query": query},
            "context": {
                "user_name": user_name,
                "style": style,
                "history_len": len(user_ctx.dialog_history),
            },
            "telos": "unknown",
            "anamnesis": anamnesis,
            "freedom": "ACCEPTED",
            "timestamp": time.monotonic(),
        }

        logger.debug(
            "Gift send: giver=%s query=%.60s anamnesis=%d",
            gift["giver"], query, len(anamnesis),
        )

        # Делегируем в Orchestrator (LangGraph внутри)
        response, chunks = await self._orchestrator.route(
            user_id, query, style=style, user_name=user_name,
        )

        # Обновить SharedContext историей диалога (только если есть ответ)
        if response:
            user_ctx.append_history(query, response)

        # Определить intent и получателя
        intent = self._orchestrator.get_intent(user_id) or "consult"
        gift["telos"] = intent
        gift["receiver"] = _INTENT_TO_AGENT.get(intent, "beebot")
        gift["content"]["response"] = response
        gift["content"]["chunks"] = chunks

        logger.debug(
            "Gift delivered: intent=%s receiver=%s response_len=%d chunks=%d",
            gift["telos"], gift["receiver"], len(response), len(chunks),
        )

        return response, chunks

    def get_intent(self, user_id: int) -> str | None:
        """Делегировать get_intent в оркестратор."""
        return self._orchestrator.get_intent(user_id)

    def suggest_interface(
        self,
        user_id: int,
        is_worker: bool = False,
    ) -> str:
        """Context-aware выбор интерфейса для /start.

        Проверяет SharedContext: если работник явно переключился в режим
        покупателя (interface_mode="client"), возвращает "client".

        Returns:
            "worker" | "client"
        """
        if not is_worker:
            return "client"
        ctx = self._ctx.get(user_id)
        if ctx.interface_mode == "client":
            return "client"
        return "worker"

    def set_interface_mode(self, user_id: int, mode: str) -> None:
        """Установить режим интерфейса пользователя ("default" | "client")."""
        ctx = self._ctx.get(user_id)
        ctx.interface_mode = mode
        ctx.touch()

    async def defer(self, user_id: int, reason: str) -> None:
        """Отложить Gift (агент занят или данные временно недоступны)."""
        logger.info("Gift DEFERRED: user=%d reason=%s", user_id, reason)
