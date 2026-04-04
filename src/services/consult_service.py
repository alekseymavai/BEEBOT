"""ConsultService — бизнес-логика консультаций (KB + LLM).

Извлечено из src/agents/beebot.py. Агент становится тонкой обёрткой.
Не зависит от Telegram, FastAPI или Redis.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.knowledge_base import KnowledgeBase
    from src.llm_client import LLMClient
    from src.tunnel_monitor import TunnelMonitor

logger = logging.getLogger(__name__)

_TUNNEL_DOWN_HEADER = (
    "⚠️ Сейчас я работаю в автономном режиме (связь с ИИ временно недоступна). "
    "Вот что я нашёл по вашему вопросу в базе знаний:\n\n"
)

_TUNNEL_DOWN_FOOTER = (
    "\n\n_Для полноценного ответа с рекомендациями — попробуйте задать вопрос чуть позже, "
    "когда связь восстановится._"
)


class ConsultService:
    """Поиск по базе знаний + генерация ответа через LLM."""

    def __init__(
        self,
        kb: "KnowledgeBase",
        llm: "LLMClient",
        tunnel_monitor: "TunnelMonitor | None" = None,
    ) -> None:
        self.kb = kb
        self.llm = llm
        self.tunnel_monitor = tunnel_monitor

    def answer(
        self,
        query: str,
        history: list[dict] | None = None,
        style: str | None = None,
        memory_facts: list[str] | None = None,
        advice_text: str | None = None,
        user_name: str | None = None,
        system_prompt_override: str | None = None,
    ) -> tuple[str, list[dict]]:
        """Ответить на вопрос. Возвращает (ответ, список чанков).

        Если TunnelMonitor сигнализирует о недоступности туннеля —
        возвращает FAQ-ответ на основе поиска по базе знаний без LLM.
        """
        if self.tunnel_monitor is not None and not self.tunnel_monitor.is_healthy:
            return self.faq_fallback(query)

        chunks = self.kb.search(query)
        response = self.llm.generate(
            query, chunks,
            history=history,
            style=style,
            memory_facts=memory_facts,
            advice_text=advice_text,
            user_name=user_name,
            system_prompt_override=system_prompt_override,
        )
        return response, chunks

    def faq_fallback(self, query: str) -> tuple[str, list[dict]]:
        """Возвращает FAQ-ответ без LLM при недоступности туннеля."""
        chunks = self.kb.search(query)
        if not chunks:
            text = (
                _TUNNEL_DOWN_HEADER
                + "К сожалению, по вашему вопросу в базе знаний ничего не найдено.\n"
                "Попробуйте спросить о перге, прополисе, ПЖВМ или других продуктах пасеки."
                + _TUNNEL_DOWN_FOOTER
            )
            return text, []

        top_chunks = chunks[:3]
        excerpts = []
        for i, chunk in enumerate(top_chunks, 1):
            excerpt = chunk.get("text", "").strip()
            if excerpt:
                if len(excerpt) > 300:
                    excerpt = excerpt[:297] + "…"
                excerpts.append(f"{i}. {excerpt}")

        body = "\n\n".join(excerpts) if excerpts else "Информация найдена в базе знаний."
        text = _TUNNEL_DOWN_HEADER + body + _TUNNEL_DOWN_FOOTER
        return text, top_chunks
