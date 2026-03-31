"""Агент-консультант BEEBOT — отвечает на вопросы о продуктах пчеловодства.

Содержит логику поиска по базе знаний и генерации ответов через LLM.
Извлечено из src/bot.py для подготовки к мультиагентной архитектуре.
"""

from collections import Counter
from typing import TYPE_CHECKING

from src.knowledge_base import KnowledgeBase
from src.llm_client import LLMClient

if TYPE_CHECKING:
    from src.tunnel_monitor import TunnelMonitor

# (kb_source_stem, display_name, pdf_filename, category)
INSTRUCTIONS = [
    # Продукты пчеловодства
    ("Перга",                                        "Перга",                        "Перга.pdf",                                        "bee"),
    ("Пчелиная обножка",                             "Обножка (пыльца)",             "Пчелиная обножка.pdf",                             "bee"),
    ("Трутнёвый гомогенат",                          "Трутнёвый гомогенат",          "Трутнёвый гомогенат.pdf",                          "bee"),
    # Настойки
    ("Прополис_ сухой + настойка",                   "Прополис",                     "Прополис_ сухой + настойка.pdf",                   "tincture"),
    ("Настойка ПЖВМ",                                "ПЖВМ (огнёвка)",               "Настойка ПЖВМ.pdf",                                "tincture"),
    ("Настойка Подмора пчелиного (на самогоне 40°)", "Подмор пчелиный",              "Настойка Подмора пчелиного (на самогоне 40°).pdf", "tincture"),
    ("Настойка «Успокоин» (Травяная)",               "Успокоин",                     "Настойка «Успокоин» (Травяная).pdf",               "tincture"),
    ("Антивирус",                                    "Антивирус",                    "Антивирус.pdf",                                    "tincture"),
    ("ФитоЭнергия",                                  "ФитоЭнергия",                  "ФитоЭнергия.pdf",                                  "tincture"),
    ("Настойка для ЖКТ",                             "Настойка для ЖКТ",             "Настойка для ЖКТ.pdf",                             "tincture"),
    # Программы здоровья
    ("«УНИВЕРСАЛЬНАЯ_ПРОГРАММА_ОЗДОРОВЛЕНИЯ»",       "Программа оздоровления (УПО)", "«УНИВЕРСАЛЬНАЯ_ПРОГРАММА_ОЗДОРОВЛЕНИЯ».pdf",       "program"),
    ("Приложение к УПО (1)",                         "Приложение к УПО",             "Приложение к УПО (1).pdf",                         "program"),
    ("Иммунитет ребенка",                            "Иммунитет ребёнка",            "Иммунитет ребенка.pdf",                            "program"),
    ("Инструкция ТГ",                                "Инструкция ТГ",                "Инструкция ТГ.pdf",                                "program"),
]

STEM_TO_INSTRUCTION = {
    stem: (i, name, fname)
    for i, (stem, name, fname, _cat) in enumerate(INSTRUCTIONS)
}

CATEGORY_LABELS = {
    "bee":      "🍯 Продукты пчеловодства",
    "tincture": "🌿 Настойки",
    "program":  "📋 Программы здоровья",
}

_PRODUCTS_TRIGGERS = {
    "какие продукты", "список продуктов", "что есть", "что у тебя есть",
    "какие настойки", "какие товары", "что продаёшь", "что продаешь",
    "ассортимент", "что можно купить", "какие препараты",
}

# --- FAQ fallback при недоступности туннеля ---

_FAQ_QUERIES = [
    "как принимать пергу",
    "прополис для иммунитета",
    "трутнёвый гомогенат применение",
    "ПЖВМ огнёвка дозировка",
    "подмор пчелиный настойка",
]

_TUNNEL_DOWN_HEADER = (
    "⚠️ Сейчас я работаю в автономном режиме (связь с ИИ временно недоступна). "
    "Вот что я нашёл по вашему вопросу в базе знаний:\n\n"
)

_TUNNEL_DOWN_FOOTER = (
    "\n\n_Для полноценного ответа с рекомендациями — попробуйте задать вопрос чуть позже, "
    "когда связь восстановится._"
)


def is_products_query(query: str) -> bool:
    """Возвращает True, если запрос — запрос о каталоге продуктов."""
    query_lower = query.lower()
    return any(trigger in query_lower for trigger in _PRODUCTS_TRIGGERS)


def get_top_instruction(chunks: list[dict]) -> tuple[int, str, str] | None:
    """Возвращает (idx, name, filename) наиболее релевантной инструкции из чанков."""
    stems = [
        chunk["source"][4:]
        for chunk in chunks
        if chunk.get("source", "").startswith("pdf:")
        and chunk["source"][4:] in STEM_TO_INSTRUCTION
    ]
    if not stems:
        return None
    top_stem = Counter(stems).most_common(1)[0][0]
    return STEM_TO_INSTRUCTION[top_stem]


class BeebotAgent:
    """Агент-консультант: ищет в базе знаний и генерирует ответы."""

    def __init__(self):
        self.kb = KnowledgeBase()
        self.llm = LLMClient()
        self.tunnel_monitor: "TunnelMonitor | None" = None

    def _faq_fallback(self, query: str) -> tuple[str, list[dict]]:
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

        # Берём до 3 наиболее релевантных чанков
        top_chunks = chunks[:3]
        excerpts = []
        for i, chunk in enumerate(top_chunks, 1):
            excerpt = chunk.get("text", "").strip()
            if excerpt:
                # Ограничиваем каждый чанк 300 символами для компактности
                if len(excerpt) > 300:
                    excerpt = excerpt[:297] + "…"
                excerpts.append(f"{i}. {excerpt}")

        body = "\n\n".join(excerpts) if excerpts else "Информация найдена в базе знаний."
        text = _TUNNEL_DOWN_HEADER + body + _TUNNEL_DOWN_FOOTER
        return text, top_chunks

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
            return self._faq_fallback(query)

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
