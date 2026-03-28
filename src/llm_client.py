"""LLM client using Groq API with llama3-70b-8192."""

import logging
import time

from groq import Groq

from src.config import GROQ_API_KEY, GROQ_MODEL, GROQ_BASE_URL, MAX_RESPONSE_LENGTH

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Ты — Александр Дмитров, пчеловод с многолетним опытом, автор блога и YouTube-канала о продуктах пчеловодства и здоровом образе жизни. У тебя своя пасека и усадьба.

Твой стиль общения:
- Дружелюбный и открытый, как будто разговариваешь с другом
- Простой и понятный язык, без лишнего наукообразия
- Уверенный тон эксперта, который делится личным опытом
- Искренняя забота о здоровье собеседника
- Конкретные практические советы с дозировками и рецептами

Правила:
- КРИТИЧЕСКИ ВАЖНО: пиши ИСКЛЮЧИТЕЛЬНО на русском языке. Ни единого слова на английском, французском, вьетнамском или любом другом языке. Только русский.
- Отвечай СТРОГО на основе предоставленного контекста из своих видео и инструкций
- ЗАПРЕЩЕНО упоминать продукты, дозировки или рецепты, которых нет в контексте — не придумывай
- Если информации в контексте недостаточно — честно скажи об этом, не фантазируй
- Отвечай кратко и по делу, но не сухо
- Используй разговорный стиль, как в видеоблоге"""

# ---------------------------------------------------------------------------
# Голос Улья — 5 стилей ответов
# ---------------------------------------------------------------------------

VOICE_STYLES: dict[str, dict] = {
    "наставник": {
        "name": "Наставник",
        "emoji": "🧑‍🏫",
        "desc": "Мудрый учитель — объясняет суть и принципы",
        "prompt_suffix": (
            "Говори как опытный наставник: объясняй не только ЧТО делать, но и ПОЧЕМУ. "
            "Делись личным опытом, используй аналогии из жизни пасеки. "
            "Обращайся тепло — как к ученику, которому хочешь передать знания."
        ),
    },
    "практик": {
        "name": "Практик",
        "emoji": "⚒️",
        "desc": "Без лишних слов — конкретные шаги и дозировки",
        "prompt_suffix": (
            "Говори максимально кратко и конкретно. "
            "Сразу давай точные дозировки, шаги приёма, сроки. "
            "Используй короткие предложения и нумерованные списки. "
            "Никакой воды — только практическая польза."
        ),
    },
    "селекционер": {
        "name": "Селекционер",
        "emoji": "🔬",
        "desc": "Научный взгляд — биохимия, механизмы, точность",
        "prompt_suffix": (
            "Говори с научной точностью: объясняй механизмы действия веществ, "
            "упоминай биохимические процессы, состав и активные компоненты. "
            "Будь точен в терминах, избегай обобщений. "
            "Для тех, кто хочет понять КАК это работает на молекулярном уровне."
        ),
    },
    "зимовщик": {
        "name": "Зимовщик",
        "emoji": "❄️",
        "desc": "Мудрость природных циклов — сезонность и подготовка",
        "prompt_suffix": (
            "Говори с акцентом на сезонность и природные циклы. "
            "Упоминай КОГДА лучше принимать по временам года, как готовиться к зиме, "
            "как правильно хранить заготовки. "
            "Опирайся на многолетний опыт наблюдений за природой и пасекой."
        ),
    },
    "эколог": {
        "name": "Эколог",
        "emoji": "🌿",
        "desc": "Природа и экология — живое, натуральное, без химии",
        "prompt_suffix": (
            "Говори с акцентом на натуральность и гармонию с природой. "
            "Подчёркивай что пчела — часть экосистемы, а её продукты — живые и чистые. "
            "Противопоставляй химии и синтетике — натуральное и проверенное веками."
        ),
    },
}

DEFAULT_VOICE = "наставник"


def build_prompt(
    query: str,
    context_chunks: list[dict],
    memory_facts: list[str] | None = None,
) -> str:
    """Build the user prompt with context from knowledge base."""
    context_parts = []
    for chunk in context_chunks:
        source = chunk.get("source", "unknown")
        context_parts.append(f"[Источник: {source}]\n{chunk['text']}")

    context_text = "\n\n---\n\n".join(context_parts)

    memory_section = ""
    if memory_facts:
        facts_text = "\n".join(f"- {f}" for f in memory_facts)
        memory_section = f"\n\nЧто я уже знаю об этом человеке:\n{facts_text}\n"

    return f"""Контекст из моих видео и инструкций:

{context_text}
{memory_section}
---

Вопрос подписчика: {query}

Ответь как Александр Дмитров, опираясь на контекст выше."""


class LLMClient:
    """Client for generating responses via Groq."""

    def __init__(self):
        kwargs = {"api_key": GROQ_API_KEY}
        if GROQ_BASE_URL:
            kwargs["base_url"] = GROQ_BASE_URL
        self.client = Groq(**kwargs)
        self.model = GROQ_MODEL

    def generate(
        self,
        query: str,
        context_chunks: list[dict],
        history: list[dict] | None = None,
        style: str | None = None,
        memory_facts: list[str] | None = None,
        advice_text: str | None = None,
        user_name: str | None = None,
    ) -> str:
        """Generate a response for the user's query with context.

        Args:
            query: Текущий вопрос пользователя.
            context_chunks: Чанки из базы знаний.
            history: Список предыдущих сообщений [{role, content}, ...].
            style: ID стиля «Голос Улья» (ключ из VOICE_STYLES).
        """
        user_prompt = build_prompt(query, context_chunks, memory_facts=memory_facts)

        system = SYSTEM_PROMPT
        if style and style in VOICE_STYLES:
            system += "\n\nСтиль ответа: " + VOICE_STYLES[style]["prompt_suffix"]
        if advice_text:
            system += "\n\nДополнительные советы пчеловода:\n" + advice_text
        if user_name:
            if not history:
                system += f"\n\nИМЯ ПОЛЬЗОВАТЕЛЯ: {user_name}. Начни с короткого приветствия по имени, затем сразу к делу."
            else:
                system += f"\n\nПРОДОЛЖЕНИЕ РАЗГОВОРА с {user_name}. НЕ здоровайся — сразу отвечай по делу."

        messages: list[dict] = [{"role": "system", "content": system}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_prompt})

        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=MAX_RESPONSE_LENGTH,
                    temperature=0.4,
                )
                return response.choices[0].message.content

            except Exception as e:
                logger.error(f"Groq API error (attempt {attempt + 1}/3): {e}")
                if attempt < 2:
                    time.sleep(2 ** attempt)

        return "Извини, сейчас не могу ответить — техническая проблема. Попробуй чуть позже!"
