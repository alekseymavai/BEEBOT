"""Анализатор задачи — через polza.ai (OpenAI-compatible прокси → Claude)."""

import logging
from openai import AsyncOpenAI

from src.devbot.config import POLZA_API_KEY, POLZA_BASE_URL, POLZA_MODEL, POLZA_REFERER
from src.devbot.prompts import build_analyzer_prompt

logger = logging.getLogger(__name__)

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            api_key=POLZA_API_KEY,
            base_url=POLZA_BASE_URL,
            default_headers={
                "HTTP-Referer": POLZA_REFERER,
                "X-Title": "BEEBOT DEVBOT Analyzer",
            },
        )
    return _client


async def analyze_task(
    task: str,
    dev_memory: str = "",
    advice_context: str = "",
) -> str:
    """Проанализировать задачу → вернуть план изменений.

    Args:
        task: Текст задачи от Александра
        dev_memory: Предыдущие решения из памяти разработчика
        advice_context: Советы пчеловода (категория crm/процесс)

    Returns:
        Текст плана в markdown-формате
    """
    prompt = build_analyzer_prompt(task, dev_memory, advice_context)
    logger.info("DEVBOT analyzer: анализ задачи (len=%d)", len(task))

    try:
        client = _get_client()
        resp = await client.chat.completions.create(
            model=POLZA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=800,
        )
        plan = resp.choices[0].message.content or ""
        logger.info("DEVBOT analyzer: план готов (len=%d)", len(plan))
        return plan
    except Exception as e:
        logger.error("DEVBOT analyzer error: %s", e)
        raise
