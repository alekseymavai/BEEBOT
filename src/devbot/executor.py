"""Исполнитель задач — запускает claude CLI со стримингом.

Паттерн из hive-mind: stream-json + auto-continue + feedback loop.
"""

import asyncio
import json
import logging
import re
from collections.abc import AsyncIterator, Callable, Awaitable

from src.devbot.config import BEEBOT_DIR
from src.devbot.prompts import build_system_prompt, build_user_prompt

logger = logging.getLogger(__name__)

PROGRESS_INTERVAL = 30  # секунд между апдейтами в Telegram
MAX_AUTO_CONTINUE = 3   # максимум автопродолжений при лимите токенов


async def execute(
    task: str,
    plan: str,
    dev_memory: str = "",
    advice_context: str = "",
    feedback: str | None = None,
    session_id: str | None = None,
    progress_cb: Callable[[str], Awaitable[None]] | None = None,
) -> dict:
    """Запустить claude CLI для выполнения задачи.

    Args:
        task: Текст задачи
        plan: Согласованный план изменений
        dev_memory: Контекст из памяти разработчика
        advice_context: Советы пчеловода
        feedback: Уточнение от Александра (feedback loop)
        session_id: ID сессии для --resume (auto-continue)
        progress_cb: Callback для отправки прогресса в Telegram

    Returns:
        {"events": [...], "session_id": str, "exit_code": int, "result_text": str}
    """
    user_prompt = build_user_prompt(task, plan, dev_memory, advice_context, feedback)
    system_rules = build_system_prompt()

    args = [
        "/home/new/.local/bin/claude",
        "--output-format", "stream-json",
        "--verbose",
        "--model", "claude-sonnet-4-6",
        "-p", user_prompt,
        "--append-system-prompt", system_rules,
    ]
    if session_id:
        args += ["--resume", session_id]

    logger.info("DEVBOT executor: запуск claude CLI (session=%s, feedback=%s)", session_id, bool(feedback))

    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(BEEBOT_DIR),
    )

    result_lines: list[dict] = []
    new_session_id: str | None = None
    last_progress_text = ""
    result_text_parts: list[str] = []
    auto_continue_count = 0

    _loop = asyncio.get_running_loop()
    last_progress_time = _loop.time()

    async def _send_progress(text: str) -> None:
        nonlocal last_progress_time, last_progress_text
        now = asyncio.get_running_loop().time()
        if progress_cb and (now - last_progress_time >= PROGRESS_INTERVAL) and text != last_progress_text:
            try:
                snippet = text[-200:].strip() if len(text) > 200 else text.strip()
                await progress_cb(f"⚙️ Выполняется...\n```\n{snippet}\n```")
                last_progress_time = now
                last_progress_text = text
            except Exception as e:
                logger.warning("progress_cb error: %s", e)

    try:
        async for raw_line in proc.stdout:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                event_type = event.get("type", "") if isinstance(event, dict) else ""

                if event_type == "session_id":
                    new_session_id = event.get("session_id")
                elif event_type == "text":
                    text = event.get("text", "")
                    result_text_parts.append(text)
                    await _send_progress("".join(result_text_parts))
                elif event_type == "end_turn":
                    stop_reason = event.get("stop_reason", "")
                    if stop_reason == "max_tokens" and auto_continue_count < MAX_AUTO_CONTINUE:
                        auto_continue_count += 1
                        logger.info("DEVBOT: auto-continue %d/%d", auto_continue_count, MAX_AUTO_CONTINUE)
                        if progress_cb:
                            await progress_cb(f"⏳ Продолжаю (авто {auto_continue_count}/{MAX_AUTO_CONTINUE})...")

                if isinstance(event, dict):
                    result_lines.append(event)
            except json.JSONDecodeError:
                # Не-JSON строка (stderr, диагностика) — пишем в лог
                if line:
                    logger.debug("claude CLI raw: %s", line[:200])
            except Exception as e:
                # Любая другая ошибка при обработке строки — логируем и продолжаем
                logger.warning("Executor: ошибка обработки строки (пропускаем): %s | %s", e, line[:80])
    except Exception as stream_err:
        # Ошибка самого стрима — логируем, процесс всё равно завершим
        logger.warning("Executor: ошибка чтения stdout: %s", stream_err)
    finally:
        await proc.wait()

    exit_code = proc.returncode

    # Извлечь PR-ссылку из результата если есть
    result_text = "".join(result_text_parts)
    pr_url = _extract_pr_url(result_text)
    sha = _extract_sha(result_text)

    logger.info(
        "DEVBOT executor: завершён (exit=%d, session=%s, pr=%s, sha=%s)",
        exit_code, new_session_id, pr_url, sha,
    )

    return {
        "events": result_lines,
        "session_id": new_session_id,
        "exit_code": exit_code,
        "result_text": result_text,
        "pr_url": pr_url,
        "sha": sha,
    }


def _extract_pr_url(text: str) -> str | None:
    """Найти URL GitHub PR в тексте результата."""
    m = re.search(r'https://github\.com/[^/]+/[^/]+/pull/\d+', text)
    return m.group(0) if m else None


def _extract_sha(text: str) -> str | None:
    """Найти SHA коммита (7-40 hex символов после 'commit' или SHA:)."""
    m = re.search(r'\b([0-9a-f]{7,40})\b', text)
    return m.group(1) if m else None
