"""DEVBOT — Telegram-бот автономного разработчика BEEBOT.

Запуск: python -m src.devbot.bot
"""

import asyncio
import logging
import re
from contextlib import asynccontextmanager

_COMPLEX_RE = re.compile(r"\*{0,2}сложност[ьи]\*{0,2}[:\s]+сложная", re.IGNORECASE)


def _is_complex(plan: str) -> bool:
    """Вернуть True если анализатор оценил задачу как «сложная»."""
    return bool(_COMPLEX_RE.search(plan))

from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import (
    Message, InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery,
    ReplyKeyboardMarkup, KeyboardButton, ReplyKeyboardRemove,
)
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel

from src.devbot.config import DEVBOT_TOKEN, DEVBOT_ADMIN_CHAT_ID, DEVBOT_API_PORT
from src.devbot.fsm import DevTask
from src.devbot.analyzer import analyze_task
from src.devbot.executor import execute
from src.devbot.memory import dev_memory

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

router = Router()


# ---------------------------------------------------------------------------
# Клавиатура
# ---------------------------------------------------------------------------

def _main_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="📋 Статус"), KeyboardButton(text="📜 История")],
            [KeyboardButton(text="🧠 Память"), KeyboardButton(text="🚫 Отменить")],
        ],
        resize_keyboard=True,
        input_field_placeholder="Введи /dev <задача> или выбери команду",
    )


# ---------------------------------------------------------------------------
# Проверка доступа
# ---------------------------------------------------------------------------

def _is_admin(message: Message) -> bool:
    return DEVBOT_ADMIN_CHAT_ID is None or message.chat.id == DEVBOT_ADMIN_CHAT_ID


# ---------------------------------------------------------------------------
# /start
# ---------------------------------------------------------------------------

@router.message(Command("start"))
async def cmd_start(message: Message) -> None:
    if not _is_admin(message):
        return
    await message.answer(
        "DEVBOT готов.\n\n"
        "Команды:\n"
        "/dev <задача> — поставить задачу разработчику\n"
        "/devstatus — текущая задача\n"
        "/devhistory — последние 10 задач\n"
        "/devmemory — память разработчика\n"
        "/cancel — отменить текущую задачу",
        reply_markup=_main_keyboard(),
    )


# ---------------------------------------------------------------------------
# /dev — поставить задачу
# ---------------------------------------------------------------------------

@router.message(Command("dev"))
async def cmd_dev(message: Message, state: FSMContext, bot: Bot) -> None:
    if not _is_admin(message):
        return

    current = await state.get_state()
    if current in (DevTask.executing.state, DevTask.confirming.state):
        await message.answer("⚠️ Сейчас выполняется другая задача. /cancel чтобы отменить.")
        return

    task = message.text.removeprefix("/dev").strip()
    if not task:
        await message.answer("Использование: /dev <описание задачи>")
        return

    # Сохранить задачу
    await state.set_state(DevTask.analyzing)
    task_id = await dev_memory.create_task(task)
    await state.update_data(task=task, task_id=task_id, plan="", session_id=None)

    msg = await message.answer("🔍 Анализирую задачу...")

    try:
        # Загрузить контекст
        dev_mem_text = await dev_memory.get_recent_dev_memory(10)
        advice_items = await dev_memory.get_advice(["crm", "процесс"])
        advice_text = "\n".join(f"- {i['val']}" for i in advice_items) if advice_items else ""

        plan = await analyze_task(task, dev_mem_text, advice_text)
        await state.update_data(plan=plan, dev_memory=dev_mem_text, advice_context=advice_text)
        await state.set_state(DevTask.confirming)

        complex_task = _is_complex(plan)
        approve_cb = "warn_complex" if complex_task else "approve"
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(text="✅ Выполнить", callback_data=approve_cb),
                InlineKeyboardButton(text="❌ Отменить", callback_data="cancel"),
            ]
        ])

        plan_header = "⚠️ Сложная задача — план:" if complex_task else "📋 План:"
        await bot.edit_message_text(
            chat_id=message.chat.id,
            message_id=msg.message_id,
            text=f"{plan_header}\n\n{plan}\n\nПодтверди или уточни задачу",
            reply_markup=keyboard,
        )

    except Exception as e:
        await state.clear()
        await bot.edit_message_text(
            chat_id=message.chat.id,
            message_id=msg.message_id,
            text=f"❌ Ошибка анализа: {e}",
        )
        logger.error("analyzer error: %s", e)


# ---------------------------------------------------------------------------
# Кнопка «Выполнить» (approve)
# ---------------------------------------------------------------------------

@router.callback_query(F.data == "approve", DevTask.confirming)
async def cb_approve(callback: CallbackQuery, state: FSMContext, bot: Bot) -> None:
    await callback.answer()
    data = await state.get_data()
    await _run_task(callback.message, state, bot, data, feedback=None)


# ---------------------------------------------------------------------------
# Кнопка «warn_complex» — доп. подтверждение для сложных задач
# ---------------------------------------------------------------------------

@router.callback_query(F.data == "warn_complex", DevTask.confirming)
async def cb_warn_complex(callback: CallbackQuery, state: FSMContext) -> None:
    await callback.answer()
    data = await state.get_data()
    task_preview = data.get("task", "")[:120]
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="⚠️ Да, выполнить", callback_data="approve"),
            InlineKeyboardButton(text="❌ Отменить", callback_data="cancel"),
        ]
    ])
    await callback.message.answer(
        f"⚠️ *Это сложная задача* — затрагивает несколько файлов и/или требует пересборки.\n\n"
        f"Задача: _{task_preview}_\n\n"
        f"Ты уверен? Подтверди ещё раз.",
        parse_mode="Markdown",
        reply_markup=keyboard,
    )


# ---------------------------------------------------------------------------
# Кнопка «Отменить» (cancel)
# ---------------------------------------------------------------------------

@router.callback_query(F.data == "cancel")
async def cb_cancel(callback: CallbackQuery, state: FSMContext) -> None:
    await callback.answer()
    await state.clear()
    await callback.message.edit_reply_markup(reply_markup=None)
    await callback.message.answer("🚫 Задача отменена.")


# ---------------------------------------------------------------------------
# /cancel — текстовая команда
# ---------------------------------------------------------------------------

@router.message(Command("cancel"))
@router.message(F.text == "🚫 Отменить")
async def cmd_cancel(message: Message, state: FSMContext) -> None:
    if not _is_admin(message):
        return
    current = await state.get_state()
    if current == DevTask.executing.state:
        await message.answer(
            "⚠️ Задача выполняется — дождись завершения или перезапусти бота.",
            reply_markup=_main_keyboard(),
        )
        return
    await state.clear()
    await message.answer("🚫 Задача отменена.", reply_markup=_main_keyboard())


# ---------------------------------------------------------------------------
# Уточнение в состоянии confirming → пересчитать план
# ---------------------------------------------------------------------------

@router.message(DevTask.confirming)
async def msg_clarify(message: Message, state: FSMContext, bot: Bot) -> None:
    if not _is_admin(message):
        return
    clarification = message.text.strip()
    if not clarification:
        return

    await state.set_state(DevTask.analyzing)
    data = await state.get_data()
    task = data.get("task", "")
    # Добавить уточнение к задаче
    new_task = f"{task}\n\nУточнение: {clarification}"

    msg = await message.answer("🔍 Пересчитываю план...")
    try:
        dev_mem_text = data.get("dev_memory", "")
        advice_text = data.get("advice_context", "")
        plan = await analyze_task(new_task, dev_mem_text, advice_text)
        await state.update_data(task=new_task, plan=plan)
        await state.set_state(DevTask.confirming)

        complex_task = _is_complex(plan)
        approve_cb = "warn_complex" if complex_task else "approve"
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(text="✅ Выполнить", callback_data=approve_cb),
                InlineKeyboardButton(text="❌ Отменить", callback_data="cancel"),
            ]
        ])

        plan_header = "⚠️ Сложная задача — обновлённый план:" if complex_task else "📋 Обновлённый план:"
        await bot.edit_message_text(
            chat_id=message.chat.id,
            message_id=msg.message_id,
            text=f"{plan_header}\n\n{plan}\n\nПодтверди или уточни",
            reply_markup=keyboard,
        )
    except Exception as e:
        await state.clear()
        await bot.edit_message_text(
            chat_id=message.chat.id,
            message_id=msg.message_id,
            text=f"❌ Ошибка анализа: {e}",
        )


# ---------------------------------------------------------------------------
# Фидбек после выполнения (состояние feedback)
# ---------------------------------------------------------------------------

@router.message(DevTask.feedback)
async def msg_feedback(message: Message, state: FSMContext, bot: Bot) -> None:
    if not _is_admin(message):
        return
    text = message.text.strip().lower()
    if text in ("ok", "ок", "готово", "/ok"):
        await state.clear()
        await message.answer("✅ Задача завершена.")
        return

    data = await state.get_data()
    await _run_task(message, state, bot, data, feedback=message.text)


# ---------------------------------------------------------------------------
# Внутренняя функция запуска исполнения
# ---------------------------------------------------------------------------

async def _run_task(
    message: Message,
    state: FSMContext,
    bot: Bot,
    data: dict,
    feedback: str | None,
) -> None:
    await state.set_state(DevTask.executing)
    task = data.get("task", "")
    plan = data.get("plan", "")
    task_id = data.get("task_id", 0)
    session_id = data.get("session_id")
    dev_mem = data.get("dev_memory", "")
    advice = data.get("advice_context", "")

    msg = await message.answer("🚀 Выполняю задачу...")

    async def progress_cb(text: str) -> None:
        try:
            await bot.edit_message_text(
                chat_id=message.chat.id,
                message_id=msg.message_id,
                text=text[:4000],
            )
        except Exception:
            pass

    try:
        result = await execute(
            task=task,
            plan=plan,
            dev_memory=dev_mem,
            advice_context=advice,
            feedback=feedback,
            session_id=session_id,
            progress_cb=progress_cb,
        )

        new_session_id = result.get("session_id")
        exit_code = result.get("exit_code", -1)
        result_text = result.get("result_text", "")
        pr_url = result.get("pr_url", "")
        sha = result.get("sha", "")

        await state.update_data(session_id=new_session_id)

        if exit_code == 0:
            # Записать в память (Уровень 2)
            await dev_memory.record_completion(
                task_id=task_id,
                task_text=task,
                plan=plan,
                pr_url=pr_url or "",
                sha=sha or "",
                lessons="",
            )

            summary = _build_summary(result_text, pr_url, sha)
            await state.set_state(DevTask.feedback)
            await bot.edit_message_text(
                chat_id=message.chat.id,
                message_id=msg.message_id,
                text=f"✅ Готово!\n\n{summary}\n\nЕсть замечания? Напиши или /ok",
            )
            await bot.send_message(
                message.chat.id,
                "Фидбек или /ok для закрытия задачи",
                reply_markup=_main_keyboard(),
            )

            # Авто-сброс через 10 минут если нет ответа
            asyncio.create_task(_auto_complete(state, message.chat.id, bot, 600))

        else:
            await state.clear()
            snippet = result_text[-1000:] if result_text else "нет вывода"
            await bot.edit_message_text(
                chat_id=message.chat.id,
                message_id=msg.message_id,
                text=f"❌ Ошибка (exit={exit_code}):\n\n{snippet[:800]}",
            )
            await bot.send_message(message.chat.id, "Задача завершилась с ошибкой.", reply_markup=_main_keyboard())

    except Exception as e:
        await state.clear()
        logger.error("executor error: %s", e)
        await bot.edit_message_text(
            chat_id=message.chat.id,
            message_id=msg.message_id,
            text=f"❌ Критическая ошибка: {e}",
        )
        await bot.send_message(message.chat.id, "Критическая ошибка.", reply_markup=_main_keyboard())


async def _auto_complete(state: FSMContext, chat_id: int, bot: Bot, delay: int) -> None:
    """Автозавершение feedback-режима через delay секунд."""
    await asyncio.sleep(delay)
    current = await state.get_state()
    if current == DevTask.feedback.state:
        await state.clear()
        try:
            await bot.send_message(chat_id, "⏰ Фидбек не получен — задача закрыта.")
        except Exception:
            pass


def _build_summary(result_text: str, pr_url: str | None, sha: str | None) -> str:
    parts = []
    if pr_url:
        parts.append(f"PR: {pr_url}")
    if sha:
        parts.append(f"SHA: `{sha}`")
    if not parts:
        snippet = result_text[-500:].strip() if result_text else "без деталей"
        parts.append(snippet)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# /devstatus, /devhistory, /devmemory
# ---------------------------------------------------------------------------

@router.message(Command("devstatus"))
@router.message(F.text == "📋 Статус")
async def cmd_devstatus(message: Message, state: FSMContext) -> None:
    if not _is_admin(message):
        return
    current = await state.get_state()
    data = await state.get_data()
    if not current:
        await message.answer("Нет активных задач.", reply_markup=_main_keyboard())
        return
    task = data.get("task", "—")[:100]
    state_name = {
        DevTask.analyzing.state: "🔍 Анализирую",
        DevTask.confirming.state: "⏳ Жду подтверждения",
        DevTask.executing.state: "🚀 Выполняется",
        DevTask.feedback.state: "💬 Жду фидбека",
    }.get(current, current)
    await message.answer(
        f"Состояние: {state_name}\nЗадача: `{task}`",
        parse_mode="Markdown",
        reply_markup=_main_keyboard(),
    )


@router.message(Command("devhistory"))
@router.message(F.text == "📜 История")
async def cmd_devhistory(message: Message) -> None:
    if not _is_admin(message):
        return
    mem = await dev_memory.get_recent_dev_memory(10)
    await message.answer(
        f"**Последние задачи:**\n{mem or 'пусто'}",
        parse_mode="Markdown",
        reply_markup=_main_keyboard(),
    )


@router.message(Command("devmemory"))
@router.message(F.text == "🧠 Память")
async def cmd_devmemory(message: Message) -> None:
    if not _is_admin(message):
        return
    mem = await dev_memory.get_recent_dev_memory(20)
    await message.answer(
        f"**Память разработчика:**\n{mem or 'пусто'}",
        parse_mode="Markdown",
        reply_markup=_main_keyboard(),
    )


# ---------------------------------------------------------------------------
# FastAPI — HTTP API для приёма задач от BEEBOT (/task endpoint)
# ---------------------------------------------------------------------------

class TaskRequest(BaseModel):
    id: int = 0
    text: str


_bot_instance: Bot | None = None
_dp_instance: Dispatcher | None = None


def create_api(bot: Bot, dp: Dispatcher) -> FastAPI:
    global _bot_instance, _dp_instance
    _bot_instance = bot
    _dp_instance = dp

    app = FastAPI(title="DEVBOT API")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "service": "devbot"}

    @app.post("/task")
    async def receive_task(req: TaskRequest) -> dict:
        """Принять задачу от BEEBOT и переслать в Telegram-чат администратора."""
        if not DEVBOT_ADMIN_CHAT_ID or not _bot_instance:
            return {"error": "not configured"}
        try:
            await _bot_instance.send_message(
                DEVBOT_ADMIN_CHAT_ID,
                f"📬 Новая задача #{req.id} из BEEBOT:\n\n{req.text}\n\n"
                f"Введи /dev {req.text} чтобы начать анализ.",
            )
            return {"status": "ok", "task_id": req.id}
        except Exception as e:
            logger.error("receive_task error: %s", e)
            return {"error": str(e)}

    return app


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

async def main() -> None:
    if not DEVBOT_TOKEN:
        raise ValueError("DEVBOT_TOKEN не задан в .env")

    bot = Bot(token=DEVBOT_TOKEN)
    storage = MemoryStorage()
    dp = Dispatcher(storage=storage)
    dp.include_router(router)

    api_app = create_api(bot, dp)

    # Запустить FastAPI в фоне
    config = uvicorn.Config(api_app, host="0.0.0.0", port=DEVBOT_API_PORT, log_level="warning")
    server = uvicorn.Server(config)
    api_task = asyncio.create_task(server.serve())

    logger.info("DEVBOT запущен (port=%d, admin=%s)", DEVBOT_API_PORT, DEVBOT_ADMIN_CHAT_ID)
    try:
        await dp.start_polling(bot, allowed_updates=["message", "callback_query"])
    finally:
        api_task.cancel()
        await dev_memory.close()


if __name__ == "__main__":
    asyncio.run(main())
