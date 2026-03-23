"""Telegram bot for BEEBOT — AI assistant for a beekeeper blog."""

import asyncio
import logging
from typing import Optional

from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import CommandStart, Command, StateFilter
from aiogram.enums import ChatType
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import (
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    FSInputFile,
    ReplyKeyboardMarkup,
    KeyboardButton,
    ReplyKeyboardRemove,
)

from src.config import TELEGRAM_BOT_TOKEN, BASE_DIR, PDFS_DIR, BEEKEEPER_CHAT_ID, ADMIN_CHAT_ID, TG_SOCKS_PROXY
from src.phone_utils import validate_phone, format_phone
from src.delivery.tracker import OrderTracker
from src.integrations.uds import UDSClient, UDSPoller
from src.integram_client import IntegramClient
from src import config as app_config
from src.agents.beebot import (
    INSTRUCTIONS,
    CATEGORY_LABELS as _CATEGORY_LABELS,
    is_products_query,
    get_top_instruction,
)
from src.agents.logist import (
    LogistAgent,
    OrderFSM,
    ORDER_TIMEOUT_SECONDS,
    format_order_summary,
)
from src.agents.inspector import (
    InspectorAgent,
    InspectFSM,
    INSPECT_TIMEOUT_SECONDS,
)
from src.orchestrator import Orchestrator
from src.agents.analyst import AnalystAgent
from src.admin import router as admin_router, setup_admin
from src.llm_client import VOICE_STYLES, DEFAULT_VOICE
from src.logging_config import setup_logging

logger = logging.getLogger(__name__)

# «Голос Улья» — стиль ответов, выбранный пользователем (user_id → style_id)
_user_styles: dict[int, str] = {}

bot = Bot(token=TELEGRAM_BOT_TOKEN)  # может быть переопределён в main() с прокси
dp = Dispatcher(storage=MemoryStorage())

# Админ-роутер регистрируется первым — его команды имеют приоритет
dp.include_router(admin_router)

orchestrator = Orchestrator()
logist = LogistAgent(beekeeper_chat_id=BEEKEEPER_CHAT_ID)
analyst = AnalystAgent(
    groq_client=orchestrator._groq,
    groq_model=orchestrator._model,
)
inspector = InspectorAgent()  # KB будет подключена в main() после load_kb()

# setup_admin() вызывается в main() — один раз, после подключения CRM

# Хранилище задач таймаута (user_id → asyncio.Task)
_timeout_lock = asyncio.Lock()
_timeout_tasks: dict[int, asyncio.Task] = {}

WELCOME_MESSAGE = """Привет! Я бот-помощник Александра Дмитрова — пчеловода и автора блога о продуктах пчеловодства.

Задай мне любой вопрос о:
- Продуктах пчеловодства (мёд, перга, прополис, пыльца)
- Рецептах и дозировках
- Пчеловодстве и здоровье

Просто напиши свой вопрос или выбери раздел ниже 👇"""

HELP_MESSAGE = """Как пользоваться ботом:

В личных сообщениях — просто напиши вопрос.

В группе — обратись ко мне по имени или ответь на моё сообщение:
• @AleksandrDmitrov_BEEBOT чем полезна перга?
• Или reply на моё сообщение с вопросом

Примеры вопросов:
- Как принимать настойку прополиса?
- Чем полезна перга?
- Как укрепить иммунитет ребёнка?

/start — начать
/ask — задать вопрос
/help — эта справка
/products — список продуктов с инструкциями
/order — оформить заказ
/voice — сменить стиль ответов (Голос Улья)
/inspect — персональная консультация (Осмотр улья)
/faq — топ частых вопросов (админ)
/yt_check — проверить новые видео на канале (админ)
/yt_update — обновить базу знаний из YouTube (админ)
/stats [запрос] — аналитика продаж (админ)
/orders [статус] — список заказов (админ)
/order <ID> — детали заказа (админ)
/status <ID> <статус> — сменить статус (админ)
/track <ID> <трек> — добавить трек-номер (админ)
/clients — список клиентов (админ)
/stock — каталог товаров (админ)"""

VOICE_HIVE_MESSAGE = (
    "🎙️ *Голос Улья* — выбери стиль ответов:\n\n"
    "Это не разные персонажи — это разные грани одного пчеловода."
)

ASK_MESSAGE = "Напишите свой вопрос — я отвечу на основе знаний о продуктах пчеловодства 🐝"

PRODUCTS_MESSAGE = "Выбери продукт, чтобы получить PDF с инструкцией:"

VOICE_MESSAGE = "Голосовые сообщения пока не поддерживаю — напиши вопрос текстом, отвечу сразу 🙂"

BOT_USERNAME = "AleksandrDmitrov_BEEBOT"


def _build_start_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="📦 Все продукты", callback_data="show_products"),
        InlineKeyboardButton(text="❓ Как пользоваться", callback_data="show_help"),
    ]])


def _build_back_to_products_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="📦 Все продукты", callback_data="show_products"),
    ]])


def _build_products_keyboard() -> InlineKeyboardMarkup:
    rows = []
    current_cat = None
    cat_buttons = []

    for i, (stem, name, fname, cat) in enumerate(INSTRUCTIONS):
        if not (PDFS_DIR / fname).exists():
            continue
        if cat != current_cat:
            if cat_buttons:
                rows += [cat_buttons[j:j+2] for j in range(0, len(cat_buttons), 2)]
            rows.append([InlineKeyboardButton(
                text=_CATEGORY_LABELS.get(cat, cat),
                callback_data="noop",
            )])
            cat_buttons = []
            current_cat = cat
        cat_buttons.append(InlineKeyboardButton(text=name, callback_data=f"doc:{i}"))

    if cat_buttons:
        rows += [cat_buttons[j:j+2] for j in range(0, len(cat_buttons), 2)]

    return InlineKeyboardMarkup(inline_keyboard=rows)


def _get_instruction_keyboard(chunks: list[dict]) -> InlineKeyboardMarkup:
    result = get_top_instruction(chunks)
    if result is None:
        return InlineKeyboardMarkup(inline_keyboard=[[
            InlineKeyboardButton(text="📦 Все продукты", callback_data="show_products"),
        ]])

    idx, name, filename = result
    if not (PDFS_DIR / filename).exists():
        return InlineKeyboardMarkup(inline_keyboard=[[
            InlineKeyboardButton(text="📦 Все продукты", callback_data="show_products"),
        ]])

    return InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text=f"📄 {name}", callback_data=f"doc:{idx}"),
        InlineKeyboardButton(text="📦 Все продукты", callback_data="show_products"),
    ]])


@dp.message(CommandStart())
async def cmd_start(message: types.Message):
    await message.answer(WELCOME_MESSAGE, reply_markup=_build_start_keyboard())


@dp.message(Command("help"))
async def cmd_help(message: types.Message):
    await message.answer(HELP_MESSAGE, reply_markup=_build_back_to_products_keyboard())


@dp.message(Command("ask"))
async def cmd_ask(message: types.Message):
    await message.answer(ASK_MESSAGE)


@dp.message(Command("products"))
async def cmd_products(message: types.Message):
    await message.answer(PRODUCTS_MESSAGE, reply_markup=_build_products_keyboard())


def _build_voice_keyboard(current_style: str) -> InlineKeyboardMarkup:
    """Клавиатура выбора «Голоса Улья»."""
    rows = []
    for sid, sdata in VOICE_STYLES.items():
        check = "✅ " if sid == current_style else ""
        label = f"{check}{sdata['emoji']} {sdata['name']} — {sdata['desc']}"
        rows.append([InlineKeyboardButton(text=label, callback_data=f"voice_style:{sid}")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


@dp.message(Command("voice"))
async def cmd_voice(message: types.Message):
    """Выбрать стиль ответов «Голос Улья»."""
    current = _user_styles.get(message.from_user.id, DEFAULT_VOICE)
    await message.answer(
        VOICE_HIVE_MESSAGE,
        parse_mode="Markdown",
        reply_markup=_build_voice_keyboard(current),
    )


@dp.callback_query(F.data.startswith("voice_style:"))
async def cb_voice_style(callback: types.CallbackQuery):
    """Сохранить выбранный стиль «Голос Улья»."""
    style_id = callback.data.split(":", 1)[1]
    if style_id not in VOICE_STYLES:
        await callback.answer("Неизвестный стиль", show_alert=True)
        return
    _user_styles[callback.from_user.id] = style_id
    sdata = VOICE_STYLES[style_id]
    await callback.answer(f"{sdata['emoji']} {sdata['name']} активирован!")
    await callback.message.edit_text(
        f"✅ Стиль *{sdata['emoji']} {sdata['name']}* выбран.\n\n"
        f"_{sdata['desc']}_\n\n"
        "Задай любой вопрос — отвечу в этом стиле.",
        parse_mode="Markdown",
    )


# ===========================================================================
# «Осмотр улья» — диагностический квест
# ===========================================================================

def _inspect_skip_keyboard() -> InlineKeyboardMarkup:
    """Кнопка для пропуска оставшихся вопросов."""
    return InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="💡 Получить рекомендацию", callback_data="inspect_finish"),
    ]])


@dp.message(Command("inspect"))
async def cmd_inspect(message: types.Message, state: FSMContext):
    """Запустить диагностический диалог «Осмотр улья»."""
    current_state = await state.get_state()
    if current_state and current_state.startswith("OrderFSM"):
        await message.answer("Сначала заверши или отмени текущий заказ (/cancel).")
        return

    await state.set_state(InspectFSM.describing_issue)
    await state.update_data(inspect_collected=[])
    await message.answer(
        "🐝 *Осмотр улья* — персональная консультация\n\n"
        "Опиши, что тебя беспокоит или чего хочешь достичь. "
        "Я задам пару уточняющих вопросов и дам точную рекомендацию.",
        parse_mode="Markdown",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=[[
            InlineKeyboardButton(text="❌ Отменить", callback_data="inspect_cancel"),
        ]]),
    )


@dp.message(InspectFSM.describing_issue)
async def inspect_got_issue(message: types.Message, state: FSMContext):
    """Получить описание проблемы → задать 1-й уточняющий вопрос."""
    issue = (message.text or "").strip()
    if len(issue) < 5:
        await message.answer("Расскажи подробнее — хотя бы несколько слов.")
        return

    collected = [issue]
    await state.update_data(inspect_collected=collected)
    await bot.send_chat_action(message.chat.id, "typing")

    question = inspector.generate_question(collected)
    await state.set_state(InspectFSM.answering_q1)
    await message.answer(question, reply_markup=_inspect_skip_keyboard())


@dp.message(InspectFSM.answering_q1)
async def inspect_got_q1(message: types.Message, state: FSMContext):
    """Получить ответ на 1-й вопрос → задать 2-й или завершить."""
    answer = (message.text or "").strip()
    data = await state.get_data()
    collected: list[str] = data.get("inspect_collected", [])
    collected.append(answer)
    await state.update_data(inspect_collected=collected)
    await bot.send_chat_action(message.chat.id, "typing")

    question = inspector.generate_question(collected)
    await state.set_state(InspectFSM.answering_q2)
    await message.answer(question, reply_markup=_inspect_skip_keyboard())


@dp.message(InspectFSM.answering_q2)
async def inspect_got_q2(message: types.Message, state: FSMContext):
    """Получить ответ на 2-й вопрос → выдать рекомендацию."""
    answer = (message.text or "").strip()
    data = await state.get_data()
    collected: list[str] = data.get("inspect_collected", [])
    collected.append(answer)
    await state.clear()

    await bot.send_chat_action(message.chat.id, "typing")
    style = _user_styles.get(message.from_user.id)
    recommendation = inspector.generate_recommendation(collected, style=style)
    await message.answer(
        f"🍯 *Рекомендация*\n\n{recommendation}",
        parse_mode="Markdown",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=[[
            InlineKeyboardButton(text="🔄 Новый осмотр", callback_data="inspect_restart"),
            InlineKeyboardButton(text="📦 Все продукты", callback_data="show_products"),
        ]]),
    )


@dp.callback_query(F.data == "inspect_finish")
async def inspect_cb_finish(callback: types.CallbackQuery, state: FSMContext):
    """Досрочно завершить диалог и получить рекомендацию."""
    data = await state.get_data()
    collected: list[str] = data.get("inspect_collected", [])
    await state.clear()

    if not collected:
        await callback.answer("Сначала опиши проблему.", show_alert=True)
        return

    await callback.answer()
    await callback.message.edit_reply_markup(reply_markup=None)
    await bot.send_chat_action(callback.message.chat.id, "typing")
    style = _user_styles.get(callback.from_user.id)
    recommendation = inspector.generate_recommendation(collected, style=style)
    await callback.message.answer(
        f"🍯 *Рекомендация*\n\n{recommendation}",
        parse_mode="Markdown",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=[[
            InlineKeyboardButton(text="🔄 Новый осмотр", callback_data="inspect_restart"),
            InlineKeyboardButton(text="📦 Все продукты", callback_data="show_products"),
        ]]),
    )


@dp.callback_query(F.data == "inspect_cancel")
async def inspect_cb_cancel(callback: types.CallbackQuery, state: FSMContext):
    """Отменить диагностику."""
    await state.clear()
    await callback.answer("Осмотр отменён.")
    await callback.message.edit_text("Осмотр отменён. Задай вопрос в любое время.")


@dp.callback_query(F.data == "inspect_restart")
async def inspect_cb_restart(callback: types.CallbackQuery, state: FSMContext):
    """Начать осмотр заново."""
    await callback.answer()
    await state.set_state(InspectFSM.describing_issue)
    await state.update_data(inspect_collected=[])
    await callback.message.answer(
        "🐝 *Новый осмотр*\n\nОпиши, что тебя беспокоит.",
        parse_mode="Markdown",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=[[
            InlineKeyboardButton(text="❌ Отменить", callback_data="inspect_cancel"),
        ]]),
    )


@dp.message(Command("stats"))
async def cmd_stats(message: types.Message):
    """Аналитика продаж — только для ADMIN_CHAT_ID."""
    if ADMIN_CHAT_ID is None or message.from_user.id != ADMIN_CHAT_ID:
        await message.answer("⛔ Команда доступна только администратору.")
        return

    # Получить текст запроса из аргументов команды
    query = (message.text or "").removeprefix("/stats").strip()
    if not query:
        query = "общая статистика"

    await bot.send_chat_action(message.chat.id, "typing")
    try:
        report = await analyst.handle_query(query)
        await message.answer(report, parse_mode="Markdown")
    except Exception as e:
        logger.error("Ошибка аналитики: %s", e)
        await message.answer("Не удалось получить статистику. Попробуйте позже.")


# ===========================================================================
# YouTube updater — /yt_check, /yt_update  (только ADMIN_CHAT_ID)
# ===========================================================================

@dp.message(Command("yt_check"))
async def cmd_yt_check(message: types.Message):
    """Проверить новые видео на YouTube-канале."""
    if ADMIN_CHAT_ID is None or message.from_user.id != ADMIN_CHAT_ID:
        await message.answer("⛔ Команда доступна только администратору.")
        return

    from src import config as _cfg
    if not _cfg.YOUTUBE_API_KEY:
        await message.answer(
            "⚠️ YouTube API недоступен.\n\n"
            "Добавь в `.env`:\n"
            "```\nYOUTUBE_API_KEY=ваш_ключ\n```\n"
            "Получить ключ: console.cloud.google.com → YouTube Data API v3",
            parse_mode="Markdown",
        )
        return

    await bot.send_chat_action(message.chat.id, "typing")
    from src.youtube_updater import check_new_videos
    try:
        all_ids, new_ids = await check_new_videos(_cfg.YOUTUBE_API_KEY, _cfg.YOUTUBE_CHANNEL_HANDLE)
        if not all_ids:
            await message.answer("❌ Не удалось получить список видео. Проверь YOUTUBE_API_KEY.")
            return
        text = (
            f"📺 *Канал {_cfg.YOUTUBE_CHANNEL_HANDLE}*\n\n"
            f"Всего видео: {len(all_ids)}\n"
            f"Известных в KB: {len(__import__('src.youtube_loader', fromlist=['CHANNEL_VIDEO_IDS']).CHANNEL_VIDEO_IDS)}\n"
            f"*Новых: {len(new_ids)}*"
        )
        if new_ids:
            text += f"\n\nID новых видео:\n" + "\n".join(f"• `{v}`" for v in new_ids[:10])
            text += f"\n\nДля загрузки используй /yt\\_update"
        else:
            text += "\n\n✅ База знаний актуальна — новых видео нет."
        await message.answer(text, parse_mode="Markdown")
    except Exception as e:
        logger.error("yt_check error: %s", e)
        await message.answer(f"Ошибка при проверке: {e}")


@dp.message(Command("yt_update"))
async def cmd_yt_update(message: types.Message):
    """Скачать субтитры новых видео и пересобрать базу знаний."""
    if ADMIN_CHAT_ID is None or message.from_user.id != ADMIN_CHAT_ID:
        await message.answer("⛔ Команда доступна только администратору.")
        return

    from src import config as _cfg
    if not _cfg.YOUTUBE_API_KEY:
        await message.answer("⚠️ YOUTUBE_API_KEY не задан в .env.")
        return

    await message.answer("⏳ Проверяю новые видео и обновляю базу знаний...")
    await bot.send_chat_action(message.chat.id, "typing")
    from src.youtube_updater import run_update
    try:
        report = await run_update(_cfg.YOUTUBE_API_KEY, _cfg.YOUTUBE_CHANNEL_HANDLE)
        # Перезагрузить KB в боте если пересобрана
        if "пересобрана" in report:
            try:
                orchestrator.load_kb()
                inspector.kb = orchestrator._beebot.kb
                report += f"\n\n🔄 KB перезагружена в боте: {len(orchestrator._beebot.kb.chunks)} чанков"
            except Exception as e:
                report += f"\n\n⚠️ KB пересобрана, но перезагрузка не удалась: {e}"
        await message.answer(report, parse_mode="Markdown")
    except Exception as e:
        logger.error("yt_update error: %s", e)
        await message.answer(f"Ошибка при обновлении: {e}")


# ===========================================================================
# FAQ — /faq  (только ADMIN_CHAT_ID)
# ===========================================================================

@dp.message(Command("faq"))
async def cmd_faq(message: types.Message):
    """Показать топ частых вопросов пользователей."""
    if ADMIN_CHAT_ID is None or message.from_user.id != ADMIN_CHAT_ID:
        await message.answer("⛔ Команда доступна только администратору.")
        return

    # Принудительно сохранить FAQ на диск
    orchestrator.flush_faq()

    # Получить аргумент N (по умолчанию 20)
    args = (message.text or "").removeprefix("/faq").strip()
    try:
        n = int(args) if args else 20
        n = max(5, min(50, n))
    except ValueError:
        n = 20

    top = orchestrator.get_top_queries(n)
    if not top:
        await message.answer("📝 FAQ пока пуст — пользователи ещё не задавали вопросов.")
        return

    total = sum(c for _, c in top)
    lines = [f"📝 *Топ-{n} частых вопросов* (всего запросов: {total}):\n"]
    for i, (query, count) in enumerate(top, 1):
        bar = "▪" * min(count, 10)
        lines.append(f"{i}. [{count}] {bar} {query}")

    await message.answer("\n".join(lines), parse_mode="Markdown")


@dp.callback_query(F.data == "show_products")
async def cb_show_products(callback: types.CallbackQuery):
    await callback.answer()
    await callback.message.answer(PRODUCTS_MESSAGE, reply_markup=_build_products_keyboard())


@dp.callback_query(F.data == "show_help")
async def cb_show_help(callback: types.CallbackQuery):
    await callback.answer()
    await callback.message.answer(HELP_MESSAGE, reply_markup=_build_back_to_products_keyboard())


@dp.callback_query(F.data == "noop")
async def cb_noop(callback: types.CallbackQuery):
    await callback.answer()


@dp.callback_query(F.data.startswith("ask:"))
async def cb_ask_about_product(callback: types.CallbackQuery):
    """Prompt user to ask a question about a specific product."""
    try:
        idx = int(callback.data.split(":")[1])
        _, name, _fname, _cat = INSTRUCTIONS[idx]
    except (ValueError, IndexError):
        await callback.answer()
        await callback.message.answer(ASK_MESSAGE)
        return

    await callback.answer()
    await callback.message.answer(
        f"Напишите свой вопрос о продукте «{name}» — я отвечу 🐝"
    )


def _should_respond(message: types.Message) -> bool:
    if message.chat.type == ChatType.PRIVATE:
        return True
    text = message.text or ""
    if f"@{BOT_USERNAME}" in text:
        return True
    if message.reply_to_message and message.reply_to_message.from_user:
        if message.reply_to_message.from_user.id == bot.id:
            return True
    return False


@dp.message(F.voice)
async def handle_voice(message: types.Message):
    """Respond to voice messages with a text prompt."""
    if not _should_respond(message):
        return
    await message.reply(VOICE_MESSAGE)


@dp.message(StateFilter(None))
async def handle_question(message: types.Message, state: FSMContext):
    """Handle user questions: search KB → generate response via Groq."""
    if message.text and message.text.startswith("/"):
        return

    if not _should_respond(message):
        return

    query = (message.text or "").replace(f"@{BOT_USERNAME}", "").strip()
    if len(query) < 3:
        await message.reply("Напиши вопрос подлиннее, чтобы я мог помочь.")
        return

    if is_products_query(query):
        await message.reply(PRODUCTS_MESSAGE, reply_markup=_build_products_keyboard())
        return

    logger.info(f"Question from {message.from_user.id} in {message.chat.type}: {query}")
    await bot.send_chat_action(message.chat.id, "typing")

    style = _user_styles.get(message.from_user.id)
    try:
        response, chunks = await orchestrator.route(message.from_user.id, query, style=style)
        intent = orchestrator.get_intent(message.from_user.id)
        logger.info(f"Intent: {intent}, chunks: {len(chunks)}")

        # Маршрутизация по intent
        if intent == "order":
            await cmd_order(message, state)
            return
        if intent == "track":
            await _handle_track(message)
            return
        if intent == "edit":
            await _handle_edit(message)
            return
        if intent == "greeting":
            await message.reply(response)
            return

        keyboard = _get_instruction_keyboard(chunks)
        await message.reply(response, reply_markup=keyboard)

    except Exception as e:
        logger.error(f"Error handling question: {e}")
        await message.reply(
            "Извини, что-то пошло не так. Попробуй спросить ещё раз чуть позже."
        )


# ===========================================================================
# Обработчики intent: track (отслеживание) и edit (редактирование)
# ===========================================================================


async def _handle_track(message: types.Message) -> None:
    """Показать заказы клиента с их статусами и трек-номерами."""
    tg_id = message.from_user.id
    try:
        crm = logist._crm
        if not crm:
            await message.reply(
                "Для отслеживания заказа напишите Александру — "
                "он подскажет статус и трек-номер."
            )
            return

        orders = await crm.get_orders(client_id=None)
        # Найти клиента по telegram_id
        clients = await crm.get_clients()
        client = next((c for c in clients if c.telegram_id == tg_id), None)
        if not client:
            await message.reply(
                "Не нашёл ваших заказов. Если вы оформляли заказ — "
                "напишите Александру, он поможет."
            )
            return

        my_orders = [o for o in orders if o.client_id == client.id]
        if not my_orders:
            await message.reply(
                "У вас пока нет заказов. Напишите /order чтобы оформить."
            )
            return

        # Формат: последние 5 заказов
        lines = ["📦 *Ваши заказы:*\n"]
        for o in my_orders[-5:]:
            status_emoji = {
                "Новый": "🆕", "Подтверждён": "✅", "В сборке": "📦",
                "Отправлен": "🚚", "Доставлен": "🎉", "Отменён": "❌",
            }.get(o.status, "❓")
            line = f"{status_emoji} *#{o.number}* — {o.status}"
            if o.tracking_number:
                line += f"\n   Трек: `{o.tracking_number}`"
            if o.total:
                line += f" · {o.total:.0f} ₽"
            lines.append(line)

        lines.append("\nЕсли есть вопросы — напишите Александру.")
        await message.reply("\n".join(lines), parse_mode="Markdown")

    except Exception as e:
        logger.error("Ошибка отслеживания: %s", e)
        await message.reply(
            "Не удалось загрузить информацию о заказах. Попробуйте позже."
        )


async def _handle_edit(message: types.Message) -> None:
    """Показать редактируемые заказы клиента."""
    tg_id = message.from_user.id
    editable_statuses = {"Новый", "Подтверждён", "В сборке"}

    try:
        crm = logist._crm
        if not crm:
            await message.reply(
                "Для изменения заказа напишите Александру — он внесёт правки."
            )
            return

        clients = await crm.get_clients()
        client = next((c for c in clients if c.telegram_id == tg_id), None)
        if not client:
            await message.reply("Не нашёл ваших заказов.")
            return

        orders = await crm.get_orders(client_id=None)
        my_orders = [
            o for o in orders
            if o.client_id == client.id and o.status in editable_statuses
        ]

        if not my_orders:
            await message.reply(
                "Нет заказов, которые можно изменить.\n"
                "Заказы доступны для редактирования в статусах: "
                "Новый, Подтверждён, В сборке."
            )
            return

        # Показать заказы с кнопками
        lines = ["✏️ *Заказы, доступные для изменения:*\n"]
        buttons = []
        for o in my_orders[-5:]:
            total_str = f" · {o.total:.0f} ₽" if o.total else ""
            lines.append(f"• *#{o.number}* — {o.status}{total_str}")
            buttons.append([InlineKeyboardButton(
                text=f"Изменить #{o.number}",
                callback_data=f"edit_order:{o.id}",
            )])

        lines.append(
            "\nНажмите кнопку ниже или напишите Александру — "
            "он поможет внести изменения."
        )

        keyboard = InlineKeyboardMarkup(inline_keyboard=buttons)
        await message.reply("\n".join(lines), parse_mode="Markdown", reply_markup=keyboard)

    except Exception as e:
        logger.error("Ошибка списка заказов для редактирования: %s", e)
        await message.reply("Не удалось загрузить заказы. Попробуйте позже.")


# ===========================================================================
# Callback: редактирование заказа через Telegram
# ===========================================================================


@dp.callback_query(F.data.startswith("edit_order:"))
async def cb_edit_order(callback: types.CallbackQuery):
    """Показать опции редактирования конкретного заказа."""
    order_id = int(callback.data.split(":")[1])
    await callback.answer()

    try:
        crm = logist._crm
        if not crm:
            await callback.message.answer("CRM недоступна.")
            return

        order = await crm.get_order(order_id)
        items = await crm.get_order_items(order_id)

        lines = [f"📋 *Заказ #{order.number}*\n"]
        if items:
            lines.append("*Товары:*")
            for item in items:
                lines.append(f"  • {item.product_name or 'Товар'} × {item.quantity} = {item.total:.0f} ₽")
        lines.append(f"\n🏠 Адрес: {order.delivery_address or '—'}")
        lines.append(f"🚚 Доставка: {order.delivery_method or '—'}")
        if order.delivery_cost:
            lines.append(f"💰 Доставка: {order.delivery_cost:.0f} ₽")
        lines.append(f"💰 *Итого: {order.total:.0f} ₽*")

        lines.append(
            "\nЧтобы внести изменения, напишите что хотите поменять, "
            "например:\n"
            "• «Поменяй адрес на г. Казань, ул. Мира 5»\n"
            "• «Добавь ещё одну банку мёда»\n"
            "\nАлександр обработает вашу просьбу."
        )

        await callback.message.answer("\n".join(lines), parse_mode="Markdown")

    except Exception as e:
        logger.error("Ошибка загрузки заказа %d: %s", order_id, e)
        await callback.message.answer("Не удалось загрузить заказ.")


# ===========================================================================
# Helpers: таймаут диалога заказа
# ===========================================================================


async def _cancel_timeout(user_id: int) -> None:
    """Отменить задачу таймаута для пользователя."""
    async with _timeout_lock:
        task = _timeout_tasks.pop(user_id, None)
    if task and not task.done():
        task.cancel()


async def _timeout_dialog(user_id: int, chat_id: int, state: FSMContext) -> None:
    """Завершить диалог заказа по таймауту."""
    await asyncio.sleep(ORDER_TIMEOUT_SECONDS)
    current = await state.get_state()
    if current is not None:
        await state.clear()
        async with _timeout_lock:
            _timeout_tasks.pop(user_id, None)
        try:
            await bot.send_message(
                chat_id,
                "⏰ Диалог оформления заказа завершён по таймауту (15 минут).\n"
                "Напишите /order чтобы начать заново.",
                reply_markup=ReplyKeyboardRemove(),
            )
        except Exception as e:
            logger.warning("Не удалось отправить сообщение о таймауте: %s", e)


async def _reset_timeout(user_id: int, chat_id: int, state: FSMContext) -> None:
    """Сбросить таймаут — отменить старый и запустить новый."""
    await _cancel_timeout(user_id)
    task = asyncio.create_task(_timeout_dialog(user_id, chat_id, state))
    async with _timeout_lock:
        _timeout_tasks[user_id] = task


# ===========================================================================
# Вспомогательные клавиатуры для FSM-диалога
# ===========================================================================


def _delivery_keyboard(options: list[dict]) -> ReplyKeyboardMarkup:
    """Клавиатура с вариантами доставки."""
    return ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text=opt["label"])] for opt in options],
        resize_keyboard=True,
        one_time_keyboard=True,
    )


def _confirm_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="✅ Да, подтвердить")],
            [KeyboardButton(text="❌ Нет, отменить")],
        ],
        resize_keyboard=True,
        one_time_keyboard=True,
    )


# ===========================================================================
# /order — начало диалога оформления заказа
# ===========================================================================


@dp.message(Command("order"))
async def cmd_order(message: types.Message, state: FSMContext) -> None:
    """Начать диалог оформления заказа."""
    user_id = message.from_user.id
    chat_id = message.chat.id

    # Показать каталог
    catalog_text, products = await logist.start_order(user_id)

    # Сохранить каталог в FSM-контекст
    products_data = [
        {
            "id": p.id,
            "name": p.name,
            "price": p.price,
            "weight": p.weight,
        }
        for p in products
    ]
    await state.update_data(products=products_data, cart=[])
    await state.set_state(OrderFSM.choosing_product)

    await _reset_timeout(user_id, chat_id, state)
    await message.answer(catalog_text, parse_mode="Markdown")


# ===========================================================================
# /cancel — отмена диалога на любом шаге
# ===========================================================================


@dp.message(Command("cancel"), StateFilter(OrderFSM))
async def cmd_cancel_order(message: types.Message, state: FSMContext) -> None:
    """Отменить диалог оформления заказа."""
    await _cancel_timeout(message.from_user.id)
    await state.clear()
    await message.answer(
        "Заказ отменён. Напишите /order чтобы начать заново.",
        reply_markup=ReplyKeyboardRemove(),
    )


# ===========================================================================
# Шаг 1: Выбор товаров
# ===========================================================================


@dp.message(OrderFSM.choosing_product)
async def fsm_choose_product(message: types.Message, state: FSMContext) -> None:
    """Обработать выбор товаров из каталога."""
    user_id = message.from_user.id
    data = await state.get_data()
    products = data.get("products", [])

    cart, error = logist.parse_product_selection(message.text or "", products)
    if error:
        await message.answer(error)
        return

    # Загрузить данные постоянного клиента
    existing_client = await logist.get_existing_client(user_id)

    await state.update_data(cart=cart)

    # Сохранить телефон для предзаполнения на шаге 3
    if existing_client and existing_client.phone:
        await state.update_data(_existing_client_phone=existing_client.phone)

    prefill = ""
    if existing_client and existing_client.full_name:
        name = existing_client.full_name
        # Не предлагать автоподстановку если имя = "Telegram XXXXX"
        if not name.startswith("Telegram "):
            prefill = f"\nПодсказка: в прошлый раз вы представились как *{name}*.\nПросто отправьте *да* или введите другое."
            await state.update_data(prefill_name=name)

    await state.set_state(OrderFSM.entering_name)
    await _reset_timeout(user_id, message.chat.id, state)
    await message.answer(
        f"Отлично! Введите ваше *ФИО* (Фамилия Имя Отчество).{prefill}",
        parse_mode="Markdown",
    )


# ===========================================================================
# Шаг 2: Ввод ФИО
# ===========================================================================


@dp.message(OrderFSM.entering_name)
async def fsm_enter_name(message: types.Message, state: FSMContext) -> None:
    """Обработать ввод ФИО."""
    name = (message.text or "").strip()
    if len(name) < 3:
        await message.answer("Пожалуйста, введите полное ФИО (минимум 3 символа).")
        return

    data = await state.get_data()
    # Если пользователь нажал «Enter» без текста — использовать предзаполненное
    if name.lower() in ("да", "+", "ok", "ок") and data.get("prefill_name"):
        name = data["prefill_name"]

    await state.update_data(full_name=name)

    # Предзаполнение телефона из данных существующего клиента
    data = await state.get_data()
    phone_prefill = ""
    existing_client = data.get("_existing_client_phone")
    if existing_client:
        phone_prefill = (
            f"\nПодсказка: ваш прошлый номер — *{format_phone(existing_client)}*.\n"
            "Отправьте *да* или введите другой."
        )
        await state.update_data(prefill_phone=existing_client)

    await state.set_state(OrderFSM.entering_phone)
    await _reset_timeout(message.from_user.id, message.chat.id, state)
    await message.answer(
        f"📞 Введите ваш *номер телефона*:{phone_prefill}",
        parse_mode="Markdown",
    )


# ===========================================================================
# Шаг 3: Ввод телефона
# ===========================================================================


@dp.message(OrderFSM.entering_phone)
async def fsm_enter_phone(message: types.Message, state: FSMContext) -> None:
    """Обработать ввод номера телефона."""
    phone_raw = (message.text or "").strip()
    data = await state.get_data()

    # Если пользователь подтвердил предзаполненный номер
    if phone_raw.lower() in ("да", "+", "ok", "ок") and data.get("prefill_phone"):
        phone = data["prefill_phone"]
    else:
        phone, error = validate_phone(phone_raw)
        if phone is None:
            await message.answer(error)
            return

    existing_client = None
    if data.get("prefill_address") is None:
        existing_client = await logist.get_existing_client(message.from_user.id)

    await state.update_data(phone=phone)

    prefill = ""
    if existing_client and existing_client.address:
        prefill = (
            f"\nПодсказка: ваш прошлый адрес — *{existing_client.address}*.\n"
            "Отправьте его или введите новый."
        )
        await state.update_data(prefill_address=existing_client.address)

    await state.set_state(OrderFSM.entering_address)
    await _reset_timeout(message.from_user.id, message.chat.id, state)
    await message.answer(
        f"🏠 Введите *адрес доставки* (город, улица, дом, квартира).{prefill}",
        parse_mode="Markdown",
    )


# ===========================================================================
# Шаг 4: Ввод адреса
# ===========================================================================


@dp.message(OrderFSM.entering_address)
async def fsm_enter_address(message: types.Message, state: FSMContext) -> None:
    """Обработать ввод адреса доставки."""
    address = (message.text or "").strip()
    if len(address) < 5:
        await message.answer("Введите полный адрес (минимум 5 символов).")
        return

    data = await state.get_data()
    # Если ввели подтверждение — использовать предзаполненный адрес
    if address.lower() in ("да", "+", "ok", "ок") and data.get("prefill_address"):
        address = data["prefill_address"]

    cart = data.get("cart", [])
    options = await logist.get_delivery_options(cart, address=address)

    await state.update_data(address=address, delivery_options=options)
    await state.set_state(OrderFSM.choosing_delivery)
    await _reset_timeout(message.from_user.id, message.chat.id, state)

    keyboard = _delivery_keyboard(options)
    options_text = "\n".join(f"  {opt['label']}" for opt in options)
    await message.answer(
        f"🚚 Выберите *способ доставки*:\n\n{options_text}",
        parse_mode="Markdown",
        reply_markup=keyboard,
    )


# ===========================================================================
# Шаг 5: Выбор способа доставки
# ===========================================================================


@dp.message(OrderFSM.choosing_delivery)
async def fsm_choose_delivery(message: types.Message, state: FSMContext) -> None:
    """Обработать выбор способа доставки."""
    user_input = (message.text or "").strip()
    data = await state.get_data()
    options = data.get("delivery_options", [])

    # Сопоставить ввод пользователя с вариантами
    chosen = None
    chosen_cost = 0.0
    for opt in options:
        if opt["method"].lower() in user_input.lower() or opt["label"] in user_input:
            chosen = opt["method"]
            chosen_cost = opt["cost"]
            break

    if not chosen:
        labels = "\n".join(f"  • {opt['label']}" for opt in options)
        await message.answer(
            f"Пожалуйста, выберите один из вариантов:\n{labels}",
            reply_markup=_delivery_keyboard(options),
        )
        return

    cart = data.get("cart", [])
    full_name = data.get("full_name", "")
    phone = data.get("phone", "")
    address = data.get("address", "")

    summary = format_order_summary(
        cart=cart,
        full_name=full_name,
        phone=phone,
        address=address,
        delivery=chosen,
        delivery_cost=chosen_cost,
    )

    await state.update_data(delivery=chosen, delivery_cost=chosen_cost)
    await state.set_state(OrderFSM.confirming_order)
    await _reset_timeout(message.from_user.id, message.chat.id, state)

    await message.answer(
        summary,
        parse_mode="Markdown",
        reply_markup=_confirm_keyboard(),
    )


# ===========================================================================
# Шаг 6: Подтверждение заказа
# ===========================================================================


@dp.message(OrderFSM.confirming_order)
async def fsm_confirm_order(message: types.Message, state: FSMContext) -> None:
    """Обработать подтверждение или отмену заказа."""
    answer = (message.text or "").strip().lower()

    if answer in ("да", "✅ да, подтвердить", "yes", "подтвердить", "+"):
        await state.set_state(OrderFSM.creating_order)
        await _do_create_order(message, state)
    elif answer in ("нет", "❌ нет, отменить", "no", "отменить", "-"):
        await _cancel_timeout(message.from_user.id)
        await state.clear()
        await message.answer(
            "Заказ отменён. Напишите /order чтобы начать заново.",
            reply_markup=ReplyKeyboardRemove(),
        )
    else:
        await message.answer(
            "Пожалуйста, отправьте *да* для подтверждения или *нет* для отмены.",
            parse_mode="Markdown",
            reply_markup=_confirm_keyboard(),
        )


# ===========================================================================
# Шаг 7: Создание заказа
# ===========================================================================


async def _do_create_order(message: types.Message, state: FSMContext) -> None:
    """Создать заказ в Integram и уведомить пчеловода."""
    await _cancel_timeout(message.from_user.id)
    data = await state.get_data()
    await state.clear()

    await bot.send_chat_action(message.chat.id, "typing")

    success, response_text = await logist.create_order(
        telegram_id=message.from_user.id,
        full_name=data.get("full_name", ""),
        phone=data.get("phone", ""),
        address=data.get("address", ""),
        delivery=data.get("delivery", ""),
        delivery_cost=data.get("delivery_cost", 0.0),
        cart=data.get("cart", []),
        telegram_username=message.from_user.username,
    )

    await message.answer(
        response_text,
        parse_mode="Markdown",
        reply_markup=ReplyKeyboardRemove(),
    )

    if success:
        # Уведомить пчеловода
        cart = data.get("cart", [])
        delivery_cost = data.get("delivery_cost", 0.0)
        items_total = sum(i["qty"] * i["unit_price"] for i in cart)
        total = items_total + delivery_cost
        items_str = "\n".join(f"  • {i['name']} × {i['qty']}" for i in cart)
        beekeeper_msg = (
            f"👤 {data.get('full_name', '')}\n"
            f"📞 {data.get('phone', '')}\n"
            f"🏠 {data.get('address', '')}\n"
            f"🚚 {data.get('delivery', '')} — {delivery_cost:.0f} ₽\n\n"
            f"*Товары:*\n{items_str}\n\n"
            f"💰 Итого: {total:.0f} ₽"
        )
        await logist.notify_beekeeper(bot, beekeeper_msg)


@dp.callback_query(F.data.startswith("doc:"))
async def send_instruction_pdf(callback: types.CallbackQuery):
    """Send the instruction PDF when user taps the button."""
    try:
        idx = int(callback.data.split(":")[1])
        _, name, filename, _cat = INSTRUCTIONS[idx]
    except (ValueError, IndexError):
        await callback.answer("Инструкция не найдена.", show_alert=True)
        return

    pdf_path = PDFS_DIR / filename
    if not pdf_path.exists():
        await callback.answer("Файл не найден на сервере.", show_alert=True)
        return

    await callback.answer()
    await callback.message.answer_document(
        document=FSInputFile(str(pdf_path), filename=filename),
        caption=f"📄 {name}",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=[[
            InlineKeyboardButton(text=f"❓ Задать вопрос о {name}", callback_data=f"ask:{idx}"),
        ], [
            InlineKeyboardButton(text="📦 Все продукты", callback_data="show_products"),
        ]]),
    )


async def _alert(text: str) -> None:
    """Отправить алерт пчеловоду. Тихо проглатывает ошибки."""
    if not BEEKEEPER_CHAT_ID:
        return
    try:
        await bot.send_message(BEEKEEPER_CHAT_ID, text)
    except Exception as e:
        logger.warning("Не удалось отправить Telegram-алерт: %s", e)


@dp.startup()
async def on_startup(**_kwargs) -> None:
    """Отправить алерт о старте после установки соединения с Telegram."""
    await _alert("🟢 BEEBOT запущен")


async def main():
    global bot
    setup_logging()
    # Инициализировать SOCKS5-прокси здесь, внутри event loop
    if TG_SOCKS_PROXY:
        from aiogram.client.session.aiohttp import AiohttpSession
        bot = Bot(token=TELEGRAM_BOT_TOKEN, session=AiohttpSession(proxy=TG_SOCKS_PROXY))
        logger.info("Telegram via SOCKS5 proxy: %s", TG_SOCKS_PROXY)
    logger.info("Starting BEEBOT...")
    try:
        orchestrator.load_kb()
        inspector.kb = orchestrator._beebot.kb  # разделяем KB с beebot
        kb = orchestrator._beebot.kb
        logger.info(f"Knowledge base loaded: {len(kb.chunks)} chunks")
    except FileNotFoundError:
        logger.error("Knowledge base not found! Run `python -m src.build_kb` first.")
        return

    # --- Integram CRM: общий клиент для всех агентов ---
    integram_client: Optional[IntegramClient] = None
    if app_config.INTEGRAM_URL and app_config.INTEGRAM_LOGIN:
        try:
            integram_client = IntegramClient()
            await integram_client.authenticate()
            # Передать CRM-клиент всем агентам
            logist._crm = integram_client
            analyst._crm = integram_client
            # Динамический keyword-буст из CRM
            try:
                products = await integram_client.get_products()
                names = [p.name for p in products if p.name]
                added = kb.update_keywords_from_products(names)
                if added:
                    logger.info("KB keyword-буст: добавлено %d ключей из CRM (%d товаров)", added, len(names))
            except Exception as _e:
                logger.warning("Не удалось обновить keyword-буст из CRM: %s", _e)
            logger.info("Integram CRM подключена — агенты получили доступ к данным.")
        except Exception as e:
            logger.warning("Integram CRM недоступна: %s — агенты работают без CRM.", e)
    else:
        logger.info("Integram CRM не настроена — агенты работают без CRM.")

    # Инициализировать админ-модуль (один раз, с CRM если доступна)
    setup_admin(bot, crm=integram_client)

    # --- Авто-трекинг: фоновая проверка статуса отправлений ---
    order_tracker: Optional[OrderTracker] = None
    if integram_client:
        from src.web.notifications import notify_client_status_change
        order_tracker = OrderTracker(
            crm=integram_client,
            notify_fn=notify_client_status_change,
        )
        asyncio.create_task(order_tracker.run())
        logger.info("Авто-трекинг отправлений запущен.")

    # --- UDS Poller: фоновая синхронизация заказов из UDS ---
    uds_poller: Optional[UDSPoller] = None
    uds_client: Optional[UDSClient] = None

    if app_config.UDS_API_KEY and app_config.UDS_COMPANY_ID:
        if integram_client:
            try:
                uds_client = UDSClient()
                uds_poller = UDSPoller(
                    uds_client=uds_client,
                    integram_client=integram_client,
                    bot=bot,
                    notify_chat_id=BEEKEEPER_CHAT_ID,
                )
                asyncio.create_task(uds_poller.run())
                logger.info("UDS Poller запущен.")
            except Exception as e:
                logger.warning("UDS Poller не удалось запустить: %s", e)
        else:
            logger.warning("UDS Poller пропущен — Integram CRM не подключена.")
    else:
        logger.info("UDS не настроен (UDS_API_KEY/UDS_COMPANY_ID не заданы) — поллер пропущен.")

    logger.info("Bot is running!")
    _crashed = False
    try:
        await dp.start_polling(bot)
    except Exception as exc:
        _crashed = True
        logger.exception("Критическая ошибка бота: %s", exc)
        await _alert(f"❌ BEEBOT упал: {exc}")
        raise
    finally:
        if not _crashed:
            await _alert("🔴 BEEBOT остановлен")
        # Graceful shutdown: остановить трекер, UDS poller и закрыть клиенты
        if order_tracker:
            order_tracker.stop()
        if uds_poller:
            uds_poller.stop()
        if uds_client:
            await uds_client.close()
        if integram_client:
            await integram_client.close()


if __name__ == "__main__":
    asyncio.run(main())
