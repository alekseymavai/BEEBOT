"""User-facing роутер: /start, главное меню, консультации, главный text-handler."""
import logging
from typing import Optional

from aiogram import Router, Bot, types, F
from aiogram.enums import ChatType
from aiogram.filters import CommandStart, Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton

from src.config import ADMIN_IDS, ACTIVE_GROUP_IDS, WORKER_CHAT_IDS
from src.orchestrator import Orchestrator
from src.agents.admin_chat import AdminChatAgent
from src.agents.logist import LogistAgent
from src.agents.beebot import is_products_query
from src.llm_client import VOICE_STYLES, DEFAULT_VOICE
from src.gift_protocol import GiftBroker
from src.routers._state import _user_styles, _admin_mode_users, _admin_view_mode
import src.routers._state as _state_mod
from src.routers.keyboards import (
    WELCOME_MESSAGE, HELP_MESSAGE, ASK_MESSAGE, PRODUCTS_MESSAGE,
    VOICE_MESSAGE, VOICE_HIVE_MESSAGE, BOT_USERNAME,
    BTN_ASK, BTN_PRODUCTS, BTN_ORDER, BTN_INSPECT,
    BTN_VOICE, BTN_HELP,
    build_main_keyboard, build_back_to_products_keyboard,
    build_products_keyboard, get_instruction_keyboard, build_voice_keyboard,
)

logger = logging.getLogger(__name__)
router = Router()

_orchestrator: Optional[Orchestrator] = None
_admin_chat_agent: Optional[AdminChatAgent] = None
_logist: Optional[LogistAgent] = None
_gift_broker: Optional[GiftBroker] = None


def setup_user(
    orchestrator: Orchestrator,
    admin_chat_agent: AdminChatAgent,
    logist: LogistAgent,
    gift_broker: Optional[GiftBroker] = None,
) -> None:
    global _orchestrator, _admin_chat_agent, _logist, _gift_broker
    _orchestrator = orchestrator
    _admin_chat_agent = admin_chat_agent
    _logist = logist
    _gift_broker = gift_broker


def _is_admin(user_id: int) -> bool:
    return bool(ADMIN_IDS and user_id in ADMIN_IDS)


def _should_respond(message: types.Message, bot_id: int) -> bool:
    if message.chat.type == ChatType.PRIVATE:
        return True
    if message.chat.id in ACTIVE_GROUP_IDS:
        return True
    text = message.text or ""
    if f"@{BOT_USERNAME}" in text:
        return True
    if message.reply_to_message and message.reply_to_message.from_user:
        if message.reply_to_message.from_user.id == bot_id:
            return True
    return False


# ---------------------------------------------------------------------------
# /start
# ---------------------------------------------------------------------------

@router.message(CommandStart())
async def cmd_start(message: types.Message):
    from src.routers.worker import _worker_show_queue
    uid = message.from_user.id
    is_admin = _is_admin(uid)

    if WORKER_CHAT_IDS and uid in WORKER_CHAT_IDS and not is_admin:
        # GiftBroker выбирает интерфейс по контексту (11.1):
        # если работник ранее переключился в режим покупателя — показать клиентский UI
        interface = (
            _gift_broker.suggest_interface(uid, is_worker=True)
            if _gift_broker else "worker"
        )
        if interface == "worker":
            await _worker_show_queue(message.chat.id, message)
            return
        # interface == "client" — сбросить режим и показать клиентский /start
        if _gift_broker:
            _gift_broker.set_interface_mode(uid, "default")

    if is_admin:
        _admin_view_mode[uid] = "admin"

    await message.answer(
        WELCOME_MESSAGE,
        reply_markup=build_main_keyboard(is_admin=is_admin, view="admin"),
    )


# ---------------------------------------------------------------------------
# Кнопки главного меню (user-side)
# ---------------------------------------------------------------------------

@router.message(F.text == BTN_ASK)
async def btn_ask(message: types.Message):
    await message.answer(ASK_MESSAGE)


@router.message(F.text == BTN_PRODUCTS)
async def btn_products(message: types.Message):
    await message.answer(PRODUCTS_MESSAGE, reply_markup=build_products_keyboard())


@router.message(F.text == BTN_ORDER)
async def btn_order(message: types.Message, state: FSMContext):
    from src.routers.fsm_order import cmd_order
    await cmd_order(message, state)


@router.message(F.text == BTN_INSPECT)
async def btn_inspect(message: types.Message, state: FSMContext):
    from src.routers.inspect import cmd_inspect
    await cmd_inspect(message, state)


@router.message(F.text == BTN_VOICE)
async def btn_voice(message: types.Message):
    current = _user_styles.get(message.from_user.id, DEFAULT_VOICE)
    await message.answer(
        VOICE_HIVE_MESSAGE,
        parse_mode="Markdown",
        reply_markup=build_voice_keyboard(current),
    )


@router.message(F.text == BTN_HELP)
async def btn_help(message: types.Message):
    await message.answer(HELP_MESSAGE)


# ---------------------------------------------------------------------------
# Команды
# ---------------------------------------------------------------------------

@router.message(Command("help"))
async def cmd_help(message: types.Message):
    await message.answer(HELP_MESSAGE, reply_markup=build_back_to_products_keyboard())


@router.message(Command("ask"))
async def cmd_ask(message: types.Message):
    await message.answer(ASK_MESSAGE)


@router.message(Command("products"))
async def cmd_products(message: types.Message):
    await message.answer(PRODUCTS_MESSAGE, reply_markup=build_products_keyboard())


@router.message(Command("voice"))
async def cmd_voice(message: types.Message):
    """Выбрать стиль ответов «Голос Улья»."""
    current = _user_styles.get(message.from_user.id, DEFAULT_VOICE)
    await message.answer(
        VOICE_HIVE_MESSAGE,
        parse_mode="Markdown",
        reply_markup=build_voice_keyboard(current),
    )


# ---------------------------------------------------------------------------
# Callback: выбор «Голоса Улья»
# ---------------------------------------------------------------------------

@router.callback_query(F.data.startswith("voice_style:"))
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


# ---------------------------------------------------------------------------
# Голосовые сообщения
# ---------------------------------------------------------------------------

@router.message(F.voice)
async def handle_voice(message: types.Message, bot: Bot):
    if not _should_respond(message, bot.id):
        return
    await message.reply(VOICE_MESSAGE)


# ---------------------------------------------------------------------------
# Главный текстовый обработчик (ДОЛЖЕН БЫТЬ ПОСЛЕДНИМ РОУТЕРОМ)
# ---------------------------------------------------------------------------

@router.message(StateFilter(None))
async def handle_question(message: types.Message, state: FSMContext, bot: Bot):
    """Обработать вопрос пользователя: KB → LLM → ответ."""
    if message.text and message.text.startswith("/"):
        return

    if not _should_respond(message, bot.id):
        return

    query = (message.text or "").replace(f"@{BOT_USERNAME}", "").strip()
    if len(query) < 3:
        await message.reply("Напиши вопрос подлиннее, чтобы я мог помочь.")
        return

    # Режим «Ассистент пчеловода»
    if _is_admin(message.from_user.id) and message.from_user.id in _admin_mode_users:
        await bot.send_chat_action(message.chat.id, "typing")
        try:
            snapshot = _state_mod._crm_snapshot
            response = await _admin_chat_agent.chat(message.from_user.id, query, snapshot=snapshot)
            try:
                await message.reply(response, parse_mode="Markdown")
            except Exception:
                await message.reply(response)
        except Exception as e:
            logger.error("AdminChat error: %s", e)
            await message.reply("Не удалось получить ответ. Попробуй ещё раз.")
        return

    if is_products_query(query):
        await message.reply(PRODUCTS_MESSAGE, reply_markup=build_products_keyboard())
        return

    logger.info("Question from %s in %s: %s", message.from_user.id, message.chat.type, query)
    await bot.send_chat_action(message.chat.id, "typing")

    style = _user_styles.get(message.from_user.id)
    user_name = message.from_user.first_name or None
    try:
        # GiftBroker (Фаза 9.3): обогащает запрос контекстом + анамнезом
        # Fallback на прямой вызов оркестратора если GiftBroker не инжектирован
        if _gift_broker:
            response, chunks = await _gift_broker.send(
                message.from_user.id, query, style=style, user_name=user_name,
            )
            intent = _gift_broker.get_intent(message.from_user.id)
        else:
            response, chunks = await _orchestrator.route(
                message.from_user.id, query, style=style, user_name=user_name,
            )
            intent = _orchestrator.get_intent(message.from_user.id)
        logger.info("Intent: %s, chunks: %d", intent, len(chunks))

        if intent == "order":
            from src.routers.fsm_order import cmd_order
            await cmd_order(message, state)
            return
        if intent == "track":
            await _handle_track(message)
            return
        if intent == "edit":
            await _handle_edit(message)
            return
        if intent == "inspect":
            from src.routers.inspect import cmd_inspect
            await cmd_inspect(message, state)
            return
        if intent == "greeting":
            await message.reply(response)
            return

        keyboard = get_instruction_keyboard(chunks)
        await message.reply(response, reply_markup=keyboard)

    except Exception as e:
        logger.error("Error handling question: %s", e)
        await message.reply("Извини, что-то пошло не так. Попробуй спросить ещё раз чуть позже.")


# ---------------------------------------------------------------------------
# Intent handlers: track (отслеживание) и edit (редактирование)
# ---------------------------------------------------------------------------

async def _handle_track(message: types.Message) -> None:
    """Показать заказы клиента с их статусами и трек-номерами."""
    tg_id = message.from_user.id
    try:
        crm = _logist._crm
        if not crm:
            await message.reply(
                "Для отслеживания заказа напишите Александру — "
                "он подскажет статус и трек-номер."
            )
            return

        orders = await crm.get_orders(client_id=None)
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
            await message.reply("У вас пока нет заказов. Напишите /order чтобы оформить.")
            return

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
        await message.reply("Не удалось загрузить информацию о заказах. Попробуйте позже.")


async def _handle_edit(message: types.Message) -> None:
    """Показать редактируемые заказы клиента."""
    tg_id = message.from_user.id
    editable_statuses = {"Новый", "Подтверждён", "В сборке"}

    try:
        crm = _logist._crm
        if not crm:
            await message.reply("Для изменения заказа напишите Александру — он внесёт правки.")
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
