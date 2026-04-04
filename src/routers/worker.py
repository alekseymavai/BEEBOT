"""WorkerAgent роутер — очередь сборки заказов для работников склада."""
import logging
from typing import Optional

from aiogram import Router, Bot, types, F
from aiogram.filters import Command

from src.integram_client import IntegramClient
from src.agents import worker as worker_agent
from src.agents.worker import worker_state
from src.gift_protocol import GiftBroker
from src.services.auth_service import AuthService

logger = logging.getLogger(__name__)
router = Router()

_worker_crm: Optional[IntegramClient] = None
_bot: Optional[Bot] = None
_gift_broker: Optional[GiftBroker] = None
_auth: Optional[AuthService] = None


def setup_worker(
    crm: IntegramClient, bot: Bot,
    gift_broker: Optional[GiftBroker] = None,
    auth: Optional[AuthService] = None,
) -> None:
    global _worker_crm, _bot, _gift_broker, _auth
    _worker_crm = crm
    _bot = bot
    _gift_broker = gift_broker
    _auth = auth


def _is_worker(user_id: int) -> bool:
    if _auth:
        return _auth.is_worker(user_id)
    from src.config import WORKER_CHAT_IDS
    return bool(WORKER_CHAT_IDS and user_id in WORKER_CHAT_IDS)


def _is_admin_or_worker(user_id: int) -> bool:
    if _auth:
        return _auth.is_admin_or_worker(user_id)
    from src.config import WORKER_CHAT_IDS, ADMIN_IDS
    is_admin = bool(ADMIN_IDS and user_id in ADMIN_IDS)
    return is_admin or bool(WORKER_CHAT_IDS and user_id in WORKER_CHAT_IDS)


async def _worker_show_queue(chat_id: int, source: types.Message | types.CallbackQuery) -> None:
    """Отправить/обновить сообщение с очередью заказов."""
    if not _worker_crm:
        text = "⚠️ CRM недоступна — попробуйте позже."
        if isinstance(source, types.Message):
            await source.answer(text)
        else:
            await source.message.edit_text(text)
        return
    try:
        orders = await worker_agent.get_worker_queue(_worker_crm)
        text = worker_agent.format_queue_text(orders)
        keyboard = worker_agent.build_queue_keyboard(orders)
        if isinstance(source, types.Message):
            await source.answer(text, parse_mode="Markdown", reply_markup=keyboard)
        else:
            await source.message.edit_text(text, parse_mode="Markdown", reply_markup=keyboard)
    except Exception as e:
        logger.error("Ошибка загрузки очереди работника: %s", e)
        err = "❌ Не удалось загрузить очередь. Попробуйте ещё раз."
        if isinstance(source, types.Message):
            await source.answer(err)
        else:
            await source.message.edit_text(err)


async def _worker_show_order(callback: types.CallbackQuery, order_id: int) -> None:
    """Показать карточку заказа с чеклистом."""
    if not _worker_crm:
        await callback.message.edit_text("⚠️ CRM недоступна.")
        return
    try:
        order = await _worker_crm.get_order(order_id)
        items = await _worker_crm.get_order_items(order_id)
        text = worker_agent.format_order_card(order, items, callback.from_user.id)
        keyboard = worker_agent.build_order_keyboard(order_id, items, callback.from_user.id, order.status)
        await callback.message.edit_text(text, parse_mode="Markdown", reply_markup=keyboard)
    except Exception as e:
        logger.error("Ошибка загрузки карточки заказа %d: %s", order_id, e)
        await callback.message.edit_text("❌ Не удалось загрузить заказ. Попробуйте ещё раз.")


@router.message(Command("queue"))
async def cmd_worker_queue(message: types.Message):
    """Показать очередь сборки (для работников и администраторов)."""
    if not _is_admin_or_worker(message.from_user.id):
        return
    await _worker_show_queue(message.chat.id, message)


@router.callback_query(F.data == "worker:queue")
async def cb_worker_queue(callback: types.CallbackQuery):
    if not _is_worker(callback.from_user.id):
        await callback.answer("⛔ Нет доступа.", show_alert=True)
        return
    await callback.answer()
    await _worker_show_queue(callback.message.chat.id, callback)


@router.callback_query(F.data.startswith("worker:order:"))
async def cb_worker_order(callback: types.CallbackQuery):
    if not _is_worker(callback.from_user.id):
        await callback.answer("⛔ Нет доступа.", show_alert=True)
        return
    await callback.answer()
    order_id = int(callback.data.split(":")[2])
    await _worker_show_order(callback, order_id)


@router.callback_query(F.data.startswith("worker:take:"))
async def cb_worker_take(callback: types.CallbackQuery):
    """Работник берёт заказ в работу → статус «В сборке»."""
    if not _is_worker(callback.from_user.id):
        await callback.answer("⛔ Нет доступа.", show_alert=True)
        return
    order_id = int(callback.data.split(":")[2])
    if not _worker_crm:
        await callback.answer("⚠️ CRM недоступна.", show_alert=True)
        return
    try:
        order = await _worker_crm.get_order(order_id)
        if order.status not in ("Новый", "Подтверждён"):
            await callback.answer(f"Заказ уже в статусе «{order.status}».", show_alert=True)
            return
        await _worker_crm.update_order_status(order_id, "В сборке", comment="Взят в работу работником склада")
        worker_state.set_busy(callback.from_user.id)
        await callback.answer("✅ Взят в работу!")
    except Exception as e:
        logger.error("Ошибка взятия заказа в работу: %s", e)
        await callback.answer("❌ Ошибка. Попробуйте ещё раз.", show_alert=True)
        return
    await _worker_show_order(callback, order_id)


@router.callback_query(F.data.startswith("worker:check:"))
async def cb_worker_check(callback: types.CallbackQuery):
    """Отметить/снять позицию в чеклисте."""
    if not _is_worker(callback.from_user.id):
        await callback.answer("⛔ Нет доступа.", show_alert=True)
        return
    parts = callback.data.split(":")
    order_id = int(parts[2])
    item_id = int(parts[3])
    worker_agent.toggle_item(callback.from_user.id, order_id, item_id)
    await callback.answer()
    await _worker_show_order(callback, order_id)


@router.callback_query(F.data.startswith("worker:done:"))
async def cb_worker_done(callback: types.CallbackQuery):
    """Работник завершил сборку → уведомить пчеловода."""
    if not _is_worker(callback.from_user.id):
        await callback.answer("⛔ Нет доступа.", show_alert=True)
        return
    order_id = int(callback.data.split(":")[2])
    if not _worker_crm:
        await callback.answer("⚠️ CRM недоступна.", show_alert=True)
        return
    try:
        order = await _worker_crm.get_order(order_id)
        await _worker_crm.update_order_checklist(order_id, stock_checked=True)
        worker_agent.clear_checklist(callback.from_user.id, order_id)
        await worker_state.set_idle(callback.from_user.id)

        from src.notifications import _worker_notifier
        if _worker_notifier:
            await _worker_notifier.notify_workers_assembled(order_id, order.number or str(order_id))

        await callback.answer("📦 Пчеловод уведомлён!", show_alert=True)
    except Exception as e:
        logger.error("Ошибка завершения сборки: %s", e)
        await callback.answer("❌ Ошибка. Попробуйте ещё раз.", show_alert=True)
        return

    await _worker_show_queue(callback.message.chat.id, callback)

    # Если очередь пуста — предложить переключиться в режим покупателя (11.1)
    if _worker_crm:
        try:
            remaining = await worker_agent.get_worker_queue(_worker_crm)
            if not remaining:
                from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
                kb = InlineKeyboardMarkup(inline_keyboard=[[
                    InlineKeyboardButton(
                        text="🛒 Перейти в режим покупателя",
                        callback_data="worker:client_mode",
                    )
                ]])
                await _bot.send_message(
                    callback.message.chat.id,
                    "🎉 Все заказы собраны! Хотите перейти в режим покупателя?",
                    reply_markup=kb,
                )
        except Exception as _e:
            logger.debug("cb_worker_done: ошибка проверки очереди после завершения: %s", _e)


@router.callback_query(F.data == "worker:client_mode")
async def cb_worker_client_mode(callback: types.CallbackQuery):
    """Работник переключается в режим покупателя."""
    await callback.answer()
    uid = callback.from_user.id

    # Сохранить режим в SharedContext через GiftBroker (11.1)
    if _gift_broker:
        _gift_broker.set_interface_mode(uid, "client")

    from src.routers.keyboards import WELCOME_MESSAGE, build_main_keyboard
    await callback.message.answer(
        WELCOME_MESSAGE,
        reply_markup=build_main_keyboard(is_admin=False, view="user"),
    )
