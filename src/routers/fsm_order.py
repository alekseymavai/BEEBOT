"""OrderFSM — 7-шаговый диалог оформления заказа + callbacks продуктов/инструкций."""
import asyncio
import logging
from typing import Optional

from aiogram import Router, Bot, types, F
from aiogram.filters import Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.types import (
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    FSInputFile,
    ReplyKeyboardMarkup,
    KeyboardButton,
    ReplyKeyboardRemove,
)

from src.config import PDFS_DIR
from src.agents.logist import LogistAgent, OrderFSM, ORDER_TIMEOUT_SECONDS, format_order_summary
from src.agents.beebot import INSTRUCTIONS
from src.phone_utils import validate_phone, format_phone
from src.routers._state import _timeout_lock, _timeout_tasks
from src.routers.keyboards import (
    PRODUCTS_MESSAGE, HELP_MESSAGE, ASK_MESSAGE,
    build_products_keyboard, build_back_to_products_keyboard,
)

logger = logging.getLogger(__name__)
router = Router()

_logist: Optional[LogistAgent] = None
_bot: Optional[Bot] = None


def setup_fsm_order(logist: LogistAgent, bot: Bot) -> None:
    global _logist, _bot
    _logist = logist
    _bot = bot


# ---------------------------------------------------------------------------
# Клавиатуры FSM
# ---------------------------------------------------------------------------

def _delivery_keyboard(options: list[dict]) -> ReplyKeyboardMarkup:
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


# ---------------------------------------------------------------------------
# Helpers: таймаут диалога заказа
# ---------------------------------------------------------------------------

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
            await _bot.send_message(
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


# ---------------------------------------------------------------------------
# /order — начало диалога оформления заказа
# ---------------------------------------------------------------------------

@router.message(Command("order"))
async def cmd_order(message: types.Message, state: FSMContext) -> None:
    """Начать диалог оформления заказа."""
    user_id = message.from_user.id
    chat_id = message.chat.id

    catalog_text, products = await _logist.start_order(user_id)

    products_data = [
        {"id": p.id, "name": p.name, "price": p.price, "weight": p.weight}
        for p in products
    ]
    await state.update_data(products=products_data, cart=[])
    await state.set_state(OrderFSM.choosing_product)

    await _reset_timeout(user_id, chat_id, state)
    await message.answer(catalog_text, parse_mode="Markdown")


# ---------------------------------------------------------------------------
# /cancel — отмена диалога на любом шаге
# ---------------------------------------------------------------------------

@router.message(Command("cancel"), StateFilter(OrderFSM))
async def cmd_cancel_order(message: types.Message, state: FSMContext) -> None:
    """Отменить диалог оформления заказа."""
    await _cancel_timeout(message.from_user.id)
    await state.clear()
    await message.answer(
        "Заказ отменён. Напишите /order чтобы начать заново.",
        reply_markup=ReplyKeyboardRemove(),
    )


# ---------------------------------------------------------------------------
# Шаг 1: Выбор товаров
# ---------------------------------------------------------------------------

@router.message(OrderFSM.choosing_product)
async def fsm_choose_product(message: types.Message, state: FSMContext) -> None:
    """Обработать выбор товаров из каталога."""
    user_id = message.from_user.id
    data = await state.get_data()
    products = data.get("products", [])

    cart, error = _logist.parse_product_selection(message.text or "", products)
    if error:
        await message.answer(error)
        return

    existing_client = await _logist.get_existing_client(user_id)
    await state.update_data(cart=cart)

    if existing_client and existing_client.phone:
        await state.update_data(_existing_client_phone=existing_client.phone)

    # Предзаполнение адреса: из истории заказов (приоритет) или из профиля клиента
    last_address = await _logist.get_last_delivery_address(user_id)
    prefill_address = last_address or (existing_client.address if existing_client else None)
    if prefill_address:
        await state.update_data(prefill_address=prefill_address)

    prefill = ""
    if existing_client and existing_client.full_name:
        name = existing_client.full_name
        if not name.startswith("Telegram "):
            prefill = f"\nПодсказка: в прошлый раз вы представились как *{name}*.\nПросто отправьте *да* или введите другое."
            await state.update_data(prefill_name=name)

    await state.set_state(OrderFSM.entering_name)
    await _reset_timeout(user_id, message.chat.id, state)
    await message.answer(
        f"Отлично! Введите ваше *ФИО* (Фамилия Имя Отчество).{prefill}",
        parse_mode="Markdown",
    )


# ---------------------------------------------------------------------------
# Шаг 2: Ввод ФИО
# ---------------------------------------------------------------------------

@router.message(OrderFSM.entering_name)
async def fsm_enter_name(message: types.Message, state: FSMContext) -> None:
    """Обработать ввод ФИО."""
    name = (message.text or "").strip()
    if len(name) < 3:
        await message.answer("Пожалуйста, введите полное ФИО (минимум 3 символа).")
        return

    data = await state.get_data()
    if name.lower() in ("да", "+", "ok", "ок") and data.get("prefill_name"):
        name = data["prefill_name"]

    await state.update_data(full_name=name)

    data = await state.get_data()
    phone_prefill = ""
    existing_phone = data.get("_existing_client_phone")
    if existing_phone:
        phone_prefill = (
            f"\nПодсказка: ваш прошлый номер — *{format_phone(existing_phone)}*.\n"
            "Отправьте *да* или введите другой."
        )
        await state.update_data(prefill_phone=existing_phone)

    await state.set_state(OrderFSM.entering_phone)
    await _reset_timeout(message.from_user.id, message.chat.id, state)
    await message.answer(
        f"📞 Введите ваш *номер телефона*:{phone_prefill}",
        parse_mode="Markdown",
    )


# ---------------------------------------------------------------------------
# Шаг 3: Ввод телефона
# ---------------------------------------------------------------------------

@router.message(OrderFSM.entering_phone)
async def fsm_enter_phone(message: types.Message, state: FSMContext) -> None:
    """Обработать ввод номера телефона."""
    phone_raw = (message.text or "").strip()
    data = await state.get_data()

    if phone_raw.lower() in ("да", "+", "ok", "ок") and data.get("prefill_phone"):
        phone = data["prefill_phone"]
    else:
        phone, error = validate_phone(phone_raw)
        if phone is None:
            await message.answer(error)
            return

    await state.update_data(phone=phone)

    prefill = ""
    if data.get("prefill_address"):
        prefill = (
            f"\nПодсказка: ваш прошлый адрес — *{data['prefill_address']}*.\n"
            "Отправьте *да* чтобы использовать его, или введите новый."
        )

    await state.set_state(OrderFSM.entering_address)
    await _reset_timeout(message.from_user.id, message.chat.id, state)
    await message.answer(
        f"🏠 Введите *адрес доставки* (город, улица, дом, квартира).{prefill}",
        parse_mode="Markdown",
    )


# ---------------------------------------------------------------------------
# Шаг 4: Ввод адреса
# ---------------------------------------------------------------------------

@router.message(OrderFSM.entering_address)
async def fsm_enter_address(message: types.Message, state: FSMContext) -> None:
    """Обработать ввод адреса доставки."""
    address = (message.text or "").strip()
    if len(address) < 5:
        await message.answer("Введите полный адрес (минимум 5 символов).")
        return

    data = await state.get_data()
    if address.lower() in ("да", "+", "ok", "ок") and data.get("prefill_address"):
        address = data["prefill_address"]

    cart = data.get("cart", [])
    options = await _logist.get_delivery_options(cart, address=address)

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


# ---------------------------------------------------------------------------
# Шаг 5: Выбор способа доставки
# ---------------------------------------------------------------------------

@router.message(OrderFSM.choosing_delivery)
async def fsm_choose_delivery(message: types.Message, state: FSMContext) -> None:
    """Обработать выбор способа доставки."""
    user_input = (message.text or "").strip()
    data = await state.get_data()
    options = data.get("delivery_options", [])

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
    summary = format_order_summary(
        cart=cart,
        full_name=data.get("full_name", ""),
        phone=data.get("phone", ""),
        address=data.get("address", ""),
        delivery=chosen,
        delivery_cost=chosen_cost,
    )

    await state.update_data(delivery=chosen, delivery_cost=chosen_cost)
    await state.set_state(OrderFSM.confirming_order)
    await _reset_timeout(message.from_user.id, message.chat.id, state)
    await message.answer(summary, parse_mode="Markdown", reply_markup=_confirm_keyboard())


# ---------------------------------------------------------------------------
# Шаг 6: Подтверждение заказа
# ---------------------------------------------------------------------------

@router.message(OrderFSM.confirming_order)
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


# ---------------------------------------------------------------------------
# Шаг 7: Создание заказа
# ---------------------------------------------------------------------------

async def _do_create_order(message: types.Message, state: FSMContext) -> None:
    """Создать заказ в Integram и уведомить пчеловода."""
    await _cancel_timeout(message.from_user.id)
    data = await state.get_data()
    await state.clear()

    await _bot.send_chat_action(message.chat.id, "typing")

    success, response_text = await _logist.create_order(
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
        await _logist.notify_beekeeper(_bot, beekeeper_msg)


# ---------------------------------------------------------------------------
# Callbacks: продукты, инструкции, редактирование заказа
# ---------------------------------------------------------------------------

@router.callback_query(F.data == "show_products")
async def cb_show_products(callback: types.CallbackQuery):
    await callback.answer()
    await callback.message.answer(PRODUCTS_MESSAGE, reply_markup=build_products_keyboard())


@router.callback_query(F.data == "show_help")
async def cb_show_help(callback: types.CallbackQuery):
    await callback.answer()
    await callback.message.answer(HELP_MESSAGE, reply_markup=build_back_to_products_keyboard())


@router.callback_query(F.data == "noop")
async def cb_noop(callback: types.CallbackQuery):
    await callback.answer()


@router.callback_query(F.data.startswith("ask:"))
async def cb_ask_about_product(callback: types.CallbackQuery):
    """Предложить задать вопрос о конкретном продукте."""
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


@router.callback_query(F.data.startswith("doc:"))
async def send_instruction_pdf(callback: types.CallbackQuery):
    """Отправить PDF-инструкцию при нажатии кнопки."""
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


@router.callback_query(F.data.startswith("edit_order:"))
async def cb_edit_order(callback: types.CallbackQuery):
    """Показать опции редактирования конкретного заказа."""
    order_id = int(callback.data.split(":")[1])
    await callback.answer()

    try:
        crm = _logist._crm
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
