"""Роутер редактирования состава заказа (FSM).

Обрабатывает callback edit_order:{order_id} и весь диалог правки:
- показывает текущий состав с кнопками [−] [×qty] [+] [🗑]
- позволяет добавить товар из каталога
- применяет изменения в v1 CRM
- пишет запись в «История изменений заказа» (v2 Integram, table 85932)
- уведомляет пчеловода
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Optional

import httpx
from aiogram import Router, Bot, F
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton

from src.config import (
    BEEKEEPER_CHAT_ID,
    INTEGRAM_V2_URL,
    INTEGRAM_V2_EMAIL,
    INTEGRAM_V2_PASSWORD,
    INTEGRAM_V2_WORKSPACE,
)

logger = logging.getLogger(__name__)
router = Router()

# Инициализируются через setup_fsm_edit()
_logist = None   # LogistAgent — доступ к _crm
_bot: Optional[Bot] = None

# ID таблицы «История изменений заказа» в Integram v2
V2_HISTORY_TABLE_ID = 85932


def setup_fsm_edit(logist, bot: Bot) -> None:
    global _logist, _bot
    _logist = logist
    _bot = bot


# ---------------------------------------------------------------------------
# FSM состояния
# ---------------------------------------------------------------------------

class EditFSM(StatesGroup):
    editing     = State()   # главный экран редактора (просмотр + правка)
    adding_item = State()   # каталог: выбор товара для добавления


# ---------------------------------------------------------------------------
# Форматирование и клавиатуры
# ---------------------------------------------------------------------------

def _fmt_items(
    items: list[dict],
    removed_ids: set[int],
    new_items: list[dict],
) -> str:
    lines = []
    for it in items:
        name  = it.get("product_name") or "Товар"
        qty   = it["quantity"]
        price = it.get("unit_price") or 0.0
        iid   = it["id"]
        if iid in removed_ids:
            lines.append(f"🗑 {name} ×{qty} _(будет удалён)_")
        else:
            orig   = it.get("orig_qty")
            suffix = f" _(было ×{orig})_" if orig and orig != qty else ""
            lines.append(f"• {name} ×{qty} — {int(qty * price)} ₽{suffix}")
    for ni in new_items:
        lines.append(
            f"✚ _{ni['name']} ×{ni['qty']} — {int(ni['qty'] * ni['price'])} ₽_ _(новый)_"
        )
    return "\n".join(lines) or "_(состав пуст)_"


def _build_editor_kb(
    order_id: int,
    items: list[dict],
    removed_ids: set[int],
) -> InlineKeyboardMarkup:
    rows = []
    for it in items:
        iid  = it["id"]
        name = (it.get("product_name") or "Товар")[:20]
        qty  = it["quantity"]
        if iid in removed_ids:
            rows.append([
                InlineKeyboardButton(
                    text=f"↩ Вернуть: {name}",
                    callback_data=f"edit_restore:{order_id}:{iid}",
                ),
            ])
        else:
            rows.append([
                InlineKeyboardButton(text="−",    callback_data=f"edit_qdn:{order_id}:{iid}"),
                InlineKeyboardButton(text=f"×{qty}", callback_data="edit_noop"),
                InlineKeyboardButton(text="+",    callback_data=f"edit_qup:{order_id}:{iid}"),
                InlineKeyboardButton(text="🗑",   callback_data=f"edit_rm:{order_id}:{iid}"),
            ])
    rows.append([
        InlineKeyboardButton(text="＋ Добавить товар", callback_data=f"edit_add:{order_id}"),
    ])
    rows.append([
        InlineKeyboardButton(text="✅ Сохранить",  callback_data=f"edit_confirm:{order_id}"),
        InlineKeyboardButton(text="❌ Отмена",     callback_data=f"edit_cancel:{order_id}"),
    ])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _build_catalog_kb(order_id: int, products: list) -> InlineKeyboardMarkup:
    rows = []
    for p in products:
        price_s = f"  {int(p.price)} ₽" if getattr(p, "price", None) else ""
        rows.append([
            InlineKeyboardButton(
                text=f"{p.name}{price_s}",
                callback_data=f"edit_pick:{order_id}:{p.id}",
            )
        ])
    rows.append([
        InlineKeyboardButton(text="← Назад к заказу", callback_data=f"edit_back:{order_id}"),
    ])
    return InlineKeyboardMarkup(inline_keyboard=rows)


async def _refresh_editor(cb: CallbackQuery, data: dict) -> None:
    """Обновить сообщение-редактор на основе текущего состояния."""
    items       = data.get("items") or []
    removed_ids = set(data.get("removed_ids") or [])
    new_items   = data.get("new_items") or []
    order_id    = data["order_id"]

    text = (
        f"✏️ *Редактирование #{data['order_number']}*\n"
        f"Статус: {data['order_status']}\n\n"
        f"{_fmt_items(items, removed_ids, new_items)}"
    )
    kb = _build_editor_kb(order_id, items, removed_ids)
    try:
        await cb.message.edit_text(text, parse_mode="Markdown", reply_markup=kb)
    except Exception:
        await cb.message.answer(text, parse_mode="Markdown", reply_markup=kb)


# ---------------------------------------------------------------------------
# Запись в v2 «История изменений заказа»
# ---------------------------------------------------------------------------

_v2_token: str = ""
_v2_token_ts: float = 0.0


async def _v2_auth(http: httpx.AsyncClient) -> str:
    global _v2_token, _v2_token_ts
    if _v2_token and time.monotonic() - _v2_token_ts < 850:
        return _v2_token
    r = await http.post(
        f"{INTEGRAM_V2_URL}/api/v2/iam/login",
        json={"email": INTEGRAM_V2_EMAIL, "password": INTEGRAM_V2_PASSWORD},
    )
    r.raise_for_status()
    _v2_token = r.json()["accessToken"]
    _v2_token_ts = time.monotonic()
    return _v2_token


async def _write_v2_history(
    order_number: str,
    description: str,
    status_at_change: str,
    who: str,
) -> None:
    if not (INTEGRAM_V2_EMAIL and INTEGRAM_V2_PASSWORD):
        return
    try:
        async with httpx.AsyncClient(timeout=20.0) as http:
            token = await _v2_auth(http)
            now   = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            r = await http.post(
                f"{INTEGRAM_V2_URL}/api/v2/{INTEGRAM_V2_WORKSPACE}/ai/tool",
                headers={"Authorization": f"Bearer {token}"},
                json={
                    "name": "create_object",
                    "args": {
                        "typeId": V2_HISTORY_TABLE_ID,
                        "name":   order_number,
                        "fields": {
                            "Дата изменения":                     now,
                            "Описание изменения":                 f"[{order_number}] {description}",
                            "Статус заказа на момент изменения":  status_at_change,
                            "Кто изменил":                        who,
                        },
                    },
                    "skipHitl": True,
                },
            )
            r.raise_for_status()
            logger.info("v2 история записана для %s", order_number)
    except Exception as exc:
        logger.warning("Не удалось записать v2 историю: %s", exc)


# ---------------------------------------------------------------------------
# Callback: открыть редактор (edit_order:{order_id})
# ---------------------------------------------------------------------------

@router.callback_query(F.data.startswith("edit_order:"))
async def cb_edit_order(cb: CallbackQuery, state: FSMContext) -> None:
    order_id = int(cb.data.split(":")[1])
    await cb.answer()

    crm = getattr(_logist, "_crm", None) if _logist else None
    if not crm:
        await cb.message.answer("CRM недоступна. Попробуйте позже.")
        return

    try:
        order = await crm.get_order(order_id)
        items = await crm.get_order_items(order_id)
    except Exception as exc:
        logger.error("Ошибка загрузки заказа %d: %s", order_id, exc)
        await cb.message.answer("Не удалось загрузить заказ. Попробуйте позже.")
        return

    items_data = [
        {
            "id":           it.id,
            "product_id":   it.product_id,
            "product_name": it.product_name or "Товар",
            "quantity":     it.quantity,
            "unit_price":   it.unit_price or 0.0,
            "orig_qty":     it.quantity,
        }
        for it in items
    ]

    await state.set_state(EditFSM.editing)
    await state.update_data(
        order_id     = order_id,
        order_number = order.number or str(order_id),
        order_status = order.status or "Новый",
        items        = items_data,
        removed_ids  = [],
        new_items    = [],
    )
    await _refresh_editor(cb, await state.get_data())


# ---------------------------------------------------------------------------
# Callbacks: операции с позициями
# ---------------------------------------------------------------------------

@router.callback_query(EditFSM.editing, F.data.startswith("edit_rm:"))
async def cb_edit_remove(cb: CallbackQuery, state: FSMContext) -> None:
    item_id = int(cb.data.split(":")[2])
    await cb.answer()
    data    = await state.get_data()
    removed = set(data.get("removed_ids") or [])
    removed.add(item_id)
    await state.update_data(removed_ids=list(removed))
    await _refresh_editor(cb, await state.get_data())


@router.callback_query(EditFSM.editing, F.data.startswith("edit_restore:"))
async def cb_edit_restore(cb: CallbackQuery, state: FSMContext) -> None:
    item_id = int(cb.data.split(":")[2])
    await cb.answer()
    data    = await state.get_data()
    removed = set(data.get("removed_ids") or [])
    removed.discard(item_id)
    await state.update_data(removed_ids=list(removed))
    await _refresh_editor(cb, await state.get_data())


@router.callback_query(EditFSM.editing, F.data.startswith("edit_qup:"))
async def cb_edit_qty_up(cb: CallbackQuery, state: FSMContext) -> None:
    item_id = int(cb.data.split(":")[2])
    await cb.answer()
    data  = await state.get_data()
    items = data.get("items") or []
    for it in items:
        if it["id"] == item_id:
            it["quantity"] += 1
            break
    await state.update_data(items=items)
    await _refresh_editor(cb, await state.get_data())


@router.callback_query(EditFSM.editing, F.data.startswith("edit_qdn:"))
async def cb_edit_qty_down(cb: CallbackQuery, state: FSMContext) -> None:
    item_id = int(cb.data.split(":")[2])
    await cb.answer()
    data  = await state.get_data()
    items = data.get("items") or []
    for it in items:
        if it["id"] == item_id and it["quantity"] > 1:
            it["quantity"] -= 1
            break
    await state.update_data(items=items)
    await _refresh_editor(cb, await state.get_data())


@router.callback_query(EditFSM.editing, F.data == "edit_noop")
async def cb_noop(cb: CallbackQuery) -> None:
    await cb.answer()


# ---------------------------------------------------------------------------
# Callbacks: добавление товара из каталога
# ---------------------------------------------------------------------------

@router.callback_query(EditFSM.editing, F.data.startswith("edit_add:"))
async def cb_edit_add(cb: CallbackQuery, state: FSMContext) -> None:
    order_id = int(cb.data.split(":")[1])
    await cb.answer()

    crm = getattr(_logist, "_crm", None) if _logist else None
    if not crm:
        await cb.answer("CRM недоступна", show_alert=True)
        return

    try:
        products = await crm.get_products(in_stock_only=True)
    except Exception:
        products = []

    if not products:
        await cb.answer("Каталог товаров недоступен", show_alert=True)
        return

    await state.set_state(EditFSM.adding_item)
    await state.update_data(products=[
        {"id": p.id, "name": p.name, "price": getattr(p, "price", 0.0) or 0.0}
        for p in products
    ])

    try:
        await cb.message.edit_text(
            "📦 *Выберите товар для добавления:*",
            parse_mode="Markdown",
            reply_markup=_build_catalog_kb(order_id, products),
        )
    except Exception:
        await cb.message.answer(
            "📦 *Выберите товар для добавления:*",
            parse_mode="Markdown",
            reply_markup=_build_catalog_kb(order_id, products),
        )


@router.callback_query(EditFSM.adding_item, F.data.startswith("edit_pick:"))
async def cb_edit_pick(cb: CallbackQuery, state: FSMContext) -> None:
    parts      = cb.data.split(":")
    order_id   = int(parts[1])
    product_id = int(parts[2])
    await cb.answer()

    data     = await state.get_data()
    products = data.get("products") or []
    prod     = next((p for p in products if p["id"] == product_id), None)
    if not prod:
        await cb.answer("Товар не найден", show_alert=True)
        return

    new_items = list(data.get("new_items") or [])
    existing  = next((ni for ni in new_items if ni["product_id"] == product_id), None)
    if existing:
        existing["qty"] += 1
    else:
        new_items.append({
            "product_id": product_id,
            "name":       prod["name"],
            "qty":        1,
            "price":      prod["price"],
        })

    await state.update_data(new_items=new_items)
    await state.set_state(EditFSM.editing)
    await _refresh_editor(cb, await state.get_data())


@router.callback_query(EditFSM.adding_item, F.data.startswith("edit_back:"))
async def cb_edit_back(cb: CallbackQuery, state: FSMContext) -> None:
    await cb.answer()
    await state.set_state(EditFSM.editing)
    await _refresh_editor(cb, await state.get_data())


# ---------------------------------------------------------------------------
# Callback: отмена
# ---------------------------------------------------------------------------

@router.callback_query(F.data.startswith("edit_cancel:"))
async def cb_edit_cancel(cb: CallbackQuery, state: FSMContext) -> None:
    await cb.answer("Редактирование отменено.")
    await state.clear()
    try:
        await cb.message.edit_text("Редактирование отменено.")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Callback: сохранить изменения
# ---------------------------------------------------------------------------

@router.callback_query(EditFSM.editing, F.data.startswith("edit_confirm:"))
async def cb_edit_confirm(cb: CallbackQuery, state: FSMContext) -> None:
    order_id = int(cb.data.split(":")[1])
    await cb.answer()

    data         = await state.get_data()
    items        = data.get("items") or []
    removed      = set(data.get("removed_ids") or [])
    new_items    = data.get("new_items") or []
    order_number = data.get("order_number", str(order_id))
    order_status = data.get("order_status", "Новый")

    has_changes = (
        bool(removed)
        or bool(new_items)
        or any(it["quantity"] != it.get("orig_qty", it["quantity"]) for it in items)
    )
    if not has_changes:
        await cb.answer("Нет изменений для сохранения.", show_alert=True)
        return

    crm = getattr(_logist, "_crm", None) if _logist else None
    if not crm:
        await cb.answer("CRM недоступна", show_alert=True)
        return

    try:
        change_parts: list[str] = []

        # Удалить помеченные позиции
        for it in items:
            if it["id"] in removed:
                await crm.delete_order_item(it["id"])
                change_parts.append(
                    f"Удалён: {it['product_name']} ×{it.get('orig_qty') or it['quantity']}"
                )

        # Изменить количество существующих позиций
        for it in items:
            if it["id"] not in removed:
                orig = it.get("orig_qty", it["quantity"])
                if it["quantity"] != orig:
                    await crm.update_order_item(it["id"], qty=it["quantity"])
                    change_parts.append(
                        f"Кол-во: {it['product_name']} {orig}→{it['quantity']}"
                    )

        # Добавить новые позиции
        for ni in new_items:
            await crm.add_order_item(order_id, ni["product_id"], ni["qty"], ni["price"])
            change_parts.append(f"Добавлен: {ni['name']} ×{ni['qty']}")

        # Пересчитать итоги заказа
        await crm.recalculate_order_totals(order_id)

        # Записать в v2 «История изменений заказа»
        await _write_v2_history(
            order_number     = order_number,
            description      = "; ".join(change_parts),
            status_at_change = order_status,
            who              = f"Клиент (tg:{cb.from_user.id})",
        )

        # Уведомить пчеловода
        if _bot and BEEKEEPER_CHAT_ID:
            notif = (
                f"✏️ *Заказ {order_number} изменён клиентом*\n\n"
                + "\n".join(f"• {c}" for c in change_parts)
            )
            try:
                await _bot.send_message(BEEKEEPER_CHAT_ID, notif, parse_mode="Markdown")
            except Exception as exc:
                logger.warning("Уведомление пчеловоду не отправлено: %s", exc)

        await state.clear()
        result_text = (
            f"✅ *Заказ {order_number} обновлён.*\n\n"
            + "\n".join(f"• {c}" for c in change_parts)
        )
        try:
            await cb.message.edit_text(result_text, parse_mode="Markdown")
        except Exception:
            await cb.message.answer(result_text, parse_mode="Markdown")

    except Exception as exc:
        logger.error("Ошибка применения изменений заказа %d: %s", order_id, exc)
        await cb.message.answer(
            "Не удалось применить изменения. Попробуйте позже или напишите Александру."
        )
