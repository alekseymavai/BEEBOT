"""Админ-команды BEEBOT — управление заказами из Telegram.

Доступны только пользователям с ADMIN_CHAT_ID / BEEKEEPER_CHAT_ID.

Команды:
  /orders [статус]       — список заказов (фильтр: новый, подтверждён, в сборке, отправлен)
  /order <id>            — детали заказа
  /status <id> <статус>  — сменить статус заказа
  /track <id> <трек>     — добавить трек-номер + уведомить клиента
  /clients               — список клиентов
  /stock                 — остатки товаров
"""

from __future__ import annotations

import logging
from typing import Optional

from aiogram import Router, types, F, Bot
from aiogram.filters import Command
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton

from src.integram_client import IntegramClient, IntegramError, IntegramNotFoundError
from src.services.auth_service import AuthService
from src.knowledge_base import KnowledgeBase
from src.memory import UserMemory
from src.notifications import Notifier
from src.web.notifications import notify_client_status_change, notify_beekeeper_status_change

logger = logging.getLogger(__name__)

router = Router()

# Статусы заказов (для справки и подсказок)
ORDER_STATUSES = ["Новый", "Подтверждён", "В сборке", "Отправлен", "Доставлен", "Отменён"]

# Глобальные ссылки — устанавливаются при инициализации (setup_admin)
_crm: Optional[IntegramClient] = None
_notifier: Optional[Notifier] = None
_bot: Optional[Bot] = None
_kb: Optional[KnowledgeBase] = None
_mem: Optional[UserMemory] = None
_auth: Optional[AuthService] = None


def setup_admin(
    bot: Bot,
    crm: Optional[IntegramClient] = None,
    kb: Optional[KnowledgeBase] = None,
    memory: Optional[UserMemory] = None,
    auth: Optional[AuthService] = None,
) -> None:
    """Инициализировать админ-модуль с зависимостями."""
    global _crm, _notifier, _bot, _kb, _mem, _auth
    _bot = bot
    _crm = crm
    _notifier = Notifier(bot)
    _kb = kb
    _mem = memory
    _auth = auth


def _is_admin(user_id: int) -> bool:
    """Проверить, является ли пользователь администратором."""
    if _auth:
        return _auth.is_admin_legacy(user_id)
    # Fallback без AuthService (обратная совместимость)
    from src.config import ADMIN_CHAT_ID, BEEKEEPER_CHAT_ID
    admin_ids = {id_ for id_ in (ADMIN_CHAT_ID, BEEKEEPER_CHAT_ID) if id_ is not None}
    return user_id in admin_ids


# ---------------------------------------------------------------------------
# /orders [статус] — список заказов
# ---------------------------------------------------------------------------


@router.message(Command("orders"))
async def cmd_orders(message: types.Message) -> None:
    """Список заказов с фильтрацией по статусу."""
    if not _is_admin(message.from_user.id):
        await message.answer("Команда доступна только администратору.")
        return

    if not _crm:
        await message.answer("CRM не подключена. Проверьте настройки INTEGRAM_* в .env.")
        return

    # Парсить аргумент: /orders подтверждён
    args = (message.text or "").removeprefix("/orders").strip()
    status_filter = args if args else None

    try:
        await _crm.authenticate()
        orders = await _crm.get_orders(status=status_filter)
    except IntegramError as e:
        await message.answer(f"Ошибка CRM: {e}")
        return

    if not orders:
        label = f" со статусом «{status_filter}»" if status_filter else ""
        await message.answer(f"Заказов{label} не найдено.")
        return

    lines = [f"📋 *Заказы* ({len(orders)} шт.)\n"]
    for o in orders[:20]:  # ограничение на 20
        total_str = f"{o.total:.0f} ₽" if o.total else "—"
        client = o.client_name or f"клиент #{o.client_id}"
        lines.append(
            f"  `#{o.number}` | {o.status} | {client} | {total_str}"
        )

    if len(orders) > 20:
        lines.append(f"\n... и ещё {len(orders) - 20}")

    lines.append("\nДетали: /order <номер>")
    await message.answer("\n".join(lines), parse_mode="Markdown")


# ---------------------------------------------------------------------------
# /order <id> — детали заказа
# ---------------------------------------------------------------------------


@router.message(Command("order"))
async def cmd_order_detail(message: types.Message) -> None:
    """Подробная информация о заказе."""
    if not _is_admin(message.from_user.id):
        await message.answer("Команда доступна только администратору.")
        return

    if not _crm:
        await message.answer("CRM не подключена.")
        return

    args = (message.text or "").removeprefix("/order").strip()
    if not args:
        await message.answer("Использование: /order <ID заказа>")
        return

    try:
        order_id = int(args.replace("#", ""))
    except ValueError:
        await message.answer("ID заказа должен быть числом.")
        return

    try:
        await _crm.authenticate()
        order = await _crm.get_order(order_id)
        order.items = await _crm.get_order_items(order_id)
    except IntegramNotFoundError:
        await message.answer(f"Заказ #{order_id} не найден.")
        return
    except IntegramError as e:
        await message.answer(f"Ошибка CRM: {e}")
        return

    # Форматировать детали
    items_str = ""
    if order.items:
        items_lines = []
        for item in order.items:
            name = item.product_name or f"товар #{item.product_id}"
            items_lines.append(f"  • {name} × {item.quantity} = {item.total:.0f} ₽")
        items_str = "\n".join(items_lines)
    else:
        items_str = "  (нет позиций)"

    total_str = f"{order.total:.0f} ₽" if order.total else "—"
    delivery_cost_str = f"{order.delivery_cost:.0f} ₽" if order.delivery_cost else "—"
    track_str = f"`{order.tracking_number}`" if order.tracking_number else "—"
    client = order.client_name or f"клиент #{order.client_id}"

    text = (
        f"📦 *Заказ #{order.number}*\n\n"
        f"👤 {client}\n"
        f"📊 Статус: *{order.status}*\n"
        f"📅 Дата: {order.date.strftime('%d.%m.%Y %H:%M')}\n"
        f"🚚 Доставка: {order.delivery_method or '—'} ({delivery_cost_str})\n"
        f"🏠 Адрес: {order.delivery_address or '—'}\n"
        f"📮 Трек: {track_str}\n\n"
        f"*Товары:*\n{items_str}\n\n"
        f"💰 *Итого: {total_str}*"
    )

    # Кнопки действий в зависимости от статуса
    buttons = _order_action_buttons(order.id, order.status)
    keyboard = InlineKeyboardMarkup(inline_keyboard=buttons) if buttons else None

    await message.answer(text, parse_mode="Markdown", reply_markup=keyboard)


def _order_action_buttons(order_id: int, current_status: str) -> list[list[InlineKeyboardButton]]:
    """Сгенерировать кнопки действий для заказа на основе текущего статуса."""
    rows = []
    if current_status == "Новый":
        rows.append([
            InlineKeyboardButton(text="✅ Подтвердить", callback_data=f"admin:confirm:{order_id}"),
            InlineKeyboardButton(text="❌ Отклонить", callback_data=f"admin:reject:{order_id}"),
        ])
    elif current_status == "Подтверждён":
        rows.append([
            InlineKeyboardButton(text="📦 В сборку", callback_data=f"admin:assemble:{order_id}"),
        ])
    elif current_status == "В сборке":
        rows.append([
            InlineKeyboardButton(text="🚚 Отправлен", callback_data=f"admin:ship:{order_id}"),
        ])
    return rows


# ---------------------------------------------------------------------------
# /status <id> <статус> — сменить статус заказа
# ---------------------------------------------------------------------------


@router.message(Command("status"))
async def cmd_status(message: types.Message) -> None:
    """Сменить статус заказа."""
    if not _is_admin(message.from_user.id):
        await message.answer("Команда доступна только администратору.")
        return

    if not _crm:
        await message.answer("CRM не подключена.")
        return

    args = (message.text or "").removeprefix("/status").strip()
    parts = args.split(maxsplit=1)

    if len(parts) < 2:
        statuses_str = ", ".join(ORDER_STATUSES)
        await message.answer(
            f"Использование: /status <ID> <статус>\n\n"
            f"Доступные статусы: {statuses_str}"
        )
        return

    try:
        order_id = int(parts[0].replace("#", ""))
    except ValueError:
        await message.answer("ID заказа должен быть числом.")
        return

    new_status = parts[1].strip()
    # Нечёткое сопоставление статуса
    matched = _match_status(new_status)
    if not matched:
        statuses_str = ", ".join(ORDER_STATUSES)
        await message.answer(f"Неизвестный статус «{new_status}».\nДоступные: {statuses_str}")
        return

    try:
        await _crm.authenticate()
        await _crm.update_order_status(order_id, matched)
        await message.answer(f"✅ Заказ #{order_id} → *{matched}*", parse_mode="Markdown")
        await _notify_status_change(order_id, matched)
    except IntegramError as e:
        await message.answer(f"Ошибка: {e}")


# ---------------------------------------------------------------------------
# /track <id> <трек-номер> — добавить трек и уведомить клиента
# ---------------------------------------------------------------------------


@router.message(Command("track"))
async def cmd_track(message: types.Message) -> None:
    """Добавить трек-номер к заказу и уведомить клиента."""
    if not _is_admin(message.from_user.id):
        await message.answer("Команда доступна только администратору.")
        return

    if not _crm:
        await message.answer("CRM не подключена.")
        return

    args = (message.text or "").removeprefix("/track").strip()
    parts = args.split(maxsplit=1)

    if len(parts) < 2:
        await message.answer("Использование: /track <ID заказа> <трек-номер>")
        return

    try:
        order_id = int(parts[0].replace("#", ""))
    except ValueError:
        await message.answer("ID заказа должен быть числом.")
        return

    tracking = parts[1].strip()

    try:
        await _crm.authenticate()
        # Обновить трек-номер и статус
        await _crm._request(
            "PATCH",
            f"/api/orders/{order_id}",
            json={"Трек-номер": tracking, "Статус": "Отправлен"},
        )
        await message.answer(
            f"✅ Заказ #{order_id} → *Отправлен*\n"
            f"Трек: `{tracking}`",
            parse_mode="Markdown",
        )

        # Уведомить клиента
        if _notifier:
            try:
                order = await _crm.get_order(order_id)
                # Найти telegram_id клиента
                client_tg_id = await _get_client_telegram_id(order.client_id)
                if client_tg_id:
                    await _notifier.order_shipped(order_id, tracking, client_tg_id)
                    await message.answer(f"📨 Клиент уведомлён (Telegram ID: {client_tg_id}).")
            except Exception as e:
                logger.warning("Не удалось уведомить клиента: %s", e)

    except IntegramError as e:
        await message.answer(f"Ошибка: {e}")


# ---------------------------------------------------------------------------
# /clients — список клиентов
# ---------------------------------------------------------------------------


@router.message(Command("clients"))
async def cmd_clients(message: types.Message) -> None:
    """Список клиентов из CRM."""
    if not _is_admin(message.from_user.id):
        await message.answer("Команда доступна только администратору.")
        return

    if not _crm:
        await message.answer("CRM не подключена.")
        return

    try:
        await _crm.authenticate()
        data = await _crm._request("GET", "/api/clients")
        clients = data if isinstance(data, list) else data.get("items", data.get("data", []))
    except IntegramError as e:
        await message.answer(f"Ошибка CRM: {e}")
        return

    if not clients:
        await message.answer("Клиентов не найдено.")
        return

    lines = [f"👥 *Клиенты* ({len(clients)} чел.)\n"]
    for c in clients[:25]:
        name = c.get("ФИО") or c.get("full_name", "—")
        phone = c.get("Телефон") or c.get("phone", "")
        tg = c.get("Telegram Username") or ""
        phone_str = f" | {phone}" if phone else ""
        tg_str = f" | @{tg}" if tg else ""
        lines.append(f"  • {name}{phone_str}{tg_str}")

    if len(clients) > 25:
        lines.append(f"\n... и ещё {len(clients) - 25}")

    await message.answer("\n".join(lines), parse_mode="Markdown")


# ---------------------------------------------------------------------------
# /stock — остатки товаров
# ---------------------------------------------------------------------------


@router.message(Command("stock"))
async def cmd_stock(message: types.Message) -> None:
    """Показать товары из каталога CRM."""
    if not _is_admin(message.from_user.id):
        await message.answer("Команда доступна только администратору.")
        return

    if not _crm:
        await message.answer("CRM не подключена.")
        return

    try:
        await _crm.authenticate()
        products = await _crm.get_products(in_stock_only=False)
    except IntegramError as e:
        await message.answer(f"Ошибка CRM: {e}")
        return

    if not products:
        await message.answer("Товаров в каталоге нет.")
        return

    lines = [f"📦 *Каталог* ({len(products)} товаров)\n"]
    for p in products:
        price_str = f"{p.price:.0f} ₽" if p.price else "—"
        stock = "✅" if p.in_stock else "❌"
        weight_str = f" · {p.weight}г" if p.weight else ""
        lines.append(f"  {stock} {p.name}{weight_str} — {price_str}")

    await message.answer("\n".join(lines), parse_mode="Markdown")


# ---------------------------------------------------------------------------
# /teach <текст> — добавить знание в базу
# ---------------------------------------------------------------------------


@router.message(Command("teach", ignore_mention=True))
async def cmd_teach(message: types.Message) -> None:
    """Добавить новый чанк в базу знаний без полной перестройки."""
    if not _is_admin(message.from_user.id):
        await message.answer("Команда доступна только администратору.")
        return

    if _kb is None:
        await message.answer("База знаний не загружена.")
        return

    text = (message.text or "").strip()
    # Убираем команду с учётом возможного @bot_username
    for prefix in ("/teach", "/teach@"):
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].lstrip()
            # Убрать имя бота (всё до первого пробела), если оно есть
            if prefix.endswith("@") and " " in text:
                _, text = text.split(" ", 1)
            break

    text = text.strip()

    if not text:
        await message.answer(
            "Использование: /teach <текст>\n\n"
            "Пример: /teach Мёд акациевый собирается в июне-июле, имеет светлый цвет и нежный вкус."
        )
        return

    try:
        chunk = _kb.teach(text, source=f"admin:teach:{message.from_user.id}")
        total = len(_kb.chunks)
        await message.answer(
            f"✅ Знание добавлено в базу!\n"
            f"📌 Источник: `{chunk['source']}`\n"
            f"🔢 Чанков в индексе: {total}",
            parse_mode="Markdown",
        )
    except Exception as e:
        logger.error("Ошибка /teach: %s", e)
        await message.answer(f"❌ Ошибка при добавлении знания: {e}")


# ---------------------------------------------------------------------------
# /remember <telegram_id> <факт> — добавить/посмотреть/удалить факт о пользователе
# ---------------------------------------------------------------------------


@router.message(Command("remember", ignore_mention=True))
async def cmd_remember(message: types.Message) -> None:
    """Управление долгосрочной памятью пользователей.

    Использование:
      /remember <telegram_id> <факт>    — добавить факт
      /remember <telegram_id>           — показать все факты
      /remember <telegram_id> clear     — удалить все факты
    """
    if not _is_admin(message.from_user.id):
        await message.answer("Команда доступна только администратору.")
        return

    if _mem is None:
        await message.answer("Модуль памяти не инициализирован.")
        return

    args = (message.text or "").strip()
    for prefix in ("/remember@", "/remember"):
        if args.lower().startswith(prefix.lower()):
            args = args[len(prefix):].strip()
            break

    if not args:
        await message.answer(
            "Использование:\n"
            "/remember <telegram_id> <факт> — добавить факт\n"
            "/remember <telegram_id> — показать все факты\n"
            "/remember <telegram_id> clear — удалить все факты\n\n"
            "Пример: /remember 123456789 язва желудка, интересуется пергой"
        )
        return

    parts = args.split(maxsplit=1)
    try:
        tg_id = int(parts[0])
    except ValueError:
        await message.answer("Первый аргумент — Telegram ID (число).")
        return

    # Только ID → показать факты
    if len(parts) == 1:
        facts = _mem.get_facts(tg_id)
        if not facts:
            await message.answer(f"О пользователе {tg_id} ничего не сохранено.")
        else:
            lines = [f"🧠 Память о пользователе `{tg_id}` ({len(facts)} фактов):\n"]
            for i, f in enumerate(facts, 1):
                lines.append(f"{i}. {f}")
            await message.answer("\n".join(lines), parse_mode="Markdown")
        return

    fact_text = parts[1].strip()

    # clear → удалить все факты
    if fact_text.lower() == "clear":
        count = _mem.clear_facts(tg_id)
        await message.answer(f"🗑 Удалено {count} фактов о пользователе {tg_id}.")
        return

    # Добавить факт
    added = _mem.add_fact(tg_id, fact_text, category="general", source="admin")
    if added:
        total = _mem.count_facts(tg_id)
        await message.answer(
            f"✅ Факт добавлен для пользователя `{tg_id}`.\n"
            f"Всего фактов: {total}",
            parse_mode="Markdown",
        )
    else:
        await message.answer("Этот факт уже сохранён.")


# ---------------------------------------------------------------------------
# Inline-кнопки: подтверждение/отклонение/сборка/отправка
# ---------------------------------------------------------------------------


@router.callback_query(F.data.startswith("admin:confirm:"))
async def cb_admin_confirm(callback: types.CallbackQuery) -> None:
    """Подтвердить заказ (кнопка)."""
    if not _is_admin(callback.from_user.id):
        await callback.answer("Нет доступа", show_alert=True)
        return
    await _change_order_status_cb(callback, "Подтверждён")


@router.callback_query(F.data.startswith("admin:reject:"))
async def cb_admin_reject(callback: types.CallbackQuery) -> None:
    """Отклонить заказ (кнопка)."""
    if not _is_admin(callback.from_user.id):
        await callback.answer("Нет доступа", show_alert=True)
        return
    await _change_order_status_cb(callback, "Отменён")


@router.callback_query(F.data.startswith("admin:assemble:"))
async def cb_admin_assemble(callback: types.CallbackQuery) -> None:
    """В сборку (кнопка)."""
    if not _is_admin(callback.from_user.id):
        await callback.answer("Нет доступа", show_alert=True)
        return
    await _change_order_status_cb(callback, "В сборке")


@router.callback_query(F.data.startswith("admin:ship:"))
async def cb_admin_ship(callback: types.CallbackQuery) -> None:
    """Отправлен (кнопка)."""
    if not _is_admin(callback.from_user.id):
        await callback.answer("Нет доступа", show_alert=True)
        return
    await _change_order_status_cb(callback, "Отправлен")


# Обработка кнопок из Notifier (order_confirm/order_reject)
@router.callback_query(F.data.startswith("order_confirm:"))
async def cb_notifier_confirm(callback: types.CallbackQuery) -> None:
    """Подтвердить заказ (кнопка из уведомления Notifier)."""
    if not _is_admin(callback.from_user.id):
        await callback.answer("Нет доступа", show_alert=True)
        return
    await _change_order_status_cb(callback, "Подтверждён", prefix="order_confirm:")


@router.callback_query(F.data.startswith("order_reject:"))
async def cb_notifier_reject(callback: types.CallbackQuery) -> None:
    """Отклонить заказ (кнопка из уведомления Notifier)."""
    if not _is_admin(callback.from_user.id):
        await callback.answer("Нет доступа", show_alert=True)
        return
    await _change_order_status_cb(callback, "Отменён", prefix="order_reject:")


async def _change_order_status_cb(
    callback: types.CallbackQuery,
    new_status: str,
    prefix: Optional[str] = None,
) -> None:
    """Общая логика смены статуса по callback-кнопке."""
    if not _crm:
        await callback.answer("CRM не подключена", show_alert=True)
        return

    try:
        if prefix:
            order_id = int(callback.data.removeprefix(prefix))
        else:
            # admin:action:id
            order_id = int(callback.data.split(":")[-1])
    except (ValueError, IndexError):
        await callback.answer("Ошибка ID заказа", show_alert=True)
        return

    try:
        await _crm.authenticate()
        await _crm.update_order_status(order_id, new_status)
        await callback.answer(f"Заказ #{order_id} → {new_status}")

        # Обновить сообщение — убрать кнопки, добавить новый статус
        old_text = callback.message.text or ""
        await callback.message.edit_text(
            f"{old_text}\n\n✅ Статус: *{new_status}*",
            parse_mode="Markdown",
        )

        # Уведомить клиента и пчеловода о смене статуса
        await _notify_status_change(order_id, new_status)

    except IntegramError as e:
        await callback.answer(f"Ошибка: {e}", show_alert=True)


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------


def _match_status(user_input: str) -> Optional[str]:
    """Нечёткое сопоставление статуса заказа."""
    normalized = user_input.lower().strip()
    for status in ORDER_STATUSES:
        if status.lower() == normalized:
            return status
    # Частичное совпадение
    aliases = {
        "новый": "Новый",
        "подтвержд": "Подтверждён",
        "подтверждён": "Подтверждён",
        "подтвержден": "Подтверждён",
        "сборк": "В сборке",
        "в сборке": "В сборке",
        "отправлен": "Отправлен",
        "доставлен": "Доставлен",
        "отменён": "Отменён",
        "отменен": "Отменён",
        "отмена": "Отменён",
    }
    for key, val in aliases.items():
        if key in normalized:
            return val
    return None


async def _get_client_telegram_id(client_id: int) -> Optional[int]:
    """Получить Telegram ID клиента по ID в CRM."""
    if not _crm:
        return None
    try:
        data = await _crm._request("GET", f"/api/clients/{client_id}")
        return data.get("Telegram ID") or data.get("telegram_id")
    except Exception:
        return None


async def _get_client_telegram_id_from_order(order_id: int) -> Optional[int]:
    """Получить Telegram ID клиента из заказа."""
    if not _crm:
        return None
    try:
        order = await _crm.get_order(order_id)
        return await _get_client_telegram_id(order.client_id)
    except Exception:
        return None


async def _notify_status_change(order_id: int, new_status: str) -> None:
    """Уведомить клиента и пчеловода о смене статуса заказа.

    Использует httpx-функции из src.web.notifications, которые покрывают
    все статусы жизненного цикла заказа.
    """
    if not _crm:
        return
    try:
        order = await _crm.get_order(order_id)
        order_number = order.number or str(order_id)
        client_name = getattr(order, "client_name", "") or ""
        tracking = getattr(order, "tracking_number", None)

        # Уведомить клиента
        tg_id = await _get_client_telegram_id(order.client_id)
        if tg_id:
            await notify_client_status_change(
                telegram_id=tg_id,
                order_number=order_number,
                new_status=new_status,
                tracking_number=tracking,
            )

        # Уведомить пчеловода
        await notify_beekeeper_status_change(
            order_number=order_number,
            new_status=new_status,
            client_name=client_name,
            tracking_number=tracking,
        )
    except Exception as e:
        logger.warning("Ошибка уведомления при смене статуса #%d → %s: %s", order_id, new_status, e)
