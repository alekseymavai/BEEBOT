"""Агент «Ассистент пчеловода» — прямой LLM-диалог с CRM-контекстом.

Активируется командой /admin для ADMIN_CHAT_ID.
Получает текущие данные CRM и отвечает как персональный бизнес-ассистент.
"""

import asyncio
import logging
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)


def _valid_date(order) -> bool:
    """Вернуть True если у заказа есть валидная дата (не epoch 1970)."""
    try:
        dt = order.date if isinstance(order.date, datetime) else datetime.fromisoformat(str(order.date))
        return dt.year >= 2020
    except Exception:
        return False


_SYSTEM = """Ты — личный ассистент пчеловода Александра Дмитрова, владельца «Усадьба Дмитровых».

Текущее состояние бизнеса (данные из CRM):
{crm_context}

Сегодня: {today}

Ты умеешь:
- Анализировать заказы, продажи, клиентов
- Отвечать на вопросы по статистике CRM
- Составлять тексты: ответы клиентам, посты, объявления
- Давать советы по развитию бизнеса
- Обсуждать любые вопросы

Правила:
- Отвечай только на русском языке
- Данные CRM выше — это АКТУАЛЬНЫЙ снимок, используй их напрямую
- НЕ говори что у тебя "нет доступа к CRM" — данные уже есть в контексте
- Если конкретной информации нет в снимке — скажи «в снимке этих данных нет»
- Отвечай кратко и по делу
- В поле «товары» для заказов может стоять «нет позиций в CRM» — это значит что позиции не были сохранены в базу при создании заказа (старые или ручные заказы). Это факт про данные, не ограничение твоих возможностей.
- UDS-заказы не содержат разбивку по товарам — только итоговую сумму
"""


class AdminChatAgent:
    """Персональный ассистент для ADMIN_CHAT_ID с доступом к данным CRM."""

    def __init__(self, groq_client=None, model: str = "", crm=None):
        self._groq = groq_client
        self._model = model
        self._crm = crm
        # История диалога: user_id → [{role, content}, ...]
        self._history: dict[int, list[dict]] = {}

    def set_crm(self, crm) -> None:
        self._crm = crm

    @staticmethod
    def _format_context(
        orders: list,
        products: list,
        clients_count: int,
        items_by_order: "dict | None" = None,
        snapshot_age: "str | None" = None,
    ) -> str:
        """Форматировать данные CRM в текст для системного промпта.

        items_by_order — dict[order_id → list[items]] при live-загрузке;
        если None — позиции берутся из атрибута order.items (snapshot).
        """
        from collections import Counter

        lines = []
        _RU = {"01": "Янв", "02": "Фев", "03": "Мар", "04": "Апр", "05": "Май", "06": "Июн",
               "07": "Июл", "08": "Авг", "09": "Сен", "10": "Окт", "11": "Ноя", "12": "Дек"}

        # --- Заказы ---
        active = [o for o in orders if o.status in ("Новый", "Подтверждён", "В сборке")]
        revenue_total = sum((o.total or 0) for o in orders)
        lines.append(f"ЗАКАЗЫ: {len(orders)} всего, {len(active)} активных, выручка {revenue_total:,.0f} ₽")

        # Помесячная статистика (только заказы с валидной датой > 2020)
        monthly: dict[str, dict] = defaultdict(lambda: {"count": 0, "revenue": 0.0})
        for o in orders:
            try:
                dt = o.date if isinstance(o.date, datetime) else datetime.fromisoformat(str(o.date))
                if dt.year < 2020:
                    continue
                key = dt.strftime("%Y-%m")
                monthly[key]["count"] += 1
                monthly[key]["revenue"] += o.total or 0
            except Exception:
                continue
        if monthly:
            lines.append("Статистика по месяцам:")
            for ym in sorted(monthly)[-6:]:
                y, m = ym.split("-")
                d = monthly[ym]
                lines.append(f"  {_RU.get(m, m)} {y}: {d['count']} заказ., {d['revenue']:,.0f} ₽")

        # Подневная статистика за последние 7 дней (по источникам)
        valid = [o for o in orders if _valid_date(o)]
        today_date = datetime.now().date()
        daily: dict[str, dict] = defaultdict(lambda: {"count": 0, "revenue": 0.0, "sources": defaultdict(int)})
        for o in valid:
            try:
                dt = o.date if isinstance(o.date, datetime) else datetime.fromisoformat(str(o.date))
                delta = (today_date - dt.date()).days
                if 0 <= delta <= 6:
                    key = dt.strftime("%d.%m")
                    daily[key]["count"] += 1
                    daily[key]["revenue"] += o.total or 0
                    daily[key]["sources"][o.source or "неизвестно"] += 1
            except Exception:
                continue
        if daily:
            lines.append("Заказы за последние 7 дней (дата | кол-во | выручка | источники):")
            for day in sorted(daily, key=lambda d: datetime.strptime(d, "%d.%m").replace(year=today_date.year), reverse=True):
                d = daily[day]
                src_str = ", ".join(f"{s}: {n}" for s, n in sorted(d["sources"].items(), key=lambda x: -x[1]))
                lines.append(f"  {day}: {d['count']} заказ., {d['revenue']:,.0f} ₽ | {src_str}")

        # Последние 20 заказов
        recent = list(reversed(valid[-20:])) if valid else []
        if recent:
            lines.append("Последние 20 заказов (номер | дата | статус | источник | сумма | товары):")
            for o in recent:
                try:
                    dt = o.date if isinstance(o.date, datetime) else datetime.fromisoformat(str(o.date))
                    date_str = dt.strftime("%d.%m.%Y")
                except Exception:
                    date_str = "—"
                if items_by_order is not None:
                    order_items = items_by_order.get(o.id, [])
                else:
                    order_items = getattr(o, "items", []) or []
                items_str = (
                    ", ".join(f"{i.product_name or 'позиция'} ×{i.quantity}" for i in order_items)
                    if order_items else "нет позиций в CRM"
                )
                lines.append(f"  #{o.number} | {date_str} | {o.status} | {o.source or '—'} | {o.total or 0:.0f} ₽ | {items_str}")

        # Топ-10 товаров
        product_counts: Counter = Counter()
        if items_by_order is not None:
            for items in items_by_order.values():
                for item in items:
                    product_counts[item.product_name or "неизвестно"] += item.quantity
        else:
            for o in orders:
                for item in (getattr(o, "items", []) or []):
                    product_counts[item.product_name or "неизвестно"] += item.quantity
        top = product_counts.most_common(10)
        if top:
            lines.append("Топ-10 товаров по суммарному количеству в заказах:")
            for name, qty in top:
                lines.append(f"  {name}: {qty} шт.")

        # --- Товары / склад ---
        low = [p for p in products if p.stock is not None and p.stock < 5]
        lines.append(f"ТОВАРЫ: {len(products)} позиций")
        if low:
            lines.append("Мало на складе (<5 шт.):")
            for p in low[:7]:
                lines.append(f"  {p.name}: {p.stock} шт.")

        # --- Клиенты ---
        lines.append(f"КЛИЕНТЫ: {clients_count} в базе")

        if snapshot_age:
            lines.append(f"\n(Снимок CRM: {snapshot_age})")
        return "\n".join(lines)

    async def _get_crm_context(self) -> str:
        """Собрать снимок состояния CRM в виде текста для системного промпта (live-запросы)."""
        if not self._crm:
            return "CRM не подключена."

        try:
            orders, products, clients = await asyncio.gather(
                self._crm.get_orders(),
                self._crm.get_products(),
                self._crm.get_clients(),
            )
        except Exception as e:
            return f"Ошибка загрузки CRM: {e}"

        # Загружаем позиции для последних 20 заказов параллельно
        valid = [o for o in orders if _valid_date(o)]
        recent = list(reversed(valid[-20:])) if valid else []

        async def _safe_items(order_id: int):
            try:
                return await self._crm.get_order_items(order_id)
            except Exception:
                return []

        items_lists = await asyncio.gather(*[_safe_items(o.id) for o in recent])
        items_by_order = {o.id: items for o, items in zip(recent, items_lists)}

        # Bulk-запрос для топ-10 (все позиции)
        try:
            all_items = await self._crm.get_order_items_bulk()
            for item in all_items:
                oid = getattr(item, "order_id", None)
                if oid is not None and oid not in items_by_order:
                    items_by_order.setdefault(oid, []).append(item)
        except Exception:
            pass

        return self._format_context(orders, products, len(clients), items_by_order=items_by_order)

    def _call_llm(self, messages: list[dict]) -> str:
        """Синхронный вызов Groq (совместим с существующим LLMClient)."""
        for attempt in range(3):
            try:
                resp = self._groq.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    max_tokens=1000,
                    temperature=0.6,
                )
                return resp.choices[0].message.content
            except Exception as e:
                logger.error("AdminChat LLM error (attempt %d/3): %s", attempt + 1, e)
                if attempt < 2:
                    time.sleep(2 ** attempt)
        return "⚠️ Не удалось получить ответ. Попробуй ещё раз."

    async def chat(self, user_id: int, message: str, snapshot=None) -> str:
        """Обработать сообщение пчеловода и вернуть ответ.

        snapshot: CrmSnapshot — если передан и актуален, использует
                  предзагруженные данные вместо live-запросов к CRM.
        """
        if not self._groq:
            return "⚠️ LLM не настроен. Проверь GROQ_API_KEY в .env"

        if snapshot and snapshot.is_ready:
            crm_context = await self._build_context_from_snapshot(snapshot)
        else:
            crm_context = await self._get_crm_context()

        system = _SYSTEM.format(
            crm_context=crm_context,
            today=datetime.now().strftime("%d.%m.%Y"),
        )

        history = self._history.get(user_id, [])
        messages = [{"role": "system", "content": system}] + history + [
            {"role": "user", "content": message},
        ]

        answer = await asyncio.to_thread(self._call_llm, messages)

        # Обновить историю (хранить последние 10 пар)
        updated = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": answer},
        ]
        self._history[user_id] = updated[-20:]

        return answer

    async def _build_context_from_snapshot(self, snapshot) -> str:
        """Собрать текстовый снимок CRM из предзагруженного CrmSnapshot (без API-запросов)."""
        return self._format_context(
            snapshot.orders,
            snapshot.products,
            len(snapshot.clients),
            snapshot_age=snapshot.age_str,
        )

    def clear_history(self, user_id: int) -> None:
        """Очистить историю диалога пользователя."""
        self._history.pop(user_id, None)
