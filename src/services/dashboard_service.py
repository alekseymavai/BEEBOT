"""DashboardService — агрегация данных для дашборда.

Вся бизнес-логика (фильтрация, подсчёт, агрегация) вынесена из роутера.
Роутер только вызывает методы сервиса и возвращает результат.

Best practice: Cosmic Python Ch.4 — «Routers should be thin —
parse request, call service, return response.»
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from src.crm_constants import ORDER_STATUSES

_MONTH_NAMES = {
    1: "Янв", 2: "Фев", 3: "Мар", 4: "Апр",
    5: "Май", 6: "Июн", 7: "Июл", 8: "Авг",
    9: "Сен", 10: "Окт", 11: "Ноя", 12: "Дек",
}

_PERIOD_DAYS: dict[str, Optional[int]] = {
    "today": 1,
    "7d": 7,
    "30d": 30,
    "90d": 90,
    "all": None,
}

LOW_STOCK_THRESHOLD = 5


class DashboardService:
    """Агрегация данных для дашборда веб-панели."""

    def __init__(self, crm=None):
        self._crm = crm

    def set_crm(self, crm) -> None:
        self._crm = crm

    # ------------------------------------------------------------------
    # Статистика
    # ------------------------------------------------------------------

    async def get_stats(self, period: str = "all") -> dict[str, Any]:
        """Основная статистика: заказы, клиенты, выручка."""
        if not self._crm:
            return {
                "total_orders": 0, "total_clients": 0,
                "total_revenue": 0.0, "avg_order": 0.0,
                "new_orders": 0, "delivered_orders": 0,
            }

        since = self._cutoff(period)
        if since is None:
            return await self._crm.get_dashboard_stats()

        all_orders = await self._crm.get_orders()
        orders = self._filter_by_date(all_orders, since)
        total = len(orders)
        revenue = sum(o.total or 0 for o in orders)

        return {
            "total_orders": total,
            "total_clients": len({o.client_id for o in orders if o.client_id}),
            "total_revenue": revenue,
            "avg_order": revenue / total if total else 0,
            "new_orders": sum(1 for o in orders if o.status == "Новый"),
            "delivered_orders": sum(1 for o in orders if o.status == "Доставлен"),
        }

    # ------------------------------------------------------------------
    # Графики
    # ------------------------------------------------------------------

    async def get_charts(self, period: str = "all") -> dict[str, Any]:
        """Данные графиков: выручка по месяцам, воронка, доставка, топ товаров."""
        if not self._crm:
            return {"monthly": {}, "funnel": {}, "delivery": {}, "top_products": []}

        all_orders = await self._crm.get_orders()
        since = self._cutoff(period)
        orders = self._filter_by_date(all_orders, since) if since else all_orders

        monthly = self._aggregate_monthly(orders)
        status_counts = self._count_by_field(orders, "status", "Неизвестно")
        delivery_counts = self._count_by_field(orders, "delivery_method", "Не указан")
        top_products = await self._top_products(orders)

        return {
            "monthly": monthly,
            "funnel": {
                "labels": ORDER_STATUSES,
                "data": [status_counts.get(s, 0) for s in ORDER_STATUSES],
            },
            "delivery": {
                "labels": list(delivery_counts.keys()),
                "data": list(delivery_counts.values()),
            },
            "top_products": top_products,
        }

    # ------------------------------------------------------------------
    # Алерты
    # ------------------------------------------------------------------

    async def get_alerts(self) -> dict[str, int]:
        """Счётчики «Требуют внимания»."""
        if not self._crm:
            return {"stale_new": 0, "stale_confirmed": 0, "low_stock": 0}

        now = datetime.now(tz=timezone.utc)
        all_orders = await self._crm.get_orders()

        stale_new = sum(
            1 for o in all_orders
            if o.status == "Новый"
            and o.date
            and (now - o.date.replace(tzinfo=timezone.utc)).total_seconds() > 86400
        )

        stale_confirmed = sum(
            1 for o in all_orders
            if o.status == "Подтверждён"
            and o.date
            and (now - o.date.replace(tzinfo=timezone.utc)).total_seconds() > 259200
        )

        products = await self._crm.get_products(in_stock_only=True)
        low_stock = sum(
            1 for p in products
            if p.stock is not None and p.stock < LOW_STOCK_THRESHOLD
        )

        return {
            "stale_new": stale_new,
            "stale_confirmed": stale_confirmed,
            "low_stock": low_stock,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cutoff(period: str) -> Optional[datetime]:
        days = _PERIOD_DAYS.get(period)
        if days is None:
            return None
        return datetime.now(tz=timezone.utc) - timedelta(days=days)

    @staticmethod
    def _filter_by_date(orders: list, since: datetime) -> list:
        return [
            o for o in orders
            if o.date and o.date.replace(tzinfo=timezone.utc) >= since
        ]

    @staticmethod
    def _count_by_field(orders: list, field: str, default: str = "") -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for o in orders:
            counts[getattr(o, field, None) or default] += 1
        return dict(counts)

    @staticmethod
    def _aggregate_monthly(orders: list) -> dict[str, Any]:
        monthly_revenue: dict[str, float] = defaultdict(float)
        monthly_count: dict[str, int] = defaultdict(int)

        for o in orders:
            if o.date and 2024 <= o.date.year <= 2030:
                month_key = f"{o.date.year}-{o.date.month:02d}"
                monthly_revenue[month_key] += o.total or 0
                monthly_count[month_key] += 1

        sorted_months = sorted(monthly_revenue.keys())
        labels = []
        for mk in sorted_months:
            year, month = mk.split("-")
            labels.append(f"{_MONTH_NAMES.get(int(month), month)} {year[-2:]}")

        return {
            "labels": labels,
            "revenue": [monthly_revenue[m] for m in sorted_months],
            "count": [monthly_count[m] for m in sorted_months],
        }

    async def _top_products(self, orders: list, limit: int = 5) -> list[dict]:
        order_ids = {o.id for o in orders if o.id}
        all_items = await self._crm.get_order_items_bulk()

        product_qty: dict[str, int] = defaultdict(int)
        product_revenue: dict[str, float] = defaultdict(float)

        for item in all_items:
            if item.order_id not in order_ids:
                continue
            name = item.product_name or f"Товар #{item.product_id}"
            product_qty[name] += item.quantity or 0
            product_revenue[name] += item.total or 0

        total_revenue = sum(product_revenue.values()) or 1
        top = sorted(product_qty.keys(), key=lambda k: product_qty[k], reverse=True)[:limit]

        return [
            {
                "name": name,
                "qty": product_qty[name],
                "revenue": round(product_revenue[name], 2),
                "share": round(product_revenue[name] / total_revenue * 100, 1),
            }
            for name in top
        ]
