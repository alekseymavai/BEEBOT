"""Роутер дашборда."""

import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException

from src.crm_constants import ORDER_STATUSES
from src.integram_api import IntegramAPIError
from src.integram_client import IntegramError
from src.web.deps import (
    CurrentUser,
    DashboardStats,
    _MONTH_NAMES,
    _get_crm,
    _require_role,
    get_orders_cache,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["dashboard"])

_PERIOD_DAYS: dict[str, Optional[int]] = {
    "today": 1,
    "7d": 7,
    "30d": 30,
    "90d": 90,
    "all": None,
}


def _cutoff(period: str) -> Optional[datetime]:
    days = _PERIOD_DAYS.get(period)
    if days is None:
        return None
    return datetime.now(tz=timezone.utc) - timedelta(days=days)


@router.get("/api/dashboard", response_model=DashboardStats)
async def get_dashboard(
    period: str = "all",
    _: CurrentUser = Depends(_require_role("admin")),
) -> DashboardStats:
    """Статистика. period: today|7d|30d|90d|all"""
    try:
        crm = await _get_crm()
        try:
            since = _cutoff(period)
            if since is None:
                # Без фильтра — быстрый путь через CRM-агрегат
                stats = await crm.get_dashboard_stats()
                return DashboardStats(**stats)

            # С фильтром — вычисляем из кеша заказов
            all_orders = await get_orders_cache(crm)
            orders = [
                o for o in all_orders
                if o.date and o.date.replace(tzinfo=timezone.utc) >= since
            ]
            total = len(orders)
            revenue = sum(o.total or 0 for o in orders)
            return DashboardStats(
                total_orders=total,
                total_clients=len({o.client_id for o in orders if o.client_id}),
                total_revenue=revenue,
                avg_order=revenue / total if total else 0,
                new_orders=sum(1 for o in orders if o.status == "Новый"),
                delivered_orders=sum(1 for o in orders if o.status == "Доставлен"),
            )
        finally:
            await crm.close()
    except (IntegramError, IntegramAPIError) as exc:
        logger.error("Ошибка Integram: %s", exc)
        raise HTTPException(status_code=502, detail="Ошибка CRM")


@router.get("/api/dashboard/charts")
async def get_dashboard_charts(
    period: str = "all",
    _: CurrentUser = Depends(_require_role("admin")),
) -> dict[str, Any]:
    """Данные для графиков: выручка по месяцам, воронка статусов, доставка, топ товаров."""
    try:
        crm = await _get_crm()
        try:
            all_orders = await crm.get_orders()
            since = _cutoff(period)
            orders = all_orders
            if since is not None:
                orders = [
                    o for o in all_orders
                    if o.date and o.date.replace(tzinfo=timezone.utc) >= since
                ]

            monthly_revenue: dict[str, float] = defaultdict(float)
            monthly_count: dict[str, int] = defaultdict(int)
            for o in orders:
                if o.date and 2024 <= o.date.year <= 2030:
                    month_key = f"{o.date.year}-{o.date.month:02d}"
                    monthly_revenue[month_key] += o.total or 0
                    monthly_count[month_key] += 1

            sorted_months = sorted(monthly_revenue.keys())
            month_labels = []
            for mk in sorted_months:
                year, month = mk.split("-")
                month_labels.append(f"{_MONTH_NAMES.get(int(month), month)} {year[-2:]}")

            status_counts: dict[str, int] = defaultdict(int)
            for o in orders:
                status_counts[o.status or "Неизвестно"] += 1

            delivery_counts: dict[str, int] = defaultdict(int)
            for o in orders:
                delivery_counts[o.delivery_method or "Не указан"] += 1

            # Топ товаров за период
            order_ids = {o.id for o in orders if o.id}
            all_items = await crm.get_order_items_bulk()
            product_qty: dict[str, int] = defaultdict(int)
            product_revenue: dict[str, float] = defaultdict(float)
            for item in all_items:
                if item.order_id not in order_ids:
                    continue
                name = item.product_name or f"Товар #{item.product_id}"
                product_qty[name] += item.quantity or 0
                product_revenue[name] += item.total or 0

            total_revenue_all = sum(product_revenue.values()) or 1
            top5 = sorted(product_qty.keys(), key=lambda k: product_qty[k], reverse=True)[:5]

            return {
                "monthly": {
                    "labels": month_labels,
                    "revenue": [monthly_revenue[m] for m in sorted_months],
                    "count": [monthly_count[m] for m in sorted_months],
                },
                "funnel": {
                    "labels": ORDER_STATUSES,
                    "data": [status_counts.get(s, 0) for s in ORDER_STATUSES],
                },
                "delivery": {
                    "labels": list(delivery_counts.keys()),
                    "data": list(delivery_counts.values()),
                },
                "top_products": [
                    {
                        "name": name,
                        "qty": product_qty[name],
                        "revenue": round(product_revenue[name], 2),
                        "share": round(product_revenue[name] / total_revenue_all * 100, 1),
                    }
                    for name in top5
                ],
            }
        finally:
            await crm.close()
    except (IntegramError, IntegramAPIError) as exc:
        logger.error("Ошибка Integram: %s", exc)
        raise HTTPException(502, "Ошибка CRM")


@router.get("/api/dashboard/alerts")
async def get_dashboard_alerts(
    _: CurrentUser = Depends(_require_role("admin")),
) -> dict[str, Any]:
    """Счётчики для блока «Требуют внимания»."""
    try:
        crm = await _get_crm()
        try:
            now = datetime.now(tz=timezone.utc)
            all_orders = await get_orders_cache(crm)

            # Новых заказов без подтверждения >24ч
            stale_new = sum(
                1 for o in all_orders
                if o.status == "Новый"
                and o.date
                and (now - o.date.replace(tzinfo=timezone.utc)).total_seconds() > 86400
            )

            # Подтверждённых >3 дней без отправки
            stale_confirmed = sum(
                1 for o in all_orders
                if o.status == "Подтверждён"
                and o.date
                and (now - o.date.replace(tzinfo=timezone.utc)).total_seconds() > 259200
            )

            # Товары с остатком ниже порога (5 единиц)
            LOW_STOCK_THRESHOLD = 5
            products = await crm.get_products(in_stock_only=True)
            low_stock = sum(
                1 for p in products
                if p.stock is not None and p.stock < LOW_STOCK_THRESHOLD
            )

            return {
                "stale_new": stale_new,
                "stale_confirmed": stale_confirmed,
                "low_stock": low_stock,
            }
        finally:
            await crm.close()
    except (IntegramError, IntegramAPIError) as exc:
        logger.error("Ошибка Integram: %s", exc)
        raise HTTPException(502, "Ошибка CRM")
