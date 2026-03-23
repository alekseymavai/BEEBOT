"""Роутер дашборда."""

import logging
from collections import defaultdict
from typing import Any

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
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["dashboard"])


@router.get("/api/dashboard", response_model=DashboardStats)
async def get_dashboard(
    _: CurrentUser = Depends(_require_role("admin")),
) -> DashboardStats:
    """Статистика."""
    try:
        crm = await _get_crm()
        try:
            stats = await crm.get_dashboard_stats()
            return DashboardStats(**stats)
        finally:
            await crm.close()
    except (IntegramError, IntegramAPIError) as exc:
        logger.error("Ошибка Integram: %s", exc)
        raise HTTPException(status_code=502, detail="Ошибка CRM")


@router.get("/api/dashboard/charts")
async def get_dashboard_charts(
    _: CurrentUser = Depends(_require_role("admin")),
) -> dict[str, Any]:
    """Данные для графиков: выручка по месяцам, воронка статусов, доставка."""
    try:
        crm = await _get_crm()
        try:
            orders = await crm.get_orders()

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
            }
        finally:
            await crm.close()
    except (IntegramError, IntegramAPIError) as exc:
        logger.error("Ошибка Integram: %s", exc)
        raise HTTPException(502, "Ошибка CRM")
