"""Роутер дашборда — тонкий слой, делегирует DashboardService."""

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

from src.integram_api import IntegramAPIError
from src.integram_client import IntegramError
from src.services.dashboard_service import DashboardService
from src.web.deps import (
    CurrentUser,
    DashboardStats,
    _get_crm,
    _require_role,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["dashboard"])


def _get_dashboard_service(request: Request) -> DashboardService:
    """Получить DashboardService из app.state или создать с CRM."""
    svc = getattr(request.app.state, "services", None)
    if svc and hasattr(svc, "dashboard_service") and svc.dashboard_service:
        return svc.dashboard_service
    # Fallback: сервис будет использовать CRM из _get_crm()
    crm = getattr(request.app.state, "crm", None)
    return DashboardService(crm=crm)


@router.get("/api/dashboard", response_model=DashboardStats)
async def get_dashboard(
    request: Request,
    period: str = "all",
    _: CurrentUser = Depends(_require_role("admin")),
) -> DashboardStats:
    """Статистика. period: today|7d|30d|90d|all"""
    try:
        svc = _get_dashboard_service(request)
        if not svc._crm:
            crm = await _get_crm()
            svc.set_crm(crm)
        stats = await svc.get_stats(period)
        return DashboardStats(**stats)
    except (IntegramError, IntegramAPIError) as exc:
        logger.error("Ошибка Integram: %s", exc)
        raise HTTPException(status_code=502, detail="Ошибка CRM")


@router.get("/api/dashboard/charts")
async def get_dashboard_charts(
    request: Request,
    period: str = "all",
    _: CurrentUser = Depends(_require_role("admin")),
) -> dict[str, Any]:
    """Данные для графиков."""
    try:
        svc = _get_dashboard_service(request)
        if not svc._crm:
            crm = await _get_crm()
            svc.set_crm(crm)
        return await svc.get_charts(period)
    except (IntegramError, IntegramAPIError) as exc:
        logger.error("Ошибка Integram: %s", exc)
        raise HTTPException(502, "Ошибка CRM")


@router.get("/api/dashboard/alerts")
async def get_dashboard_alerts(
    request: Request,
    _: CurrentUser = Depends(_require_role("admin")),
) -> dict[str, Any]:
    """Счётчики «Требуют внимания»."""
    try:
        svc = _get_dashboard_service(request)
        if not svc._crm:
            crm = await _get_crm()
            svc.set_crm(crm)
        return await svc.get_alerts()
    except (IntegramError, IntegramAPIError) as exc:
        logger.error("Ошибка Integram: %s", exc)
        raise HTTPException(502, "Ошибка CRM")
