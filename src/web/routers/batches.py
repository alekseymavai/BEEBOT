"""Роутер партий отправки."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from src.integram_api import IntegramAPIError
from src.integram_client import IntegramError
from src.web.deps import CurrentUser, _get_crm, _paginate, _require_role

logger = logging.getLogger(__name__)
router = APIRouter()


class BatchCreate(BaseModel):
    date: str              # "DD.MM.YYYY"
    delivery_method: Optional[str] = ""
    note: Optional[str] = ""


class BatchOrderAssign(BaseModel):
    batch_id: Optional[int] = None   # None — снять партию


@router.get("/api/batches")
async def list_batches(
    page: int = 1,
    per_page: int = 50,
    _: CurrentUser = Depends(_require_role("admin")),
):
    """Список партий отправки с количеством заказов."""
    crm = await _get_crm()
    try:
        batches = await crm.get_batches()
        orders = await crm.get_orders()
        counts: dict[int, int] = {}
        for o in orders:
            bid = getattr(o, "batch_id", None)
            if bid:
                counts[bid] = counts.get(bid, 0) + 1
        for b in batches:
            b["order_count"] = counts.get(b["id"], b.get("count", 0))
    finally:
        await crm.close()
    return _paginate(batches, page, per_page)


@router.post("/api/batches", status_code=201)
async def create_batch(
    body: BatchCreate,
    _: CurrentUser = Depends(_require_role("admin")),
):
    """Создать новую партию отправки."""
    try:
        try:
            date = datetime.strptime(body.date, "%d.%m.%Y")
        except ValueError:
            date = datetime.strptime(body.date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=422, detail="Неверный формат даты (ожидается DD.MM.YYYY)")

    crm = await _get_crm()
    try:
        try:
            batch_id = await crm.create_batch(
                date=date,
                delivery_method=body.delivery_method or "",
                note=body.note or "",
            )
        except (IntegramError, IntegramAPIError) as e:
            raise HTTPException(status_code=502, detail=str(e))
    finally:
        await crm.close()
    return {"id": batch_id, "date": body.date, "delivery_method": body.delivery_method}


@router.patch("/api/orders/{order_id}/batch")
async def assign_batch(
    order_id: int,
    body: BatchOrderAssign,
    _: CurrentUser = Depends(_require_role("admin")),
):
    """Назначить / снять партию для заказа."""
    crm = await _get_crm()
    try:
        try:
            await crm.set_order_batch(order_id, body.batch_id)
            if body.batch_id:
                await crm.update_batch_count(body.batch_id)
        except (IntegramError, IntegramAPIError) as e:
            raise HTTPException(status_code=502, detail=str(e))
    finally:
        await crm.close()
    return {"ok": True, "order_id": order_id, "batch_id": body.batch_id}


@router.get("/api/batches/{batch_id}/orders")
async def batch_orders(
    batch_id: int,
    _: CurrentUser = Depends(_require_role("admin")),
):
    """Заказы, входящие в партию."""
    crm = await _get_crm()
    try:
        all_orders = await crm.get_orders()
        orders = [o for o in all_orders if getattr(o, "batch_id", None) == batch_id]
    finally:
        await crm.close()
    return {"items": orders, "total": len(orders)}
