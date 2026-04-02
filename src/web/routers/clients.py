"""Роутер клиентов."""

import logging
from collections import defaultdict
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from src.integram_api import IntegramAPIError
from src.integram_client import IntegramError
from src.crm_constants import REQ_ORDER_CLIENT, TABLE_ORDERS
from src.web.deps import (
    CurrentUser,
    _DEFAULT_PAGE_SIZE,
    _client_to_dict,
    _get_crm,
    _order_to_dict,
    _paginate,
    _require_role,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["clients"])


class MergeRequest(BaseModel):
    primary_id: int    # клиент, которого оставить
    duplicate_id: int  # клиент, которого удалить (его заказы перейдут к primary)


@router.get("/api/clients")
async def list_clients(
    search: Optional[str] = None,
    page: int = 1,
    per_page: int = _DEFAULT_PAGE_SIZE,
    _: CurrentUser = Depends(_require_role("admin")),
) -> dict[str, Any]:
    """Список клиентов с пагинацией и поиском."""
    try:
        crm = await _get_crm()
        try:
            clients = await crm.get_clients()
            result = [_client_to_dict(c) for c in clients]
            return _paginate(
                result, page, per_page, search,
                search_fields=["name", "phone", "address", "city", "telegram_username"],
            )
        finally:
            await crm.close()
    except (IntegramError, IntegramAPIError) as exc:
        logger.error("Ошибка Integram: %s", exc)
        raise HTTPException(status_code=502, detail="Ошибка CRM")


@router.get("/api/clients/duplicates")
async def list_duplicates(
    _: CurrentUser = Depends(_require_role("admin")),
) -> dict[str, Any]:
    """Найти потенциальные дубли клиентов (одинаковый телефон)."""
    try:
        crm = await _get_crm()
        try:
            clients = await crm.get_clients()
            by_phone: dict[str, list] = defaultdict(list)
            for c in clients:
                phone = (c.phone or "").strip()
                if phone and len(phone) >= 7:
                    by_phone[phone].append(_client_to_dict(c))
            groups = [
                {"phone": phone, "clients": cs}
                for phone, cs in by_phone.items()
                if len(cs) > 1
            ]
            return {"groups": groups, "total": len(groups)}
        finally:
            await crm.close()
    except (IntegramError, IntegramAPIError) as exc:
        logger.error("Ошибка Integram: %s", exc)
        raise HTTPException(status_code=502, detail="Ошибка CRM")


@router.post("/api/clients/merge")
async def merge_clients(
    body: MergeRequest,
    _: CurrentUser = Depends(_require_role("admin")),
) -> dict[str, Any]:
    """Объединить двух клиентов: перенести заказы дубля на primary, удалить дубль."""
    if body.primary_id == body.duplicate_id:
        raise HTTPException(status_code=422, detail="primary и duplicate должны быть разными")
    try:
        crm = await _get_crm()
        try:
            clients = await crm.get_clients()
            primary = next((c for c in clients if c.id == body.primary_id), None)
            duplicate = next((c for c in clients if c.id == body.duplicate_id), None)
            if not primary:
                raise HTTPException(status_code=404, detail=f"Клиент {body.primary_id} не найден")
            if not duplicate:
                raise HTTPException(status_code=404, detail=f"Клиент {body.duplicate_id} не найден")

            # Перенести заказы дубля → primary
            dup_orders = await crm.get_orders(client_id=body.duplicate_id)
            moved = 0
            for order in dup_orders:
                try:
                    await crm.update_order(order.id, client_id=body.primary_id)
                except Exception:
                    # Fallback для v1: через low-level API если доступен
                    if hasattr(crm, "api"):
                        await crm.api.set_requisites(order.id, TABLE_ORDERS, {REQ_ORDER_CLIENT: str(body.primary_id)})
                moved += 1

            # Скопировать telegram_id если у primary нет
            if not primary.telegram_id and duplicate.telegram_id:
                await crm.update_client(body.primary_id, telegram_id=duplicate.telegram_id)

            # Скопировать telegram_username если у primary нет
            if not primary.telegram_username and duplicate.telegram_username:
                await crm.update_client(body.primary_id, telegram_username=duplicate.telegram_username)

            # Удалить дубль
            try:
                await crm.delete_client(body.duplicate_id)
            except AttributeError:
                # Fallback для v1: через low-level API
                if hasattr(crm, "api"):
                    await crm.api.delete_object(body.duplicate_id)

            logger.info(
                "Клиенты объединены: primary=%d, duplicate=%d удалён, заказов перенесено=%d",
                body.primary_id, body.duplicate_id, moved,
            )
            return {
                "ok": True,
                "primary_id": body.primary_id,
                "duplicate_id": body.duplicate_id,
                "orders_moved": moved,
            }
        finally:
            await crm.close()
    except (IntegramError, IntegramAPIError) as exc:
        logger.error("Ошибка Integram: %s", exc)
        raise HTTPException(status_code=502, detail="Ошибка CRM")


@router.get("/api/clients/{client_id}")
async def get_client(
    client_id: int,
    _: CurrentUser = Depends(_require_role("admin")),
) -> dict[str, Any]:
    """Клиент + история заказов."""
    try:
        crm = await _get_crm()
        try:
            clients = await crm.get_clients()
            client = next((c for c in clients if c.id == client_id), None)
            if not client:
                raise HTTPException(status_code=404, detail="Клиент не найден")
            orders = await crm.get_orders(client_id=client_id)
            result = _client_to_dict(client)
            result["orders"] = [_order_to_dict(o) for o in orders]
            return result
        finally:
            await crm.close()
    except (IntegramError, IntegramAPIError) as exc:
        logger.error("Ошибка Integram: %s", exc)
        raise HTTPException(status_code=502, detail="Ошибка CRM")
