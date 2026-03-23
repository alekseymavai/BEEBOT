"""Роутер клиентов."""

import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException

from src.integram_api import IntegramAPIError
from src.integram_client import IntegramError
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
