"""Роутер управления пользователями (только admin)."""

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from src.integram_api import IntegramAPIError
from src.integram_client import IntegramError
from src.web.deps import CurrentUser, UserCreate, UserUpdate, _get_crm, _require_role
from src.web.users import (
    VALID_ROLES,
    create_user,
    delete_user as delete_user_service,
    get_all_users,
    update_user as update_user_service,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["users"])


@router.get("/api/users")
async def list_users(_: CurrentUser = Depends(_require_role("admin"))) -> list[dict[str, Any]]:
    try:
        crm = await _get_crm()
        try:
            return await get_all_users(crm.api)
        finally:
            await crm.close()
    except (IntegramError, IntegramAPIError) as exc:
        logger.error("Ошибка Integram: %s", exc)
        raise HTTPException(502, "Ошибка CRM")


@router.post("/api/users")
async def create_user_endpoint(
    body: UserCreate, _: CurrentUser = Depends(_require_role("admin"))
) -> dict:
    if body.role not in VALID_ROLES:
        raise HTTPException(400, f"Недопустимая роль: {body.role}. Допустимые: {', '.join(VALID_ROLES)}")
    try:
        crm = await _get_crm()
        try:
            user_id = await create_user(crm.api, username=body.username,
                                        password=body.password, role=body.role,
                                        display_name=body.display_name)
            return {"ok": True, "user_id": user_id}
        finally:
            await crm.close()
    except ValueError as e:
        raise HTTPException(400, str(e))
    except (IntegramError, IntegramAPIError) as exc:
        logger.error("Ошибка Integram: %s", exc)
        raise HTTPException(502, "Ошибка CRM")


@router.patch("/api/users/{user_id}")
async def update_user_endpoint(
    user_id: int, body: UserUpdate, _: CurrentUser = Depends(_require_role("admin"))
) -> dict:
    if body.role is not None and body.role not in VALID_ROLES:
        raise HTTPException(400, f"Недопустимая роль: {body.role}")
    try:
        crm = await _get_crm()
        try:
            await update_user_service(crm.api, user_id, password=body.password,
                                      role=body.role, display_name=body.display_name,
                                      active=body.active)
            return {"ok": True, "user_id": user_id}
        finally:
            await crm.close()
    except ValueError as e:
        raise HTTPException(400, str(e))
    except (IntegramError, IntegramAPIError) as exc:
        logger.error("Ошибка Integram: %s", exc)
        raise HTTPException(502, "Ошибка CRM")


@router.delete("/api/users/{user_id}")
async def delete_user_endpoint(
    user_id: int, _: CurrentUser = Depends(_require_role("admin"))
) -> dict:
    try:
        crm = await _get_crm()
        try:
            await delete_user_service(crm.api, user_id)
            return {"ok": True, "user_id": user_id}
        finally:
            await crm.close()
    except (IntegramError, IntegramAPIError) as exc:
        logger.error("Ошибка Integram: %s", exc)
        raise HTTPException(502, "Ошибка CRM")
