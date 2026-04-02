"""Роутер аутентификации и справочников."""

import logging

from fastapi import APIRouter, Depends
from fastapi.security import OAuth2PasswordRequestForm

from src.crm_constants import ORDER_STATUSES, DELIVERY_METHODS, SOURCE_IDS
from src.web.deps import (
    CurrentUser,
    ReferenceData,
    TokenResponse,
    _create_token,
    _get_crm,
    _get_current_user,
    _require_role,
)
from src.web.users import get_user_by_username, verify_password

logger = logging.getLogger(__name__)

router = APIRouter(tags=["auth"])


@router.post("/api/auth/token", response_model=TokenResponse)
async def login(form_data: OAuth2PasswordRequestForm = Depends()) -> TokenResponse:
    """Получить JWT-токен по логину и паролю."""
    from fastapi import HTTPException, status

    credentials_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Неверный логин или пароль",
        headers={"WWW-Authenticate": "Bearer"},
    )

    # 1. Попробовать найти пользователя в CRM
    try:
        crm = await _get_crm()
        try:
            user = await get_user_by_username(crm.api, form_data.username)
        finally:
            await crm.close()

        if user and user.get("active") and verify_password(form_data.password, user["password_hash"]):
            token = _create_token(user["username"], role=user["role"])
            return TokenResponse(access_token=token)
    except Exception as e:
        logger.warning("Ошибка поиска пользователя в CRM: %s", e)

    raise credentials_exc


@router.get("/api/auth/me")
async def get_me(user: CurrentUser = Depends(_get_current_user)) -> dict:
    """Информация о текущем пользователе."""
    return {"username": user.username, "role": user.role}


@router.get("/api/reference", response_model=ReferenceData)
async def get_reference(
    _: CurrentUser = Depends(_require_role("admin", "warehouse")),
) -> ReferenceData:
    return ReferenceData(
        order_statuses=ORDER_STATUSES,
        delivery_methods=DELIVERY_METHODS,
        order_sources=list(SOURCE_IDS.keys()),
    )
