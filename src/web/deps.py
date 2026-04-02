"""Общие зависимости, модели, хелперы для FastAPI-роутеров."""

from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from pydantic import BaseModel

from src.crm_factory import get_crm_client

# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------

WEB_USERNAME = os.getenv("WEB_USERNAME", "admin")
WEB_PASSWORD = os.getenv("WEB_PASSWORD", "")
if not WEB_PASSWORD:
    raise RuntimeError(
        "WEB_PASSWORD не задан! Установите переменную окружения WEB_PASSWORD в .env"
    )
WEB_SECRET = os.getenv("WEB_SECRET", "")
if not WEB_SECRET:
    raise RuntimeError(
        "WEB_SECRET не задан! Установите переменную окружения WEB_SECRET в .env"
    )
_TOKEN_TTL = int(os.getenv("WEB_TOKEN_TTL", "60"))  # minutes
ALGORITHM = "HS256"
INTERNAL_SECRET = os.getenv("WEB_INTERNAL_SECRET", "")

# ---------------------------------------------------------------------------
# Pydantic-схемы
# ---------------------------------------------------------------------------


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class CurrentUser(BaseModel):
    username: str
    role: str  # "admin" | "warehouse"


class DashboardStats(BaseModel):
    total_orders: int
    total_clients: int
    total_revenue: float
    avg_order: float
    new_orders: int
    delivered_orders: int


class ReferenceData(BaseModel):
    order_statuses: list[str]
    delivery_methods: list[str]
    order_sources: list[str]


class StatusUpdate(BaseModel):
    status: str


class TrackingUpdate(BaseModel):
    tracking_number: str


class OrderUpdate(BaseModel):
    delivery_address: Optional[str] = None
    delivery_method: Optional[str] = None
    delivery_cost: Optional[float] = None
    comment: Optional[str] = None


class ChecklistUpdate(BaseModel):
    cdek_confirmed: Optional[bool] = None
    client_notified: Optional[bool] = None
    stock_checked: Optional[bool] = None


class ItemCreate(BaseModel):
    product_id: int
    quantity: int = 1
    unit_price: float


class ItemUpdate(BaseModel):
    quantity: Optional[int] = None
    unit_price: Optional[float] = None


class ProductCreate(BaseModel):
    name: str
    category: Optional[str] = None
    price: Optional[float] = None
    weight: Optional[float] = None
    description: Optional[str] = None
    in_stock: bool = True
    sku_uds: Optional[str] = None
    short_name: Optional[str] = None
    stock: Optional[int] = None


class ProductUpdate(BaseModel):
    name: Optional[str] = None
    category: Optional[str] = None
    price: Optional[float] = None
    weight: Optional[float] = None
    description: Optional[str] = None
    in_stock: Optional[bool] = None
    sku_uds: Optional[str] = None
    short_name: Optional[str] = None
    stock: Optional[int] = None


class StockUpdate(BaseModel):
    stock: int


class OrderCreate(BaseModel):
    client_name: str
    phone: Optional[str] = None
    delivery_method: Optional[str] = None
    delivery_address: Optional[str] = None
    delivery_cost: Optional[float] = None
    source: Optional[str] = "Сайт"
    comment: Optional[str] = None
    items: list[ItemCreate] = []


class UserCreate(BaseModel):
    username: str
    password: str
    role: str
    display_name: str = ""


class UserUpdate(BaseModel):
    password: Optional[str] = None
    role: Optional[str] = None
    display_name: Optional[str] = None
    active: Optional[bool] = None


# ---------------------------------------------------------------------------
# Пагинация и поиск
# ---------------------------------------------------------------------------

_DEFAULT_PAGE_SIZE = 50
_MAX_PAGE_SIZE = 1000


def _paginate(
    items: list[dict],
    page: int = 1,
    per_page: int = _DEFAULT_PAGE_SIZE,
    search: Optional[str] = None,
    search_fields: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Пагинация + поиск по списку элементов."""
    if search and search_fields:
        q = search.lower()
        items = [
            item for item in items
            if any(q in str(item.get(f, "")).lower() for f in search_fields)
        ]

    total = len(items)
    per_page = min(max(per_page, 1), _MAX_PAGE_SIZE)
    page = max(page, 1)
    pages = max((total + per_page - 1) // per_page, 1)
    start = (page - 1) * per_page

    return {
        "items": items[start:start + per_page],
        "total": total,
        "page": page,
        "per_page": per_page,
        "pages": pages,
    }


# ---------------------------------------------------------------------------
# JWT helpers
# ---------------------------------------------------------------------------

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token")


def _create_token(username: str, role: str = "admin") -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=_TOKEN_TTL)
    payload = {"sub": username, "role": role, "exp": expire}
    return jwt.encode(payload, WEB_SECRET, algorithm=ALGORITHM)


async def _get_current_user(token: str = Depends(oauth2_scheme)) -> CurrentUser:
    credentials_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Неверный или просроченный токен",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, WEB_SECRET, algorithms=[ALGORITHM])
        username: Optional[str] = payload.get("sub")
        if username is None:
            raise credentials_exc
        role = payload.get("role", "admin")
    except JWTError:
        raise credentials_exc
    return CurrentUser(username=username, role=role)


def _require_role(*roles: str):
    """Фабрика зависимостей: проверить что роль пользователя в списке."""
    async def checker(user: CurrentUser = Depends(_get_current_user)) -> CurrentUser:
        if user.role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Недостаточно прав",
            )
        return user
    return checker


# ---------------------------------------------------------------------------
# CRM-клиент
# ---------------------------------------------------------------------------

async def _get_crm():
    """Создать и авторизовать CRM-клиент (v1 или v2 по feature flag INTEGRAM_V2)."""
    client = get_crm_client()
    await client.authenticate()
    return client


# ---------------------------------------------------------------------------
# Кеш заказов (orders list)
# ---------------------------------------------------------------------------

_ORDERS_CACHE_TTL = 90  # 1.5 минуты
_orders_cache: Optional[list] = None
_orders_cache_ts: float = 0.0
_orders_cache_lock: asyncio.Lock = asyncio.Lock()


async def get_orders_cache(crm: "IntegramClient") -> list:
    """Вернуть кеш всех заказов, при необходимости обновить (TTL 90 сек)."""
    global _orders_cache, _orders_cache_ts
    now = time.monotonic()
    if _orders_cache is None or (now - _orders_cache_ts) > _ORDERS_CACHE_TTL:
        async with _orders_cache_lock:
            if _orders_cache is None or (time.monotonic() - _orders_cache_ts) > _ORDERS_CACHE_TTL:
                import logging
                _log = logging.getLogger(__name__)
                _log.info("Загрузка кеша заказов (TTL истёк)...")
                _orders_cache = await crm.get_orders()
                _orders_cache_ts = time.monotonic()
                _log.info("Кеш заказов: %d заказов", len(_orders_cache))
    return _orders_cache  # type: ignore[return-value]


def invalidate_orders_cache() -> None:
    """Сбросить кеш заказов (вызывать после создания/обновления заказа)."""
    global _orders_cache_ts
    _orders_cache_ts = 0.0


# ---------------------------------------------------------------------------
# Кеш позиций заказов (items_by_order)
# ---------------------------------------------------------------------------

_ITEMS_CACHE_TTL = 600  # 10 минут
_items_by_order: Optional[dict[int, list]] = None
_items_cache_ts: float = 0.0
_items_cache_lock: asyncio.Lock = asyncio.Lock()


async def get_items_cache(crm: "IntegramClient") -> dict[int, list]:
    """Вернуть кеш позиций {order_id: [OrderItem, ...]}, при необходимости обновить.

    Первый запрос (или после TTL) загружает ВСЕ позиции один раз.
    Последующие запросы — мгновенно из памяти.
    """
    global _items_by_order, _items_cache_ts
    now = time.monotonic()
    if _items_by_order is None or (now - _items_cache_ts) > _ITEMS_CACHE_TTL:
        async with _items_cache_lock:
            # Повторная проверка под локом
            if _items_by_order is None or (time.monotonic() - _items_cache_ts) > _ITEMS_CACHE_TTL:
                import logging
                _log = logging.getLogger(__name__)
                _log.info("Загрузка кеша позиций заказов (TTL истёк)...")
                all_items = await crm.get_order_items_bulk()
                cache: dict[int, list] = {}
                for item in all_items:
                    cache.setdefault(item.order_id, []).append(item)
                _items_by_order = cache
                _items_cache_ts = time.monotonic()
                _log.info("Кеш позиций: %d позиций для %d заказов", len(all_items), len(cache))
    return _items_by_order  # type: ignore[return-value]


def invalidate_items_cache() -> None:
    """Сбросить кеш позиций (вызывать после добавления/удаления позиций)."""
    global _items_cache_ts
    _items_cache_ts = 0.0


# ---------------------------------------------------------------------------
# Сериализаторы моделей
# ---------------------------------------------------------------------------

def _order_to_dict(order) -> dict[str, Any]:
    d = order.model_dump()
    if order.date:
        d["date"] = order.date.strftime("%d.%m.%Y %H:%M:%S")
    if order.items:
        d["items"] = [i.model_dump() for i in order.items]
    return d


def _client_to_dict(client) -> dict[str, Any]:
    d = client.model_dump()
    d["name"] = d.pop("full_name", "")
    return d


def _product_to_dict(product) -> dict[str, Any]:
    return product.model_dump()


# ---------------------------------------------------------------------------
# Константы бизнес-логики
# ---------------------------------------------------------------------------

EDITABLE_STATUSES = {"Новый", "Подтверждён", "В сборке"}

_WAREHOUSE_STATUS_TRANSITIONS = {
    ("Подтверждён", "В сборке"),
    ("В сборке", "Отправлен"),
}

_MONTH_NAMES = {
    1: "Янв", 2: "Фев", 3: "Мар", 4: "Апр",
    5: "Май", 6: "Июн", 7: "Июл", 8: "Авг",
    9: "Сен", 10: "Окт", 11: "Ноя", 12: "Дек",
}

# ---------------------------------------------------------------------------
# SSE — серверные уведомления
# ---------------------------------------------------------------------------

_event_subscribers: list[asyncio.Queue] = []


async def push_event(event_type: str, data: dict) -> None:
    """Отправить событие всем подписанным SSE-клиентам."""
    payload = {"type": event_type, **data}
    dead: list[asyncio.Queue] = []
    for q in _event_subscribers:
        try:
            q.put_nowait(payload)
        except asyncio.QueueFull:
            dead.append(q)
    for q in dead:
        _event_subscribers.remove(q)
