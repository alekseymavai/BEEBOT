"""FastAPI-бэкенд веб-панели управления заказами «Усадьба Дмитровых».

Конфигурация через .env:
  WEB_USERNAME      — логин администратора-фоллбэк (по умолчанию: admin)
  WEB_PASSWORD      — пароль администратора-фоллбэк (ОБЯЗАТЕЛЬНО)
  WEB_SECRET        — секрет JWT (ОБЯЗАТЕЛЬНО)
  WEB_TOKEN_TTL     — время жизни токена в минутах (по умолчанию: 60)
  WEB_CORS_ORIGINS  — разрешённые домены через запятую
  WEB_INTERNAL_SECRET — секрет для внутреннего SSE-эндпоинта
"""

from __future__ import annotations

import asyncio
import json as json_lib
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, StreamingResponse
from jose import JWTError, jwt
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src.web.deps import (
    ALGORITHM,
    INTERNAL_SECRET,
    WEB_SECRET,
    CurrentUser,
    _event_subscribers,
    _require_role,
    push_event,
)
from src.web.routers.auth import router as auth_router
from src.web.routers.batches import router as batches_router
from src.web.routers.clients import router as clients_router
from src.web.routers.dashboard import router as dashboard_router
from src.web.routers.export import router as export_router
from src.web.routers.orders import order_items_router, router as orders_router
from src.web.routers.products import router as products_router
from src.web.routers.users import router as users_router

logger = logging.getLogger(__name__)

_CORS_ORIGINS_RAW = os.getenv("WEB_CORS_ORIGINS", "http://185.233.200.13:8088,http://localhost:5173")
_CORS_ORIGINS = [o.strip() for o in _CORS_ORIGINS_RAW.split(",") if o.strip()]


async def _telegram_alert(text: str) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("BEEKEEPER_CHAT_ID") or os.getenv("ADMIN_CHAT_ID")
    if not token or not chat_id:
        return
    try:
        async with httpx.AsyncClient(timeout=5) as http_client:
            await http_client.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": int(chat_id), "text": text},
            )
    except Exception as e:
        logger.warning("Не удалось отправить Telegram-алерт: %s", e)


limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    from src.logging_config import setup_logging
    setup_logging()
    await _telegram_alert("🌐 Веб-панель BEEBOT запущена")
    yield
    await _telegram_alert("🌐 Веб-панель BEEBOT остановлена")


app = FastAPI(
    title="BEEBOT — Веб-панель",
    description="Управление заказами «Усадьба Дмитровых»",
    version="2.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=429, content={"detail": "Слишком много запросов, попробуйте позже"})


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    from fastapi.responses import JSONResponse
    logger.error("Необработанная ошибка: %s %s — %s", request.method, request.url.path, exc)
    return JSONResponse(status_code=500, content={"detail": "Внутренняя ошибка сервера"})


app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(batches_router)
app.include_router(dashboard_router)
app.include_router(orders_router)
app.include_router(order_items_router)
app.include_router(clients_router)
app.include_router(products_router)
app.include_router(export_router)
app.include_router(users_router)


@app.get("/api/health", tags=["health"])
async def health_check():
    return {"status": "ok", "service": "beebot-web"}


@app.get("/api/docs/architecture", tags=["docs"], response_class=PlainTextResponse)
async def get_architecture_doc(
    _: CurrentUser = Depends(_require_role("admin")),
) -> PlainTextResponse:
    """Архитектурный документ BEEBOT в формате Markdown."""
    from src.config import BASE_DIR
    doc_path = BASE_DIR / "docs" / "architecture" / "BEEBOT_ARCHITECTURE.md"
    if not doc_path.exists():
        raise HTTPException(status_code=404, detail="Документ не найден")
    return PlainTextResponse(doc_path.read_text(encoding="utf-8"))


@app.get("/api/events", tags=["events"])
async def sse_events(token: Optional[str] = None):
    if token:
        try:
            payload = jwt.decode(token, WEB_SECRET, algorithms=[ALGORITHM])
            role = payload.get("role", "")
            if role not in ("admin", "warehouse"):
                raise HTTPException(status_code=403, detail="Недостаточно прав")
        except JWTError:
            raise HTTPException(status_code=401, detail="Неверный токен")
    else:
        raise HTTPException(status_code=401, detail="Токен не передан")

    queue: asyncio.Queue = asyncio.Queue(maxsize=100)
    _event_subscribers.append(queue)

    async def event_stream():
        try:
            while True:
                event = await queue.get()
                yield f"data: {json_lib.dumps(event, ensure_ascii=False)}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            if queue in _event_subscribers:
                _event_subscribers.remove(queue)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/internal/push-event", tags=["internal"], include_in_schema=False)
async def internal_push_event(request: Request, body: dict) -> dict:
    if not INTERNAL_SECRET:
        raise HTTPException(status_code=403, detail="Внутренний эндпоинт отключён")
    secret = request.headers.get("X-Internal-Secret", "")
    if secret != INTERNAL_SECRET:
        raise HTTPException(status_code=403, detail="Неверный секрет")
    event_type = body.pop("type", "unknown")
    await push_event(event_type, body)
    return {"ok": True}
