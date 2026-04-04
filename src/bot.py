"""Telegram bot for BEEBOT — единая точка входа.

Один процесс запускает и Telegram polling, и FastAPI (uvicorn).
Сервисы создаются один раз в startup.py и разделяются между ботом и веб-панелью.

Best practice: «Run bot and web server in the same asyncio loop
to share services without IPC overhead.»
"""

import asyncio
import logging
import os

from aiogram import Bot, Dispatcher
from aiogram.fsm.storage.memory import MemoryStorage

from src.config import (
    TELEGRAM_BOT_TOKEN,
    BEEKEEPER_CHAT_ID,
    TG_SOCKS_PROXY,
    WORKER_CHAT_IDS,
)
from src.admin import router as admin_router, setup_admin
from src.logging_config import setup_logging
from src.startup import Services, create_services, start_background_tasks

# Роутеры из src/routers/
from src.routers.inspect import router as inspect_router, setup_inspect
from src.routers.fsm_order import router as fsm_order_router, setup_fsm_order
from src.routers.worker import router as worker_router, setup_worker
from src.routers.bot_admin import router as bot_admin_router, setup_bot_admin
from src.routers.fsm_edit import router as fsm_edit_router, setup_fsm_edit
from src.routers.user import router as user_router, setup_user

logger = logging.getLogger(__name__)

bot = Bot(token=TELEGRAM_BOT_TOKEN)  # может быть переопределён в main() с прокси
dp = Dispatcher(storage=MemoryStorage())

# Порядок регистрации роутеров критичен!
# admin_router — ПЕРВЫМ (приоритет команд /orders, /status, /track и т.д.)
dp.include_router(admin_router)
# bot_admin_router — admin-команды из bot.py (stats, yt_*, faq, dev, admin-mode)
dp.include_router(bot_admin_router)
# inspect_router — InspectFSM (до OrderFSM чтобы перехватить состояния)
dp.include_router(inspect_router)
# fsm_order_router — OrderFSM + callbacks продуктов/инструкций
dp.include_router(fsm_order_router)
# worker_router — очередь сборки работников склада
dp.include_router(worker_router)
# fsm_edit_router — редактирование состава заказа
dp.include_router(fsm_edit_router)
# user_router — ПОСЛЕДНИМ (StateFilter(None) — ловит все остальные сообщения)
dp.include_router(user_router)


async def _alert(text: str) -> None:
    """Отправить алерт пчеловоду. Тихо проглатывает ошибки."""
    if not BEEKEEPER_CHAT_ID:
        return
    try:
        await bot.send_message(BEEKEEPER_CHAT_ID, text)
    except Exception as e:
        logger.warning("Не удалось отправить Telegram-алерт: %s", e)


async def _send_tg(chat_id: int, text: str) -> bool:
    """Обёртка для отправки сообщений через aiogram Bot."""
    try:
        await bot.send_message(chat_id, text)
        return True
    except Exception:
        return False


@dp.startup()
async def on_startup(**_kwargs) -> None:
    """Отправить алерт о старте после установки соединения с Telegram."""
    await _alert("🟢 BEEBOT запущен")


def setup_routers(svc: Services) -> None:
    """Подключить сервисы к aiogram-роутерам.

    Роутеры — тонкие обёртки: парсят Telegram-update, вызывают сервис,
    форматируют ответ. Ноль бизнес-логики.
    """
    setup_admin(
        bot, crm=svc.crm, kb=svc.kb,
        memory=svc.orchestrator._memory, auth=svc.auth,
    )
    setup_inspect(svc.inspector)
    setup_fsm_order(svc.logist, bot)
    setup_fsm_edit(svc.logist, bot)
    setup_bot_admin(
        svc.analyst, svc.orchestrator, svc.admin_chat_agent,
        svc.inspector, bot, auth=svc.auth,
    )
    setup_user(
        svc.orchestrator, svc.admin_chat_agent, svc.logist,
        gift_broker=svc.gift_broker, auth=svc.auth,
    )

    # Режим работника склада
    if svc.crm:
        setup_worker(svc.crm, bot, gift_broker=svc.gift_broker, auth=svc.auth)
        from src.notifications import Notifier
        import src.notifications as _notif_module
        _notif_module._worker_notifier = Notifier(bot)
        if WORKER_CHAT_IDS:
            logger.info("Режим работника склада включён (%d работников).", len(WORKER_CHAT_IDS))


async def _run_web(svc: Services) -> None:
    """Запустить FastAPI (uvicorn) в том же event loop."""
    import uvicorn
    from src.web.api import inject_services

    # Передать сервисы в FastAPI (unified-режим — без повторного create_services)
    inject_services(svc)

    host = os.getenv("WEB_HOST", "0.0.0.0")
    port = int(os.getenv("WEB_PORT", "8088"))

    config = uvicorn.Config(
        "src.web.server:app",
        host=host,
        port=port,
        log_level="info",
    )
    server = uvicorn.Server(config)
    await server.serve()


async def main():
    """Unified-режим: бот + веб-панель в одном процессе."""
    global bot
    setup_logging()

    if TG_SOCKS_PROXY:
        from aiogram.client.session.aiohttp import AiohttpSession
        bot = Bot(token=TELEGRAM_BOT_TOKEN, session=AiohttpSession(proxy=TG_SOCKS_PROXY))
        logger.info("Telegram via SOCKS5 proxy: %s", TG_SOCKS_PROXY)

    logger.info("Starting BEEBOT (unified: bot + web)...")

    # --- Создание всех сервисов через единую точку ---
    svc = await create_services(alert_fn=_alert, send_telegram=_send_tg)

    # --- Подключение сервисов к роутерам ---
    setup_routers(svc)

    # --- Фоновые задачи ---
    await start_background_tasks(svc, bot=bot, alert_fn=_alert)

    logger.info("Bot + Web are running!")
    _crashed = False
    try:
        await asyncio.gather(
            dp.start_polling(bot),
            _run_web(svc),
        )
    except Exception as exc:
        _crashed = True
        logger.exception("Критическая ошибка: %s", exc)
        await _alert(f"❌ BEEBOT упал: {exc}")
        raise
    finally:
        if not _crashed:
            await _alert("🔴 BEEBOT остановлен")
        await svc.close()


if __name__ == "__main__":
    asyncio.run(main())
