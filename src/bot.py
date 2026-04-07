"""BEEBOT — точка входа. Микроядерная архитектура.

Регистрирует плагины в BeeBotApp и запускает единый процесс:
  Telegram polling + FastAPI (uvicorn) в одном event loop.

Добавление нового модуля = создать Plugin-подкласс + app.register(NewPlugin()).
"""

import asyncio
import logging
import os

from aiogram import Bot, Dispatcher
from aiogram.fsm.storage.memory import MemoryStorage

from src.config import TELEGRAM_BOT_TOKEN, BEEKEEPER_CHAT_ID, TG_SOCKS_PROXY
from src.logging_config import setup_logging
from src.kernel import BeeBotApp

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Telegram helpers (нужны до создания плагинов)                               #
# --------------------------------------------------------------------------- #

_bot: Bot | None = None


async def _alert(text: str) -> None:
    if not BEEKEEPER_CHAT_ID or _bot is None:
        return
    try:
        await _bot.send_message(BEEKEEPER_CHAT_ID, text)
    except Exception as e:
        logger.warning("Не удалось отправить алерт: %s", e)


async def _send_tg(chat_id: int, text: str) -> bool:
    if _bot is None:
        return False
    try:
        await _bot.send_message(chat_id, text)
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Запуск uvicorn (FastAPI) в том же event loop                                 #
# --------------------------------------------------------------------------- #

async def _run_web() -> None:
    import uvicorn
    from src.web.server import app as fastapi_app

    host = os.getenv("WEB_HOST", "0.0.0.0")
    port = int(os.getenv("WEB_PORT", "8088"))
    config = uvicorn.Config(fastapi_app, host=host, port=port, log_level="info")
    await uvicorn.Server(config).serve()


# --------------------------------------------------------------------------- #
# main                                                                         #
# --------------------------------------------------------------------------- #

async def main() -> None:
    global _bot

    setup_logging()

    # Создаём Bot (с SOCKS5-прокси если нужно)
    if TG_SOCKS_PROXY:
        from aiogram.client.session.aiohttp import AiohttpSession
        _bot = Bot(token=TELEGRAM_BOT_TOKEN, session=AiohttpSession(proxy=TG_SOCKS_PROXY))
        logger.info("Telegram via SOCKS5: %s", TG_SOCKS_PROXY)
    else:
        _bot = Bot(token=TELEGRAM_BOT_TOKEN)

    dp = Dispatcher(storage=MemoryStorage())

    # ---------------------------------------------------------------------- #
    # Регистрация плагинов                                                    #
    # ---------------------------------------------------------------------- #
    from src.plugins.crm import CrmPlugin
    from src.plugins.knowledge import KnowledgePlugin
    from src.plugins.agents import AgentsPlugin
    from src.plugins.orders import OrdersPlugin
    from src.plugins.analytics import AnalyticsPlugin
    from src.plugins.delivery import DeliveryPlugin
    from src.plugins.workers import WorkersPlugin
    from src.plugins.gift import GiftPlugin
    from src.plugins.monitoring import MonitoringPlugin
    from src.plugins.telegram import TelegramPlugin
    from src.plugins.web import WebPlugin

    app = (
        BeeBotApp(_bot, dp)
        .register(CrmPlugin())
        .register(KnowledgePlugin())
        .register(AgentsPlugin())
        .register(OrdersPlugin(send_telegram=_send_tg))
        .register(AnalyticsPlugin())
        .register(DeliveryPlugin())
        .register(WorkersPlugin(bot=_bot))
        .register(GiftPlugin())
        .register(MonitoringPlugin(alert_fn=_alert, bot=_bot))
        .register(TelegramPlugin(bot=_bot, alert_fn=_alert))
        .register(WebPlugin(bot=_bot))
    )

    # ---------------------------------------------------------------------- #
    # Запуск                                                                  #
    # ---------------------------------------------------------------------- #
    await app.start()

    dp.startup.register(lambda **_: _alert("🟢 BEEBOT запущен"))

    logger.info("BEEBOT запущен (unified: bot + web).")
    _crashed = False
    try:
        await asyncio.gather(
            dp.start_polling(_bot),
            _run_web(),
        )
    except Exception as exc:
        _crashed = True
        logger.exception("Критическая ошибка: %s", exc)
        await _alert(f"❌ BEEBOT упал: {exc}")
        raise
    finally:
        if not _crashed:
            await _alert("🔴 BEEBOT остановлен")
        await app.stop()


if __name__ == "__main__":
    asyncio.run(main())
