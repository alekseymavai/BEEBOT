"""Telegram bot for BEEBOT — AI assistant for a beekeeper blog."""

import asyncio
import logging
from typing import Optional

from aiogram import Bot, Dispatcher, types
from aiogram.fsm.storage.memory import MemoryStorage

from src.config import (
    TELEGRAM_BOT_TOKEN, BASE_DIR, BEEKEEPER_CHAT_ID, ADMIN_IDS,
    TG_SOCKS_PROXY, WORKER_CHAT_IDS,
)
from src import config as app_config
from src.delivery.tracker import OrderTracker
from src.integrations.uds import UDSClient, UDSPoller
from src.integram_client import IntegramClient
from src.agents.logist import LogistAgent
from src.agents.inspector import InspectorAgent
from src.agents.admin_chat import AdminChatAgent
from src.orchestrator import Orchestrator
from src.agents.analyst import AnalystAgent
from src.crm_agent import CrmAgent
from src.anamnesis import AnamnesisCache
from src.gift_protocol import GiftBroker
from src.agent_specs import AgentSpecsCache
from src.agent_bus import create_agent_bus_client
from src.backup import BackupManager
from src.admin import router as admin_router, setup_admin
from src.logging_config import setup_logging

# Роутеры из src/routers/
from src.routers.inspect import router as inspect_router, setup_inspect
from src.routers.fsm_order import router as fsm_order_router, setup_fsm_order
from src.routers.worker import router as worker_router, setup_worker
from src.routers.bot_admin import router as bot_admin_router, setup_bot_admin
from src.routers.user import router as user_router, setup_user
import src.routers._state as _state

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


@dp.startup()
async def on_startup(**_kwargs) -> None:
    """Отправить алерт о старте после установки соединения с Telegram."""
    await _alert("🟢 BEEBOT запущен")


async def main():
    global bot
    setup_logging()

    if TG_SOCKS_PROXY:
        from aiogram.client.session.aiohttp import AiohttpSession
        bot = Bot(token=TELEGRAM_BOT_TOKEN, session=AiohttpSession(proxy=TG_SOCKS_PROXY))
        logger.info("Telegram via SOCKS5 proxy: %s", TG_SOCKS_PROXY)

    logger.info("Starting BEEBOT...")

    # --- Загрузка базы знаний ---
    orchestrator = Orchestrator()
    inspector = InspectorAgent()
    logist = LogistAgent(beekeeper_chat_id=BEEKEEPER_CHAT_ID)
    analyst = AnalystAgent(
        groq_client=orchestrator._groq,
        groq_model=orchestrator._model,
    )
    admin_chat_agent = AdminChatAgent(
        groq_client=orchestrator._groq,
        model=orchestrator._model,
    )

    try:
        orchestrator.load_kb()
        inspector.kb = orchestrator._beebot.kb
        kb = orchestrator._beebot.kb
        logger.info("Knowledge base loaded: %d chunks", len(kb.chunks))
    except FileNotFoundError:
        logger.error("Knowledge base not found! Run `python -m src.build_kb` first.")
        return

    # --- Integram CRM ---
    integram_client: Optional[IntegramClient] = None
    if app_config.INTEGRAM_URL and app_config.INTEGRAM_LOGIN:
        try:
            integram_client = IntegramClient()
            await integram_client.authenticate()
            logist._crm = integram_client
            analyst._crm = integram_client
            try:
                products = await integram_client.get_products()
                names = [p.name for p in products if p.name]
                added = kb.update_keywords_from_products(names)
                if added:
                    logger.info("KB keyword-буст: добавлено %d ключей из CRM (%d товаров)", added, len(names))
            except Exception as _e:
                logger.warning("Не удалось обновить keyword-буст из CRM: %s", _e)
            logger.info("Integram CRM подключена — агенты получили доступ к данным.")
        except Exception as e:
            logger.warning("Integram CRM недоступна: %s — агенты работают без CRM.", e)
    else:
        logger.info("Integram CRM не настроена — агенты работают без CRM.")

    # --- Онтология ---
    try:
        await orchestrator.load_ontology()
    except Exception as _e:
        logger.warning("Онтология недоступна (продолжаем без неё): %s", _e)

    admin_chat_agent.set_crm(integram_client)
    setup_admin(bot, crm=integram_client, kb=kb, memory=orchestrator._memory)

    # --- Gift Protocol: CrmAgent + AnamnesisCache + GiftBroker (Фаза 9) ---
    crm_agent = CrmAgent(integram_client)   # None если integram_client=None — ОК
    orchestrator.set_crm_agent(crm_agent)
    anamnesis_cache = AnamnesisCache(orchestrator._memory)
    gift_broker = GiftBroker(
        orchestrator=orchestrator,
        context_store=orchestrator._shared_ctx,
        anamnesis=anamnesis_cache,
        crm_agent=crm_agent,
    )
    logger.info("Gift Protocol инициализирован (CrmAgent.available=%s)", crm_agent.available)

    # --- AgentSpecsCache — спецификации агентов из Integram (Фаза 9.5) ---
    agent_specs = AgentSpecsCache()
    try:
        await agent_specs.load()
    except Exception as _e:
        logger.warning("AgentSpecs: недоступны (продолжаем без них): %s", _e)
    orchestrator.set_agent_specs(agent_specs)

    # --- Инициализация роутеров ---
    setup_inspect(inspector)
    setup_fsm_order(logist, bot)
    setup_bot_admin(analyst, orchestrator, admin_chat_agent, inspector, bot)
    setup_user(orchestrator, admin_chat_agent, logist, gift_broker=gift_broker)

    # --- Режим работника склада ---
    if integram_client:
        setup_worker(integram_client, bot, gift_broker=gift_broker)
        from src.notifications import Notifier
        import src.notifications as _notif_module
        _notif_module._worker_notifier = Notifier(bot)
        if WORKER_CHAT_IDS:
            logger.info("Режим работника склада включён (%d работников).", len(WORKER_CHAT_IDS))

    # --- CRM Snapshot ---
    if integram_client:
        from src.crm_snapshot import CrmSnapshot
        _state._crm_snapshot = CrmSnapshot(integram_client, alert_fn=_alert)
        asyncio.create_task(_state._crm_snapshot.run())
        logger.info("CRM snapshot запущен (интервал %d сек).", _state._crm_snapshot._refresh_interval)

    # --- Авто-трекинг ---
    order_tracker: Optional[OrderTracker] = None
    if integram_client:
        from src.web.notifications import notify_client_status_change
        order_tracker = OrderTracker(
            crm=integram_client,
            notify_fn=notify_client_status_change,
        )
        asyncio.create_task(order_tracker.run())
        logger.info("Авто-трекинг отправлений запущен.")

    # --- UDS Poller ---
    uds_poller: Optional[UDSPoller] = None
    uds_client: Optional[UDSClient] = None

    if app_config.UDS_API_KEY and app_config.UDS_COMPANY_ID:
        if integram_client:
            try:
                uds_client = UDSClient()
                uds_poller = UDSPoller(
                    uds_client=uds_client,
                    integram_client=integram_client,
                    bot=bot,
                    notify_chat_id=BEEKEEPER_CHAT_ID,
                )
                asyncio.create_task(uds_poller.run())
                logger.info("UDS Poller запущен.")
            except Exception as e:
                logger.warning("UDS Poller не удалось запустить: %s", e)
        else:
            logger.warning("UDS Poller пропущен — Integram CRM не подключена.")
    else:
        logger.info("UDS не настроен — поллер пропущен.")

    # --- TunnelMonitor — мониторинг SSH-туннеля к Groq (Фаза 12.1) ---
    from src.tunnel_monitor import TunnelMonitor
    _tunnel_monitor = TunnelMonitor(alert_fn=_alert)
    orchestrator._beebot.tunnel_monitor = _tunnel_monitor
    asyncio.create_task(_tunnel_monitor.run())
    logger.info("TunnelMonitor запущен (порт 8990).")

    # --- BackupManager — Яндекс Диск (Фаза 12.2) ---
    _backup = BackupManager(
        memory_db_path=app_config.MEMORY_DB_PATH,
        crm=integram_client,
    )
    asyncio.create_task(_backup.run())
    if _backup.available:
        logger.info("BackupManager запущен (daily memory.db + weekly CRM CSV).")
    else:
        logger.info("BackupManager: YADISK_TOKEN не задан — бэкапы отключены.")

    # --- AgentBus — регистрация в dronedoc2026 (Фаза 11.2) ---
    _agent_bus = create_agent_bus_client()
    if _agent_bus:
        await _agent_bus.start(
            kb=orchestrator._beebot.kb,
            crm=integram_client,
        )
    else:
        logger.info("AgentBus: AGENT_BUS_URL не задан — работаем автономно.")

    logger.info("Bot is running!")
    _crashed = False
    try:
        await dp.start_polling(bot)
    except Exception as exc:
        _crashed = True
        logger.exception("Критическая ошибка бота: %s", exc)
        await _alert(f"❌ BEEBOT упал: {exc}")
        raise
    finally:
        if not _crashed:
            await _alert("🔴 BEEBOT остановлен")
        if order_tracker:
            order_tracker.stop()
        if uds_poller:
            uds_poller.stop()
        if uds_client:
            await uds_client.close()
        if integram_client:
            await integram_client.close()


if __name__ == "__main__":
    asyncio.run(main())
