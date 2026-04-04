"""Unified startup — создание всех сервисов в одном месте.

Единая точка инициализации: и бот, и веб-бэкенд вызывают
create_services() вместо дублирования логики создания CRM-клиентов,
агентов и сервисов.

Best practice: Cosmic Python Ch.4 — Service Layer,
FastAPI Lifespan pattern — singleton сервисы через app.state.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Optional

from src.config import (
    ADMIN_IDS,
    BEEKEEPER_CHAT_ID,
    WORKER_CHAT_IDS,
)
from src import config as app_config
from src.crm_factory import get_crm_client
from src.services.auth_service import AuthService
from src.services.order_service import OrderService
from src.services.notification_service import NotificationService
from src.services.analytics_service import AnalyticsService
from src.services.consult_service import ConsultService
from src.services.worker_service import WorkerService
from src.services.delivery_service import DeliveryService
from src.services.dashboard_service import DashboardService
from src.services.state_store import StateStore

logger = logging.getLogger(__name__)


AlertFn = Callable[[str], Coroutine[Any, Any, None]]


@dataclass
class Services:
    """Контейнер всех сервисов приложения.

    Создаётся один раз в create_services(), передаётся и боту, и бэкенду.
    """

    # Core
    crm: Optional[Any] = None
    auth: Optional[AuthService] = None
    order_service: Optional[OrderService] = None
    notification_service: Optional[NotificationService] = None
    analytics_service: Optional[AnalyticsService] = None
    consult_service: Optional[ConsultService] = None
    worker_service: Optional[WorkerService] = None
    delivery_service: Optional[DeliveryService] = None
    dashboard_service: Optional[DashboardService] = None
    state_store: Optional[StateStore] = None

    # Agents (тонкие обёртки)
    orchestrator: Optional[Any] = None
    analyst: Optional[Any] = None
    admin_chat_agent: Optional[Any] = None
    inspector: Optional[Any] = None
    logist: Optional[Any] = None

    # Infrastructure
    kb: Optional[Any] = None
    gift_broker: Optional[Any] = None
    crm_agent: Optional[Any] = None
    anamnesis_cache: Optional[Any] = None
    agent_specs: Optional[Any] = None
    bg_manager: Optional[Any] = None

    # State
    _closed: bool = field(default=False, repr=False)

    async def close(self) -> None:
        """Graceful shutdown всех ресурсов."""
        if self._closed:
            return
        self._closed = True

        if self.bg_manager:
            await self.bg_manager.stop_all()

        if self.state_store:
            try:
                await self.state_store.close()
            except Exception as e:
                logger.warning("Ошибка закрытия StateStore: %s", e)

        if self.crm:
            try:
                await self.crm.close()
            except Exception as e:
                logger.warning("Ошибка закрытия CRM: %s", e)


async def create_services(
    *,
    alert_fn: Optional[AlertFn] = None,
    send_telegram: Optional[Callable] = None,
) -> Services:
    """Создать и инициализировать все сервисы.

    Args:
        alert_fn: async-функция для алертов пчеловоду (Telegram).
        send_telegram: async-функция отправки сообщений в Telegram
                       (для NotificationService).

    Returns:
        Services — контейнер со всеми инициализированными сервисами.
    """
    svc = Services()

    # --- StateStore (Redis shared state) ---
    from src.config import REDIS_URL
    svc.state_store = StateStore()
    await svc.state_store.connect_redis(REDIS_URL)

    # --- AuthService ---
    svc.auth = AuthService(
        admin_ids=ADMIN_IDS,
        worker_ids=WORKER_CHAT_IDS,
        beekeeper_id=BEEKEEPER_CHAT_ID,
    )

    # --- Агенты и KB ---
    from src.orchestrator import Orchestrator
    from src.agents.inspector import InspectorAgent
    from src.agents.logist import LogistAgent
    from src.agents.analyst import AnalystAgent
    from src.agents.admin_chat import AdminChatAgent

    svc.orchestrator = Orchestrator()
    svc.inspector = InspectorAgent()
    svc.logist = LogistAgent(beekeeper_chat_id=BEEKEEPER_CHAT_ID)
    svc.analyst = AnalystAgent(
        groq_client=svc.orchestrator._groq,
        groq_model=svc.orchestrator._model,
    )
    svc.admin_chat_agent = AdminChatAgent(
        groq_client=svc.orchestrator._groq,
        model=svc.orchestrator._model,
    )

    # --- KB ---
    try:
        svc.orchestrator.load_kb()
        svc.inspector.kb = svc.orchestrator._beebot.kb
        svc.kb = svc.orchestrator._beebot.kb
        logger.info("Knowledge base loaded: %d chunks", len(svc.kb.chunks))
    except FileNotFoundError:
        logger.error("Knowledge base not found! Run `python -m src.build_kb` first.")
        raise

    # --- ConsultService ---
    svc.consult_service = ConsultService(
        kb=svc.kb,
        llm=svc.orchestrator._groq,
    )

    # --- WorkerService ---
    svc.worker_service = WorkerService()

    # --- CRM ---
    try:
        svc.crm = get_crm_client()
        await svc.crm.authenticate()
        svc.logist.set_crm(svc.crm)
        svc.analyst.set_crm(svc.crm)
        svc.admin_chat_agent.set_crm(svc.crm)

        # --- NotificationService ---
        if send_telegram:
            svc.notification_service = NotificationService(
                send_telegram=send_telegram,
                beekeeper_chat_id=BEEKEEPER_CHAT_ID,
                worker_ids=WORKER_CHAT_IDS,
                get_client_tg_id=svc.crm.get_client_telegram_id,
            )

        # --- OrderService ---
        svc.order_service = OrderService(
            crm=svc.crm, notifier=svc.notification_service,
        )
        svc.logist.set_order_service(svc.order_service)

        # --- AnalyticsService ---
        svc.analytics_service = AnalyticsService(
            crm=svc.crm,
            groq_client=svc.orchestrator._groq,
            groq_model=svc.orchestrator._model,
        )

        # --- DeliveryService ---
        from src.delivery.calculator import DeliveryCalculator
        svc.delivery_service = DeliveryService(
            calculator=DeliveryCalculator(),
        )

        # --- DashboardService ---
        svc.dashboard_service = DashboardService(crm=svc.crm)

        # --- KB keyword-буст из CRM ---
        try:
            products = await svc.crm.get_products()
            names = [p.name for p in products if p.name]
            added = svc.kb.update_keywords_from_products(names)
            if added:
                logger.info(
                    "KB keyword-буст: добавлено %d ключей из CRM (%d товаров)",
                    added, len(names),
                )
        except Exception as _e:
            logger.warning("Не удалось обновить keyword-буст из CRM: %s", _e)

        logger.info("Integram CRM подключена — сервисы получили доступ к данным.")
    except Exception as e:
        logger.warning("Integram CRM недоступна: %s — сервисы работают без CRM.", e)

    # --- Онтология ---
    try:
        await svc.orchestrator.load_ontology()
    except Exception as _e:
        logger.warning("Онтология недоступна (продолжаем без неё): %s", _e)

    # --- Gift Protocol ---
    from src.crm_agent import CrmAgent
    from src.anamnesis import AnamnesisCache
    from src.gift_protocol import GiftBroker

    svc.crm_agent = CrmAgent(svc.crm)
    svc.orchestrator.set_crm_agent(svc.crm_agent)
    svc.anamnesis_cache = AnamnesisCache(svc.orchestrator._memory)
    svc.gift_broker = GiftBroker(
        orchestrator=svc.orchestrator,
        context_store=svc.orchestrator._shared_ctx,
        anamnesis=svc.anamnesis_cache,
        crm_agent=svc.crm_agent,
    )
    logger.info("Gift Protocol инициализирован (CrmAgent.available=%s)", svc.crm_agent.available)

    # --- AgentSpecsCache ---
    from src.agent_specs import AgentSpecsCache

    svc.agent_specs = AgentSpecsCache()
    try:
        await svc.agent_specs.load()
    except Exception as _e:
        logger.warning("AgentSpecs: недоступны (продолжаем без них): %s", _e)
    svc.orchestrator.set_agent_specs(svc.agent_specs)

    # --- BackgroundTaskManager ---
    from src.web.bg_tasks import BackgroundTaskManager

    svc.bg_manager = BackgroundTaskManager(alert_fn=alert_fn)

    return svc


async def start_background_tasks(
    svc: Services,
    *,
    bot: Any = None,
    alert_fn: Optional[AlertFn] = None,
) -> None:
    """Запустить фоновые задачи (CRM snapshot, трекинг, UDS, мониторинг, бэкапы).

    Args:
        svc: контейнер сервисов.
        bot: aiogram Bot (для UDS Poller уведомлений).
        alert_fn: async-функция алертов.
    """
    import src.routers._state as _state

    bg = svc.bg_manager
    if not bg:
        return

    # --- CRM Snapshot ---
    if svc.crm:
        from src.crm_snapshot import CrmSnapshot
        _state._crm_snapshot = CrmSnapshot(svc.crm, alert_fn=alert_fn)
        await bg.start("crm_snapshot", _state._crm_snapshot.run)
        logger.info("CRM snapshot запущен.")

    # --- Авто-трекинг ---
    if svc.crm:
        from src.delivery.tracker import OrderTracker
        from src.web.notifications import notify_client_status_change
        order_tracker = OrderTracker(
            crm=svc.crm,
            notify_fn=notify_client_status_change,
        )
        await bg.start("order_tracker", order_tracker.run)
        logger.info("Авто-трекинг отправлений запущен.")

    # --- UDS Poller ---
    if app_config.UDS_API_KEY and app_config.UDS_COMPANY_ID and svc.crm:
        try:
            from src.integrations.uds import UDSClient, UDSPoller
            uds_client = UDSClient()
            uds_poller = UDSPoller(
                uds_client=uds_client,
                integram_client=svc.crm,
                bot=bot,
                notify_chat_id=BEEKEEPER_CHAT_ID,
            )
            await bg.start("uds_poller", uds_poller.run)
            logger.info("UDS Poller запущен.")
        except Exception as e:
            logger.warning("UDS Poller не удалось запустить: %s", e)

    # --- TunnelMonitor ---
    from src.tunnel_monitor import TunnelMonitor
    tunnel_monitor = TunnelMonitor(alert_fn=alert_fn)
    if svc.orchestrator and hasattr(svc.orchestrator, '_beebot'):
        svc.orchestrator._beebot.tunnel_monitor = tunnel_monitor
    if svc.consult_service:
        svc.consult_service.tunnel_monitor = tunnel_monitor
    await bg.start("tunnel_monitor", tunnel_monitor.run)
    logger.info("TunnelMonitor запущен.")

    # --- BackupManager ---
    from src.backup import BackupManager
    backup = BackupManager(
        memory_db_path=app_config.MEMORY_DB_PATH,
        crm=svc.crm,
    )
    await bg.start("backup", backup.run)
    if backup.available:
        logger.info("BackupManager запущен.")
    else:
        logger.info("BackupManager: YADISK_TOKEN не задан — бэкапы отключены.")
