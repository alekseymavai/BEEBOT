"""MonitoringPlugin — фоновые задачи: CrmSnapshot, OrderTracker, UDSPoller,
TunnelMonitor, BackupManager.

Зависимости: crm

Публикует в контейнере:
  "crm_snapshot"   → CrmSnapshot
  "bg_manager"     → BackgroundTaskManager

BG задачи (через get_bg_tasks()):
  crm_snapshot, order_tracker, uds_poller, tunnel_monitor, backup
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Coroutine, Optional

from src.kernel.plugin import Plugin, BgTask

if TYPE_CHECKING:
    from src.kernel.container import Container

logger = logging.getLogger(__name__)

AlertFn = Optional[Callable[[str], Coroutine[Any, Any, None]]]


class MonitoringPlugin(Plugin):
    name = "monitoring"
    dependencies = ["crm"]

    def __init__(self, alert_fn: AlertFn = None, bot: Any = None) -> None:
        self._alert_fn = alert_fn
        self._bot = bot
        self._tasks: list[BgTask] = []

    async def setup(self, container: "Container") -> None:
        from src.web.bg_tasks import BackgroundTaskManager

        crm = container.get("crm")
        bg_manager = BackgroundTaskManager(alert_fn=self._alert_fn)
        container.set("bg_manager", bg_manager)

        # --- CrmSnapshot ---
        if crm:
            from src.crm_snapshot import CrmSnapshot
            import src.routers._state as _state

            snapshot = CrmSnapshot(crm, alert_fn=self._alert_fn)
            _state._crm_snapshot = snapshot
            container.set("crm_snapshot", snapshot)
            self._tasks.append(BgTask("crm_snapshot", snapshot.run))

        # --- OrderTracker ---
        if crm:
            from src.delivery.tracker import OrderTracker
            from src.web.notifications import notify_client_status_change

            tracker = OrderTracker(crm=crm, notify_fn=notify_client_status_change)
            self._tasks.append(BgTask("order_tracker", tracker.run))

        # --- UDSPoller ---
        import src.config as app_config

        if app_config.UDS_API_KEY and app_config.UDS_COMPANY_ID and crm:
            try:
                from src.integrations.uds import UDSClient, UDSPoller
                from src.config import BEEKEEPER_CHAT_ID

                uds_poller = UDSPoller(
                    uds_client=UDSClient(),
                    integram_client=crm,
                    bot=self._bot,
                    notify_chat_id=BEEKEEPER_CHAT_ID,
                )
                self._tasks.append(BgTask("uds_poller", uds_poller.run))
            except Exception as e:
                logger.warning("UDSPoller не удалось инициализировать: %s", e)

        # --- TunnelMonitor ---
        from src.tunnel_monitor import TunnelMonitor

        tunnel_monitor = TunnelMonitor(alert_fn=self._alert_fn)
        self._tasks.append(BgTask("tunnel_monitor", tunnel_monitor.run))

        # Привязать TunnelMonitor к агентам, если доступны
        orchestrator = container.get("orchestrator")
        if orchestrator and hasattr(orchestrator, "_beebot"):
            orchestrator._beebot.tunnel_monitor = tunnel_monitor
        consult_service = container.get("consult_service")
        if consult_service:
            consult_service.tunnel_monitor = tunnel_monitor

        # --- BackupManager ---
        from src.backup import BackupManager

        backup = BackupManager(
            memory_db_path=app_config.MEMORY_DB_PATH,
            crm=crm,
        )
        self._tasks.append(BgTask("backup", backup.run))
        if backup.available:
            logger.info("BackupManager: Яндекс Диск бэкапы включены.")
        else:
            logger.info("BackupManager: YADISK_TOKEN не задан — бэкапы отключены.")

        logger.info(
            "MonitoringPlugin: %d фоновых задач подготовлено.", len(self._tasks)
        )

    def get_bg_tasks(self) -> list[BgTask]:
        return self._tasks
