"""Резервное копирование данных BEEBOT на Яндекс Диск.

Стратегия:
- Ежедневно: data/memory.db → /BEEBOT/daily/memory_YYYY-MM-DD.db
- Еженедельно (воскресенье): CRM-экспорт CSV → /BEEBOT/weekly/crm_YYYY-MM-DD.csv
- Хранить последние 30 ежедневных бэкапов (старые удалять)

Требует:
- YADISK_TOKEN — OAuth-токен Яндекс Диска (get.token / OAuth через yandex.ru/dev)
- `pip install yadisk` уже в зависимостях (yadisk>=2.0)

Запуск (вручную или через asyncio.create_task из bot.py):
    backup = BackupManager()
    asyncio.create_task(backup.run())
"""

from __future__ import annotations

import asyncio
import csv
import io
import logging
import os
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Папка на Яндекс Диске
_YD_ROOT = "/BEEBOT"
_YD_DAILY = f"{_YD_ROOT}/daily"
_YD_WEEKLY = f"{_YD_ROOT}/weekly"

# Сколько ежедневных бэкапов хранить
_KEEP_DAILY = 30

# Интервал между проверками (1 час)
_CHECK_INTERVAL = 3600


class BackupManager:
    """Менеджер резервного копирования на Яндекс Диск."""

    def __init__(
        self,
        token: str | None = None,
        memory_db_path: Path | None = None,
        crm=None,
    ):
        self._token = token or os.getenv("YADISK_TOKEN", "")
        self._memory_db = memory_db_path
        self._crm = crm
        self._last_daily: str | None = None  # дата последнего daily бэкапа
        self._last_weekly: str | None = None  # неделя последнего weekly бэкапа
        self._yd = None  # yadisk.AsyncClient

    @property
    def available(self) -> bool:
        return bool(self._token)

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Фоновый цикл: проверяет раз в час, нужен ли бэкап."""
        if not self.available:
            logger.info("BackupManager: YADISK_TOKEN не задан — бэкапы отключены.")
            return

        try:
            import yadisk
            self._yd = yadisk.AsyncClient(token=self._token)
        except ImportError:
            logger.warning("BackupManager: пакет yadisk не установлен — бэкапы отключены.")
            return

        await self._ensure_dirs()
        logger.info("BackupManager запущен (интервал %d сек).", _CHECK_INTERVAL)

        while True:
            try:
                await self._maybe_daily()
                await self._maybe_weekly()
            except Exception as exc:
                logger.warning("BackupManager: ошибка при бэкапе: %s", exc)
            await asyncio.sleep(_CHECK_INTERVAL)

    async def backup_now(self) -> dict:
        """Принудительный немедленный бэкап (для /admin команды).

        Returns:
            {"daily": "путь или ошибка", "weekly": "..."}
        """
        result = {}
        if not self.available:
            return {"error": "YADISK_TOKEN не задан"}
        try:
            import yadisk
            if not self._yd:
                self._yd = yadisk.AsyncClient(token=self._token)
            await self._ensure_dirs()
            daily_path = await self._do_daily()
            result["daily"] = daily_path or "skipped"
            weekly_path = await self._do_weekly(force=True)
            result["weekly"] = weekly_path or "skipped"
        except Exception as exc:
            result["error"] = str(exc)
        return result

    # ------------------------------------------------------------------
    # Внутренние методы
    # ------------------------------------------------------------------

    async def _ensure_dirs(self) -> None:
        """Создать папки на Яндекс Диске если не существуют."""
        assert self._yd is not None
        for folder in (_YD_ROOT, _YD_DAILY, _YD_WEEKLY):
            try:
                if not await self._yd.exists(folder):
                    await self._yd.mkdir(folder)
                    logger.info("BackupManager: создана папка %s", folder)
            except Exception as exc:
                logger.warning("BackupManager: не удалось создать %s: %s", folder, exc)

    async def _maybe_daily(self) -> None:
        today = datetime.now().strftime("%Y-%m-%d")
        if self._last_daily == today:
            return
        path = await self._do_daily()
        if path:
            self._last_daily = today

    async def _maybe_weekly(self) -> None:
        now = datetime.now()
        if now.weekday() != 6:  # только воскресенье
            return
        week_key = now.strftime("%Y-W%W")
        if self._last_weekly == week_key:
            return
        path = await self._do_weekly()
        if path:
            self._last_weekly = week_key

    async def _do_daily(self) -> str | None:
        """Загрузить memory.db на Яндекс Диск."""
        if not self._memory_db or not self._memory_db.exists():
            logger.warning("BackupManager: memory.db не найден (%s)", self._memory_db)
            return None

        date_str = datetime.now().strftime("%Y-%m-%d")
        remote_path = f"{_YD_DAILY}/memory_{date_str}.db"

        try:
            with open(self._memory_db, "rb") as f:
                data = f.read()
            buf = io.BytesIO(data)
            await self._yd.upload(buf, remote_path, overwrite=True)
            logger.info("BackupManager: daily бэкап → %s (%d байт)", remote_path, len(data))
            await self._cleanup_daily()
            return remote_path
        except Exception as exc:
            logger.warning("BackupManager: ошибка daily бэкапа: %s", exc)
            return None

    async def _do_weekly(self, force: bool = False) -> str | None:
        """Экспортировать заказы CRM в CSV и загрузить на Яндекс Диск."""
        if not self._crm:
            logger.info("BackupManager: CRM не подключена — weekly бэкап пропущен.")
            return None

        date_str = datetime.now().strftime("%Y-%m-%d")
        remote_path = f"{_YD_WEEKLY}/crm_{date_str}.csv"

        try:
            orders = await self._crm.get_orders()
            buf = io.StringIO()
            writer = csv.writer(buf)
            writer.writerow(["id", "date", "status", "client", "total", "address"])
            for o in orders:
                writer.writerow([
                    o.id,
                    o.date,
                    o.status,
                    o.client_name or "",
                    o.total or 0,
                    o.address or "",
                ])
            csv_bytes = buf.getvalue().encode("utf-8-sig")
            await self._yd.upload(io.BytesIO(csv_bytes), remote_path, overwrite=True)
            logger.info(
                "BackupManager: weekly CRM-экспорт → %s (%d заказов)", remote_path, len(orders)
            )
            return remote_path
        except Exception as exc:
            logger.warning("BackupManager: ошибка weekly CRM-экспорта: %s", exc)
            return None

    async def _cleanup_daily(self) -> None:
        """Удалить старые ежедневные бэкапы (оставить последние _KEEP_DAILY)."""
        try:
            files = []
            async for item in self._yd.listdir(_YD_DAILY):
                if item.name.startswith("memory_") and item.name.endswith(".db"):
                    files.append(item.name)
            files.sort()
            excess = files[: max(0, len(files) - _KEEP_DAILY)]
            for name in excess:
                await self._yd.remove(f"{_YD_DAILY}/{name}", permanently=True)
                logger.info("BackupManager: удалён старый бэкап %s", name)
        except Exception as exc:
            logger.warning("BackupManager: ошибка очистки daily: %s", exc)
