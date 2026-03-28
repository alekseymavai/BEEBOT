"""Интеграция с UDS (Unified Discount System).

Получение транзакций из UDS и автоматическое создание заказов в Integram CRM.

Поток:
  UDS транзакция → polling → найти/создать клиента (по телефону)
                           → найти/создать товары (по артикулу UDS)
                           → создать заказ (источник "UDS")
                           → уведомить пчеловода

Конфигурация через .env:
  UDS_API_KEY    — API-ключ UDS
  UDS_COMPANY_ID — ID компании в UDS
"""

from __future__ import annotations

import asyncio
import base64
import logging
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

from src import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_UDS_BASE_URL = getattr(config, "UDS_BASE_URL", None) or "https://api.uds.app/partner/v2"
_POLL_INTERVAL_SECONDS = 300  # опрос каждые 5 минут
_MAX_RETRIES = 3
_RETRY_BACKOFF_BASE = 1.0


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class UDSError(Exception):
    """Базовое исключение UDS-клиента."""


class UDSAuthError(UDSError):
    """Ошибка аутентификации UDS."""


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _parse_transaction(data: dict) -> dict:
    """Нормализовать транзакцию UDS в общий формат."""
    customer = data.get("customer") or data.get("user") or {}
    goods = data.get("receipt", {}).get("items") or data.get("items") or []
    return {
        "id": str(data.get("id", "")),
        "created_at": data.get("dateCreated") or data.get("created_at") or "",
        "total": float(data.get("totalPurchase") or data.get("total") or 0.0),
        "customer_phone": (
            customer.get("phone")
            or customer.get("mobilePhone")
            or ""
        ),
        "customer_name": (
            customer.get("displayName")
            or customer.get("name")
            or ""
        ),
        "customer_uds_id": str(customer.get("uid") or customer.get("id") or ""),
        "goods": [
            {
                "sku": str(item.get("product", {}).get("externalId") or item.get("sku") or ""),
                "name": item.get("product", {}).get("name") or item.get("name") or "",
                "quantity": int(item.get("count") or item.get("quantity") or 1),
                "unit_price": float(item.get("price") or 0.0),
            }
            for item in goods
        ],
    }


# ---------------------------------------------------------------------------
# UDSClient
# ---------------------------------------------------------------------------


class UDSClient:
    """Клиент для работы с API UDS.

    Использование::

        client = UDSClient()
        transactions = await client.get_transactions()

    Или через polling::

        async with UDSPoller(uds_client, integram_client, bot) as poller:
            await poller.run()
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        company_id: Optional[str] = None,
        base_url: str = _UDS_BASE_URL,
    ) -> None:
        self._api_key = api_key or config.UDS_API_KEY or ""
        self._company_id = company_id or config.UDS_COMPANY_ID or ""
        self._base_url = base_url.rstrip("/")
        self._http: Optional[httpx.AsyncClient] = None

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    async def _get_http(self) -> httpx.AsyncClient:
        if self._http is None or self._http.is_closed:
            # UDS Partner API использует Basic Auth: companyId:apiKey
            creds = base64.b64encode(
                f"{self._company_id}:{self._api_key}".encode()
            ).decode()
            self._http = httpx.AsyncClient(
                base_url=self._base_url,
                headers={
                    "Authorization": f"Basic {creds}",
                    "Accept": "application/json",
                    "Accept-Language": "ru",
                },
                timeout=30.0,
            )
        return self._http

    async def close(self) -> None:
        """Закрыть HTTP-сессию."""
        if self._http and not self._http.is_closed:
            await self._http.aclose()

    async def __aenter__(self) -> "UDSClient":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Retry logic
    # ------------------------------------------------------------------

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[dict] = None,
        json: Optional[dict] = None,
    ) -> Any:
        """Выполнить HTTP-запрос с retry (3 попытки, exponential backoff)."""
        http = await self._get_http()
        last_exc: Exception = RuntimeError("no attempts made")

        for attempt in range(_MAX_RETRIES):
            try:
                response = await http.request(
                    method,
                    path,
                    params=params,
                    json=json,
                )
                if response.status_code in (401, 403):
                    raise UDSAuthError(
                        f"Ошибка аутентификации UDS (HTTP {response.status_code}). "
                        "Проверьте UDS_API_KEY и UDS_COMPANY_ID."
                    )
                response.raise_for_status()
                return response.json()
            except UDSAuthError:
                raise
            except Exception as exc:
                last_exc = exc
                if attempt < _MAX_RETRIES - 1:
                    wait = _RETRY_BACKOFF_BASE * (2 ** attempt)
                    logger.warning(
                        "UDS: попытка %d/%d не удалась: %s. Повтор через %.1f с.",
                        attempt + 1,
                        _MAX_RETRIES,
                        exc,
                        wait,
                    )
                    await asyncio.sleep(wait)

        raise UDSError(f"UDS: все {_MAX_RETRIES} попытки исчерпаны: {last_exc}") from last_exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_transactions(self, limit: int = 50) -> list[dict]:
        """Получить последние транзакции (первая страница, newest-first).

        Args:
            limit: максимальное число транзакций в ответе (макс 50).

        Returns:
            Список нормализованных транзакций.
        """
        data = await self._request(
            "GET",
            "/operations",
            params={"max": limit},
        )
        rows = data if isinstance(data, list) else data.get("rows", data.get("items", []))
        return [_parse_transaction(row) for row in rows]

    async def get_transactions_since(
        self,
        since: datetime,
        page_size: int = 50,
    ) -> list[dict]:
        """Получить все транзакции начиная с даты ``since`` (cursor-пагинация).

        UDS API не поддерживает ``skip`` — используем ``cursor`` из ответа
        для пагинации от новых к старым. Останавливаемся, когда встречаем
        транзакцию старше ``since``.

        Args:
            since: дата, начиная с которой нужны транзакции (UTC).
            page_size: размер страницы (макс 50).

        Returns:
            Список нормализованных транзакций (newest-first).
        """
        txs: list[dict] = []
        cursor: str | None = None
        since_str = since.strftime("%Y-%m-%d")

        while True:
            params: dict[str, Any] = {"max": page_size}
            if cursor:
                params["cursor"] = cursor

            data = await self._request("GET", "/operations", params=params)
            rows = data if isinstance(data, list) else data.get("rows", data.get("items", []))
            new_cursor = data.get("cursor") if isinstance(data, dict) else None

            if not rows:
                break

            hit_old = False
            for row in rows:
                date_str = row.get("dateCreated", "")[:10]
                if date_str < since_str:
                    hit_old = True
                    break
                txs.append(_parse_transaction(row))

            if hit_old or not new_cursor or new_cursor == cursor:
                break
            cursor = new_cursor
            # Пауза между страницами — UDS API имеет недокументированный
            # rate limit, без задержки catch-up вызывает 429.
            await asyncio.sleep(3)

        return txs

    async def get_customer(self, customer_uds_id: str) -> dict:
        """Получить данные клиента по его UID в UDS.

        Args:
            customer_uds_id: UID клиента в UDS.

        Returns:
            Словарь с данными клиента.
        """
        data = await self._request("GET", f"/customers/{customer_uds_id}")
        return data

    async def get_catalog(self) -> list[dict]:
        """Получить каталог товаров из UDS.

        Returns:
            Список товаров UDS с полями id, name, price, externalId.
        """
        data = await self._request("GET", "/goods")
        rows = data if isinstance(data, list) else data.get("rows", data.get("items", []))
        return rows


# ---------------------------------------------------------------------------
# Deduplication helper
# ---------------------------------------------------------------------------


class TransactionDeduplicator:
    """Хранит ID уже обработанных транзакций UDS.

    При старте загружает существующие ``UDS-*`` заказы из CRM, чтобы
    не дублировать их после рестарта.
    """

    def __init__(self) -> None:
        self._seen: set[str] = set()

    async def load_existing_from_crm(self, integram_client: Any) -> int:
        """Загрузить ID уже созданных UDS-заказов из CRM.

        Returns:
            Количество загруженных ID.
        """
        try:
            orders = await integram_client.get_orders()
            for order in orders:
                if order.number and order.number.startswith("UDS-"):
                    tid = order.number.replace("UDS-", "")
                    self._seen.add(tid)
            logger.info(
                "UDS Dedup: загружено %d существующих UDS-заказов из CRM.",
                len(self._seen),
            )
        except Exception as e:
            logger.error("UDS Dedup: не удалось загрузить заказы из CRM: %s", e)
        return len(self._seen)

    def is_new(self, transaction: dict, since: datetime | None = None) -> bool:
        """Вернуть True, если транзакция новая и ещё не обрабатывалась.

        Args:
            transaction: нормализованная транзакция.
            since: если указано, транзакции старше этой даты считаются «не новыми».
        """
        tid = transaction.get("id", "")
        if tid in self._seen:
            return False

        if since:
            created_raw = transaction.get("created_at", "")
            if created_raw:
                try:
                    created = datetime.fromisoformat(created_raw.replace("Z", "+00:00"))
                    if created < since:
                        return False
                except ValueError:
                    pass

        return True

    def mark_seen(self, transaction_id: str) -> None:
        """Пометить транзакцию как обработанную."""
        self._seen.add(transaction_id)


# ---------------------------------------------------------------------------
# UDS → Integram sync logic
# ---------------------------------------------------------------------------


async def sync_uds_transaction(
    transaction: dict,
    integram_client: Any,
    notify_chat_id: Optional[int] = None,
    bot: Optional[Any] = None,
) -> None:
    """Синхронизировать одну UDS-транзакцию с Integram CRM.

    Args:
        transaction:      Нормализованная транзакция из UDSClient.
        integram_client:  Экземпляр IntegramClient.
        notify_chat_id:   Telegram chat_id для уведомления пчеловода.
        bot:              Экземпляр aiogram Bot (нужен для отправки уведомления).
    """
    tid = transaction["id"]
    phone = transaction["customer_phone"]
    name = transaction["customer_name"] or f"UDS-клиент {transaction['customer_uds_id']}"
    total = transaction["total"]

    # Дата транзакции из UDS (или текущая, если не удалось распарсить)
    order_date = datetime.now(tz=timezone.utc)
    created_raw = transaction.get("created_at", "")
    if created_raw:
        try:
            order_date = datetime.fromisoformat(created_raw.replace("Z", "+00:00"))
        except ValueError:
            pass

    logger.info("UDS: обрабатываем транзакцию %s (клиент: %s, сумма: %.2f)", tid, name, total)

    # 1. Найти или создать клиента по телефону
    client = await _get_or_create_client_by_phone(integram_client, phone, name)

    # 2. Собрать позиции заказа из UDS-транзакции
    items = await _build_order_items(integram_client, transaction["goods"], total)

    # 3. Создать заказ
    items_total = sum(i["quantity"] * i["unit_price"] for i in items)
    order = await integram_client.create_order(
        client.id,
        items,
        source="UDS",
        number=f"UDS-{tid}",
        items_total=items_total,
        total=total,
        status="Новый",
        date=order_date,
    )
    logger.info("UDS: создан заказ #%s (Integram ID=%d)", order.number, order.id)

    # 4. Уведомить пчеловода
    if bot and notify_chat_id:
        await _notify_beekeeper(bot, notify_chat_id, order, client, transaction)


async def _get_or_create_client_by_phone(
    integram_client: Any,
    phone: str,
    name: str,
) -> Any:
    """Найти клиента в Integram по телефону или создать нового."""
    # get_or_create_client уже ищет по телефону и создаёт при отсутствии
    return await integram_client.get_or_create_client(
        telegram_id=0,  # UDS-клиент без Telegram
        full_name=name,
        phone=phone,
        source="UDS",
    )


async def _build_order_items(
    integram_client: Any,
    goods: list[dict],
    fallback_total: float,
) -> list[dict]:
    """Сопоставить товары UDS с каталогом Integram и вернуть позиции заказа.

    Если товар не найден — создаём позицию с нулевым product_id и именем из UDS.
    """
    items = []

    for good in goods:
        sku = good.get("sku", "")
        product_id = 0
        unit_price = good.get("unit_price", 0.0)

        if sku:
            try:
                product = await integram_client.get_product_by_name(good["name"])
                if product:
                    product_id = product.id
                    if not unit_price and product.price:
                        unit_price = product.price
            except Exception as e:
                logger.warning("UDS: не удалось найти товар '%s' (SKU=%s): %s", good["name"], sku, e)

        items.append(
            {
                "product_id": product_id,
                "quantity": good["quantity"],
                "unit_price": unit_price,
            }
        )

    # Если нет позиций — добавить одну сводную строку на всю сумму
    if not items:
        items.append(
            {
                "product_id": 0,
                "quantity": 1,
                "unit_price": fallback_total,
            }
        )

    return items


async def _notify_beekeeper(
    bot: Any,
    chat_id: int,
    order: Any,
    client: Any,
    transaction: dict,
) -> None:
    """Отправить уведомление пчеловоду о новом заказе из UDS."""
    goods_lines = "\n".join(
        f"  • {g['name']} × {g['quantity']}"
        for g in transaction["goods"]
    ) or "  • (позиции не указаны)"

    text = (
        f"🛒 *Новый заказ из UDS*\n\n"
        f"Заказ: *{order.number}*\n"
        f"Клиент: {client.full_name}"
        + (f" ({client.phone})" if client.phone else "")
        + f"\nСумма: *{transaction['total']:.0f} ₽*\n\n"
        f"Состав:\n{goods_lines}"
    )
    try:
        await bot.send_message(chat_id, text, parse_mode="Markdown")
    except Exception as e:
        logger.error("UDS: не удалось отправить уведомление пчеловоду: %s", e)
    for group_id in config.ACTIVE_GROUP_IDS:
        try:
            await bot.send_message(group_id, text, parse_mode="Markdown")
        except Exception as e:
            logger.warning("UDS: не удалось отправить уведомление в группу %d: %s", group_id, e)


# ---------------------------------------------------------------------------
# UDS Poller
# ---------------------------------------------------------------------------


class UDSPoller:
    """Polling-сервис: периодически опрашивает UDS и синхронизирует заказы.

    При запуске:
      1. Загружает из CRM уже существующие UDS-заказы (для дедупликации).
      2. Делает начальный catch-up: cursor-пагинация всех транзакций
         начиная с ``sync_since`` и синхронизирует пропущенные.
      3. Переходит в режим обычного polling (первая страница раз в минуту).

    Запуск::

        poller = UDSPoller(uds_client, integram_client, bot)
        asyncio.create_task(poller.run())
    """

    # Дата, начиная с которой синхронизируем UDS → CRM (10-значные ID).
    # Все транзакции до этой даты — из файла пчеловода (7-значные ID).
    _SYNC_SINCE = datetime(2026, 3, 17, tzinfo=timezone.utc)

    def __init__(
        self,
        uds_client: UDSClient,
        integram_client: Any,
        bot: Optional[Any] = None,
        notify_chat_id: Optional[int] = None,
        poll_interval: float = _POLL_INTERVAL_SECONDS,
    ) -> None:
        self._uds = uds_client
        self._integram = integram_client
        self._bot = bot
        self._notify_chat_id = notify_chat_id or config.BEEKEEPER_CHAT_ID
        self._poll_interval = poll_interval
        self._dedup = TransactionDeduplicator()
        self._running = False

    async def __aenter__(self) -> "UDSPoller":
        return self

    async def __aexit__(self, *_: Any) -> None:
        self.stop()

    def stop(self) -> None:
        """Остановить polling-цикл."""
        self._running = False

    async def run(self) -> None:
        """Запустить polling: catch-up → бесконечный цикл."""
        self._running = True

        # 1. Загрузить существующие UDS-заказы из CRM
        await self._dedup.load_existing_from_crm(self._integram)

        # 2. Начальный catch-up: подтянуть всё с 17.03.2026
        try:
            await self._initial_sync()
        except Exception as e:
            logger.error("UDS Poller: ошибка при начальной синхронизации: %s", e)

        # 3. Обычный polling
        logger.info("UDS Poller: запущен (интервал опроса %ds).", self._poll_interval)
        while self._running:
            try:
                await self._poll_once()
            except Exception as e:
                logger.error("UDS Poller: ошибка при опросе: %s", e)
            await asyncio.sleep(self._poll_interval)

        logger.info("UDS Poller: остановлен.")

    async def _initial_sync(self) -> None:
        """Catch-up: синхронизировать все транзакции с ``_SYNC_SINCE``."""
        logger.info(
            "UDS Poller: начальная синхронизация (с %s)...",
            self._SYNC_SINCE.strftime("%Y-%m-%d"),
        )
        txs = await self._uds.get_transactions_since(self._SYNC_SINCE)
        new_count = 0

        for tx in txs:
            if not self._dedup.is_new(tx):
                continue
            try:
                await sync_uds_transaction(
                    tx,
                    self._integram,
                    notify_chat_id=self._notify_chat_id,
                    bot=self._bot,
                )
                self._dedup.mark_seen(tx["id"])
                new_count += 1
            except Exception as e:
                logger.error(
                    "UDS Poller: catch-up ошибка транзакции %s: %s",
                    tx.get("id"), e,
                )

        logger.info(
            "UDS Poller: catch-up завершён — загружено %d транзакций, синхронизировано %d новых.",
            len(txs), new_count,
        )

    async def _poll_once(self) -> None:
        """Один цикл опроса: получить последние транзакции и обработать новые."""
        transactions = await self._uds.get_transactions(limit=50)
        new_count = 0

        for tx in transactions:
            if not self._dedup.is_new(tx, since=self._SYNC_SINCE):
                continue

            try:
                await sync_uds_transaction(
                    tx,
                    self._integram,
                    notify_chat_id=self._notify_chat_id,
                    bot=self._bot,
                )
                self._dedup.mark_seen(tx["id"])
                new_count += 1
            except Exception as e:
                logger.error(
                    "UDS Poller: не удалось обработать транзакцию %s: %s",
                    tx.get("id"),
                    e,
                )

        if new_count:
            logger.info("UDS Poller: обработано %d новых транзакций.", new_count)


# ---------------------------------------------------------------------------
# Catalog sync: UDS ↔ Integram
# ---------------------------------------------------------------------------


async def sync_uds_catalog(
    uds_client: UDSClient,
    integram_client: Any,
) -> dict[str, int]:
    """Синхронизировать каталог UDS с Integram CRM.

    Для каждого товара UDS проверяет, есть ли соответствующий товар в Integram
    (по полю «Артикул UDS»). Возвращает маппинг externalId → Integram product ID.

    Args:
        uds_client:      Экземпляр UDSClient.
        integram_client: Экземпляр IntegramClient.

    Returns:
        Словарь {uds_external_id: integram_product_id}.
    """
    uds_goods = await uds_client.get_catalog()
    integram_products = await integram_client.get_products(in_stock_only=False)

    # Индекс Integram-товаров по артикулу UDS
    sku_to_integram: dict[str, int] = {
        p.sku_uds: p.id
        for p in integram_products
        if p.sku_uds
    }

    mapping: dict[str, int] = {}
    synced = 0
    missing = 0

    for good in uds_goods:
        ext_id = str(good.get("externalId") or good.get("id") or "")
        if not ext_id:
            continue

        if ext_id in sku_to_integram:
            mapping[ext_id] = sku_to_integram[ext_id]
            synced += 1
        else:
            logger.debug(
                "UDS: товар '%s' (externalId=%s) не найден в Integram.",
                good.get("name"),
                ext_id,
            )
            missing += 1

    logger.info(
        "UDS каталог: синхронизировано %d товаров, не найдено %d.",
        synced,
        missing,
    )
    return mapping
