"""Патч: заполнить пустые статусы (21391) в заказах Integram v2.

Ситуация: первый прогон sync_uds_shop_orders_v2.py создал заказы с пустым
полем статуса (21391) — chip-статус не был установлен. Скрипт находит такие
заказы, определяет актуальный статус через UDS Admin API и проставляет
правильный chip ID.

Запуск:
    cd /home/hive/BEEBOT
    python scripts/patch_v2_statuses.py [--dry-run] [--limit 100]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------

UDS_ADMIN_BASE = "https://api.uds.app/admin"
UDS_COMPANY_ID = getattr(config, "UDS_COMPANY_ID", "") or "549756192009"
UDS_ADMIN_TOKEN = (
    getattr(config, "UDS_ADMIN_TOKEN", "")
    or "MTM3NDM4OTk1MDM1MjowZDBjZDFhNi0wM2RkLTQ5NDUtOTQ3NS00MDFkYzEyMTc4Y2M6"
)

V2_URL = config.INTEGRAM_V2_URL
V2_WS = config.INTEGRAM_V2_WORKSPACE
V2_EMAIL = config.INTEGRAM_V2_EMAIL
V2_PASSWORD = config.INTEGRAM_V2_PASSWORD

TABLE_ORDERS = 2165
TABLE_ORDER_ITEMS = 2166

REQ_ORDER_NUMBER = "2195"
REQ_ORDER_STATUS = "21391"   # chip → table 21383
REQ_ITEM_ORDER   = "2207"
REQ_ITEM_UDS_ID  = "39181"
REQ_ITEM_PRODUCT = "2208"

ORDER_NUMBER_PREFIX = "UDS-SHOP-"

STATUS_MAP = {
    "COMPLETED":       "21406",  # Доставлен
    "CANCELLED":       "21408",  # Отменён
    "ACCEPTED":        "21387",  # Подтверждён
    "READY":           "21389",  # В сборке (готов к выдаче)
    "NEW":             "21385",  # Новый
    "NEED_ACK":        "21385",
    "WAITING_PAYMENT": "21385",
}

SLEEP_BETWEEN = 0.3   # пауза между запросами к UDS (чтобы не словить throttle)


# ---------------------------------------------------------------------------
# Integram v2 клиент
# ---------------------------------------------------------------------------

class V2Client:
    def __init__(self, http: httpx.AsyncClient, token: str):
        self._http = http
        self._base = f"{V2_URL}/api/v2/{V2_WS}"
        self._set_token(token)

    def _set_token(self, token: str) -> None:
        self._hdrs = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    async def _reauth(self) -> None:
        r = await self._http.post(
            f"{V2_URL}/api/v2/iam/login",
            json={"workspace": V2_WS, "email": V2_EMAIL, "password": V2_PASSWORD},
        )
        self._set_token(r.json()["accessToken"])
        logger.info("Integram v2: токен обновлён")

    async def _with_reauth(self, coro_factory):
        """Выполнить запрос, при AUTH_REQUIRED — перелогиниться и повторить."""
        result = await coro_factory()
        d = result.json() if hasattr(result, "json") else result
        if isinstance(d, dict) and not d.get("ok"):
            err = d.get("error", {})
            if err.get("code") == "AUTH_REQUIRED":
                await self._reauth()
                result = await coro_factory()
                d = result.json() if hasattr(result, "json") else result
        return d

    async def list_objects(self, type_id: int) -> list[dict]:
        """Загрузить все объекты через page-based пагинацию (v2 API: page/pageSize)."""
        all_items: list[dict] = []
        page = 1
        page_size = 50
        while True:
            r = await self._http.get(
                f"{self._base}/objects",
                params={"typeId": type_id, "page": page, "pageSize": page_size},
                headers=self._hdrs,
            )
            d = r.json()
            batch = d.get("data", [])
            all_items.extend(batch)
            meta = d.get("meta", {})
            total_pages = meta.get("totalPages", 1)
            if page >= total_pages:
                break
            page += 1
        return all_items

    async def get_object(self, obj_id: int) -> dict:
        """Получить один объект со всеми реквизитами."""
        d = await self._with_reauth(
            lambda: self._http.get(f"{self._base}/objects/{obj_id}", headers=self._hdrs)
        )
        return d.get("data") or {}

    async def update_object(self, obj_id: int, requisites: dict) -> bool:
        """Обновить реквизиты объекта через PUT /objects/{id}."""
        payload = {"requisites": requisites}

        async def do_put():
            return await self._http.patch(
                f"{self._base}/objects/{obj_id}",
                json=payload,
                headers=self._hdrs,
            )

        d = await self._with_reauth(do_put)
        if not d.get("ok"):
            logger.warning("update_object(%d) failed: %s", obj_id, d)
            return False
        return True

    async def list_items_for_order(self, order_id: int) -> list[dict]:
        """Вернуть все позиции заказа (через список + фильтрация по REQ_ITEM_ORDER)."""
        # Используем фильтр по реквизиту через search
        r = await self._http.get(
            f"{self._base}/objects",
            params={"typeId": TABLE_ORDER_ITEMS, "filter": f"{REQ_ITEM_ORDER}:{order_id}", "limit": 100},
            headers=self._hdrs,
        )
        d = r.json()
        return d.get("data", [])


# ---------------------------------------------------------------------------
# UDS клиент (только детали заказа)
# ---------------------------------------------------------------------------

class UDSClient:
    def __init__(self, token: str):
        self._http = httpx.AsyncClient(
            base_url=UDS_ADMIN_BASE,
            headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
            timeout=30.0,
        )

    async def close(self):
        await self._http.aclose()

    async def get_order_detail(self, order_id: int) -> dict | None:
        try:
            r = await self._http.get(f"/companies/{UDS_COMPANY_ID}/goods-orders/{order_id}")
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.warning("UDS get_order_detail(%d) error: %s", order_id, e)
            return None


# ---------------------------------------------------------------------------
# Патч статусов
# ---------------------------------------------------------------------------

async def patch_statuses(v2: V2Client, uds: UDSClient, dry_run: bool, limit: int | None) -> None:
    logger.info("Загружаю все заказы из v2...")
    all_orders = await v2.list_objects(TABLE_ORDERS)
    logger.info("Всего заказов в v2: %d", len(all_orders))

    # Найти UDS-SHOP заказы без статуса (21391 пустой или отсутствует)
    need_patch: list[tuple[int, str, int]] = []  # (v2_id, order_number, uds_id)
    for obj in all_orders:
        num = obj.get("value", "")
        if not num.startswith(ORDER_NUMBER_PREFIX):
            continue
        reqs = obj.get("requisites") or {}
        status_val = reqs.get(REQ_ORDER_STATUS)
        if status_val:
            continue   # статус уже есть

        try:
            uds_id = int(num[len(ORDER_NUMBER_PREFIX):])
        except ValueError:
            continue
        need_patch.append((obj["id"], num, uds_id))

    logger.info("Заказов без chip-статуса: %d", len(need_patch))

    if limit:
        need_patch = need_patch[:limit]
        logger.info("Обрабатываю первые %d", limit)

    patched = 0
    skipped = 0
    errors = 0

    for v2_id, num, uds_id in need_patch:
        await asyncio.sleep(SLEEP_BETWEEN)

        detail = await uds.get_order_detail(uds_id)
        if not detail:
            logger.warning("  %s — UDS не вернул детали, пропускаю", num)
            skipped += 1
            continue

        state = detail.get("state") or "NEW"
        chip_id = STATUS_MAP.get(state, "21385")

        if dry_run:
            logger.info("[DRY-RUN] %s → state=%s chip=%s", num, state, chip_id)
            patched += 1
            continue

        try:
            ok = await v2.update_object(v2_id, {REQ_ORDER_STATUS: chip_id})
        except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.TimeoutException) as e:
            logger.warning("  %s — таймаут, повтор через 5с...", num)
            await asyncio.sleep(5)
            try:
                ok = await v2.update_object(v2_id, {REQ_ORDER_STATUS: chip_id})
            except Exception:
                ok = False

        if ok:
            logger.info("✅ %s | state=%-20s → chip %s", num, state, chip_id)
            patched += 1
        else:
            logger.error("❌ %s | update failed", num)
            errors += 1

    logger.info("=== Статусы: пропатчено %d, пропущено %d, ошибок %d ===",
                patched, skipped, errors)


# ---------------------------------------------------------------------------
# Патч UDS ID в позициях заказа
# ---------------------------------------------------------------------------

async def patch_item_uds_ids(v2: V2Client, uds: UDSClient, dry_run: bool, limit: int | None) -> None:
    """Заполнить поле REQ_ITEM_UDS_ID (39181) в позициях заказа, если оно пустое.

    Алгоритм:
    1. Список всех позиций (table 2166)
    2. Для позиций без 39181 — найти родительский заказ (поле 2207)
    3. Из номера заказа получить UDS order ID, сходить в UDS за деталями
    4. Сопоставить позицию по имени → взять goods.rows[i].id
    5. Обновить поле 39181
    """
    logger.info("Загружаю все позиции заказов из v2...")
    all_items = await v2.list_objects(TABLE_ORDER_ITEMS)
    logger.info("Всего позиций: %d", len(all_items))

    # Позиции без UDS ID
    need_patch = [it for it in all_items
                  if not (it.get("requisites") or {}).get(REQ_ITEM_UDS_ID)]
    logger.info("Позиций без UDS ID: %d", len(need_patch))

    if not need_patch:
        return

    if limit:
        need_patch = need_patch[:limit]

    # Кэш: v2_order_id → (uds_order_id, goods_name_map)
    # goods_name_map: {name_lower: uds_product_id}
    order_cache: dict[int, dict[str, int]] = {}

    # Нам нужен маппинг v2 order_id → номер заказа
    logger.info("Загружаю заказы для маппинга...")
    all_orders = await v2.list_objects(TABLE_ORDERS)
    v2_order_id_to_num: dict[int, str] = {}
    for obj in all_orders:
        num = obj.get("value", "")
        if num.startswith(ORDER_NUMBER_PREFIX):
            v2_order_id_to_num[obj["id"]] = num

    patched = 0
    skipped = 0
    errors = 0

    for item in need_patch:
        await asyncio.sleep(SLEEP_BETWEEN)
        item_id = item["id"]
        item_name = item.get("value", "").lower().strip()
        reqs = item.get("requisites") or {}

        # Получить v2 order_id из позиции
        order_ref = reqs.get(REQ_ITEM_ORDER)
        if not order_ref:
            skipped += 1
            continue
        try:
            v2_order_id = int(order_ref)
        except (ValueError, TypeError):
            skipped += 1
            continue

        order_num = v2_order_id_to_num.get(v2_order_id)
        if not order_num:
            skipped += 1
            continue

        # Получить UDS order ID
        try:
            uds_order_id = int(order_num[len(ORDER_NUMBER_PREFIX):])
        except ValueError:
            skipped += 1
            continue

        # Загрузить UDS детали (с кэшированием)
        if v2_order_id not in order_cache:
            detail = await uds.get_order_detail(uds_order_id)
            if not detail:
                order_cache[v2_order_id] = {}
            else:
                goods_rows = (detail.get("goods") or {}).get("rows") or []
                order_cache[v2_order_id] = {
                    (g.get("name") or "").lower().strip(): g.get("id")
                    for g in goods_rows
                    if g.get("type") != "OPTION" and g.get("id")
                }

        goods_map = order_cache[v2_order_id]
        if not goods_map:
            skipped += 1
            continue

        uds_product_id = goods_map.get(item_name)
        if not uds_product_id:
            # Попробовать подстроку
            for k, v in goods_map.items():
                if item_name in k or k in item_name:
                    uds_product_id = v
                    break

        if not uds_product_id:
            logger.debug("  Не найден UDS ID для '%s' в заказе %s", item_name, order_num)
            skipped += 1
            continue

        if dry_run:
            logger.info("[DRY-RUN] Позиция %d '%s' → uds_product_id=%s",
                        item_id, item_name, uds_product_id)
            patched += 1
            continue

        ok = await v2.update_object(item_id, {REQ_ITEM_UDS_ID: str(uds_product_id)})
        if ok:
            logger.info("✅ Позиция %d '%s' → uds_product_id=%s", item_id, item_name, uds_product_id)
            patched += 1
        else:
            logger.error("❌ Позиция %d update failed", item_id)
            errors += 1

    logger.info("=== UDS ID позиций: пропатчено %d, пропущено %d, ошибок %d ===",
                patched, skipped, errors)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(args: argparse.Namespace) -> None:
    token = args.token or UDS_ADMIN_TOKEN
    if not token:
        logger.error("UDS Bearer-токен не указан")
        sys.exit(1)

    uds = UDSClient(token=token)

    async with httpx.AsyncClient(timeout=120) as http:
        r = await http.post(
            f"{V2_URL}/api/v2/iam/login",
            json={"email": V2_EMAIL, "password": V2_PASSWORD},
        )
        rj = r.json()
        if "accessToken" not in rj:
            logger.error("Integram v2 auth failed: %s", rj)
            sys.exit(1)
        v2 = V2Client(http, rj["accessToken"])
        logger.info("Integram v2 авторизация OK")

        if not args.items_only:
            await patch_statuses(v2, uds, dry_run=args.dry_run, limit=args.limit)

        if not args.statuses_only:
            await patch_item_uds_ids(v2, uds, dry_run=args.dry_run, limit=args.limit)

    await uds.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Патч статусов и UDS ID в Integram v2")
    parser.add_argument("--dry-run", action="store_true", help="Только вывод, без записи")
    parser.add_argument("--limit", type=int, default=None, help="Максимум объектов для обработки")
    parser.add_argument("--token", default="", help="UDS Admin Bearer-токен")
    parser.add_argument("--statuses-only", action="store_true", help="Только патч статусов")
    parser.add_argument("--items-only", action="store_true", help="Только патч UDS ID в позициях")
    args = parser.parse_args()
    asyncio.run(main(args))
