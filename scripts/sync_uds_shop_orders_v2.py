"""Синхронизация UDS-магазина (goods-orders) → Integram CRM v2 (ai2o.online).

Запуск:
    cd /home/hive/BEEBOT
    python scripts/sync_uds_shop_orders_v2.py [--dry-run] [--since 2024-01-01] [--limit 100]

Параметры:
    --dry-run    Не писать в CRM, только показать что будет сделано
    --since      Заказы начиная с даты (YYYY-MM-DD), по умолчанию 2024-01-01
    --limit      Максимум заказов для обработки
    --token      Bearer-токен UDS admin
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import config
from src.phone_utils import normalize_phone

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# UDS Admin API
# ---------------------------------------------------------------------------

UDS_ADMIN_BASE = "https://api.uds.app/admin"
UDS_COMPANY_ID = getattr(config, "UDS_COMPANY_ID", "") or "549756192009"
UDS_ADMIN_TOKEN = getattr(config, "UDS_ADMIN_TOKEN", "") or \
    "MTM3NDM4OTk1MDM1MjowZDBjZDFhNi0wM2RkLTQ5NDUtOTQ3NS00MDFkYzEyMTc4Y2M6"

ORDER_NUMBER_PREFIX = "UDS-SHOP-"
SLEEP_DETAIL = 0.3
SLEEP_PAGE = 1.0
PAGE_SIZE = 50

# ---------------------------------------------------------------------------
# Integram v2 константы
# ---------------------------------------------------------------------------

V2_URL = config.INTEGRAM_V2_URL
V2_WS  = config.INTEGRAM_V2_WORKSPACE
V2_EMAIL    = config.INTEGRAM_V2_EMAIL
V2_PASSWORD = config.INTEGRAM_V2_PASSWORD

TABLE_PRODUCTS   = 2163
TABLE_CLIENTS    = 2164
TABLE_ORDERS     = 2165
TABLE_ORDER_ITEMS = 2166

# Реквизиты товаров
REQ_PRODUCT_PRICE   = "2180"
REQ_PRODUCT_STOCK   = "2186"
REQ_PRODUCT_INSTOCK = "2183"

# Реквизиты клиентов
REQ_CLIENT_PHONE  = "2188"
REQ_CLIENT_SOURCE = "2194"

# Реквизиты заказов
REQ_ORDER_NUMBER        = "2195"
REQ_ORDER_DATE          = "2196"
REQ_ORDER_CLIENT        = "2197"
REQ_ORDER_SOURCE        = "2198"
REQ_ORDER_STATUS        = "21391"  # chip → TABLE_STATUS_CHIPS (21383)
REQ_ORDER_DELIVERY      = "2200"
REQ_ORDER_ADDRESS       = "2201"
REQ_ORDER_TRACKING      = "2202"
REQ_ORDER_ITEMS_TOTAL   = "2203"
REQ_ORDER_DELIVERY_COST = "2204"
REQ_ORDER_TOTAL         = "2205"
REQ_ORDER_COMMENT       = "2206"

# Реквизиты позиций
REQ_ITEM_ORDER   = "2207"
REQ_ITEM_PRODUCT = "2208"
REQ_ITEM_QTY     = "2209"
REQ_ITEM_PRICE   = "2210"
REQ_ITEM_SUM     = "2211"
REQ_ITEM_UDS_ID  = "39181"   # UDS internal product ID

# Lookup IDs (записи в справочниках — shared между v1/v2)
SOURCE_UDS      = "21"
STATUS_MAP = {
    "COMPLETED":       "21406",  # Доставлен
    "CANCELLED":       "21408",  # Отменён
    "ACCEPTED":        "21387",  # Подтверждён
    "NEW":             "21385",  # Новый
    "NEED_ACK":        "21385",
    "WAITING_PAYMENT": "21385",
}
DELIVERY_MAP = {
    "СДЭК":          "191",
    "Почта России":  "193",
    "Самовывоз":     "195",
}


# ---------------------------------------------------------------------------
# Integram v2 клиент (прямой REST)
# ---------------------------------------------------------------------------

class V2Client:
    def __init__(self, http: httpx.AsyncClient, token: str):
        self._http = http
        self._base = f"{V2_URL}/api/v2/{V2_WS}"
        self._set_token(token)

    def _set_token(self, token: str) -> None:
        self._hdrs = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    async def _reauth(self) -> None:
        """Перелогиниться при истечении JWT (токен живёт ~15 мин)."""
        r = await self._http.post(f"{V2_URL}/api/v2/iam/login",
            json={"workspace": V2_WS, "email": V2_EMAIL, "password": V2_PASSWORD})
        self._set_token(r.json()["accessToken"])
        logger.info("Integram v2: токен обновлён")

    async def list_objects(self, type_id: int) -> list[dict]:
        """Загрузить все объекты через page-based пагинацию (v2 API: page/pageSize)."""
        all_items: list[dict] = []
        page = 1
        page_size = 50
        while True:
            r = await self._http.get(f"{self._base}/objects",
                params={"typeId": type_id, "page": page, "pageSize": page_size},
                headers=self._hdrs)
            d = r.json()
            batch = d.get("data", [])
            all_items.extend(batch)
            meta = d.get("meta", {})
            total_pages = meta.get("totalPages", 1)
            if page >= total_pages:
                break
            page += 1
        return all_items

    async def create_object(self, type_id: int, value: str, requisites: dict) -> int:
        r = await self._http.post(f"{self._base}/objects",
            json={"typeId": type_id, "value": value, "requisites": requisites},
            headers=self._hdrs)
        d = r.json()
        if not d.get("ok"):
            err = d.get("error", {})
            if err.get("code") == "AUTH_REQUIRED":
                await self._reauth()
                r = await self._http.post(f"{self._base}/objects",
                    json={"typeId": type_id, "value": value, "requisites": requisites},
                    headers=self._hdrs)
                d = r.json()
        if not d.get("ok"):
            raise RuntimeError(f"create_object failed: {d}")
        return d["data"]["id"]


# ---------------------------------------------------------------------------
# UDS Admin клиент
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

    async def get_orders_page(self, offset: int, limit: int = PAGE_SIZE) -> dict:
        r = await self._http.get(f"/companies/{UDS_COMPANY_ID}/goods-orders",
            params={"max": limit, "offset": offset})
        r.raise_for_status()
        return r.json()

    async def get_order_detail(self, order_id: int) -> dict:
        r = await self._http.get(f"/companies/{UDS_COMPANY_ID}/goods-orders/{order_id}")
        r.raise_for_status()
        return r.json()


# ---------------------------------------------------------------------------
# Парсинг UDS-заказа
# ---------------------------------------------------------------------------

def parse_order(detail: dict) -> dict:
    customer = detail.get("customer") or {}
    delivery = detail.get("deliveryData") or {}
    purchase = detail.get("purchase") or {}
    goods_rows = (detail.get("goods") or {}).get("rows") or []

    delivery_case = delivery.get("deliveryCase") or {}
    delivery_name = delivery_case.get("name") or ""
    delivery_cost = float(delivery_case.get("value") or 0)
    extras = purchase.get("extras") or {}
    if not delivery_cost:
        delivery_cost = float(extras.get("delivery") or 0)

    items = [
        {
            "name": g.get("name") or "",
            "qty": int(g.get("qty") or 1),
            "price": float(g.get("price") or 0),
            "uds_product_id": g.get("id"),   # UDS internal product ID для точной привязки
        }
        for g in goods_rows
        if g.get("type") != "OPTION"
    ]

    total = float(purchase.get("total") or 0) or float(detail.get("price") or 0)
    return {
        "uds_order_id": detail.get("id"),
        "date_created": detail.get("dateCreated") or "",
        "state": detail.get("state") or "",
        "payment_status": detail.get("paymentStatus") or "",
        "total": total,
        "items_total": total - delivery_cost,
        "delivery_cost": delivery_cost,
        "delivery_name": delivery_name,
        "customer_name": delivery.get("receiverName") or customer.get("displayName") or "",
        "customer_phone": normalize_phone(
            delivery.get("receiverPhone") or customer.get("phone") or ""
        ) or (delivery.get("receiverPhone") or customer.get("phone") or ""),
        "address": delivery.get("address") or "",
        "comment": delivery.get("userComment") or "",
        "items": items,
    }


# ---------------------------------------------------------------------------
# Основная логика
# ---------------------------------------------------------------------------

async def get_or_create_client(v2: V2Client, parsed: dict, client_cache: dict) -> int:
    phone = parsed["customer_phone"]
    name = parsed["customer_name"] or "UDS-покупатель"
    key = "".join(c for c in phone if c.isdigit())

    if key and key in client_cache:
        return client_cache[key]

    reqs = {REQ_CLIENT_SOURCE: SOURCE_UDS}
    if phone:
        reqs[REQ_CLIENT_PHONE] = phone

    obj_id = await v2.create_object(TABLE_CLIENTS, name, reqs)
    logger.info("Создан клиент '%s' id=%d", name, obj_id)
    if key:
        client_cache[key] = obj_id
    return obj_id


def resolve_products(items: list[dict], name_map: dict, uds_id_map: dict) -> list[dict]:
    """Привязывает позиции к каталогу v2.

    uds_id_map: {uds_product_id: v2_product_id} — точный маппинг по UDS ID.
    name_map: {name_lower: v2_product_id} — фолбэк по имени.
    """
    result = []
    for item in items:
        uds_pid = item.get("uds_product_id")
        # 1. Точный маппинг по UDS product ID
        prod_id = uds_id_map.get(uds_pid, 0) if uds_pid else 0
        # 2. Фолбэк: по имени (точное, затем подстрока)
        if not prod_id:
            name_lower = item["name"].lower().strip()
            prod_id = name_map.get(name_lower, 0)
            if not prod_id:
                for k, v in name_map.items():
                    if name_lower in k or k in name_lower:
                        prod_id = v
                        break
        result.append({**item, "product_id": prod_id})
    return result


async def sync_order(v2: V2Client, parsed: dict, client_cache: dict,
                     name_map: dict, uds_id_map: dict, existing_numbers: set, dry_run: bool) -> bool:
    order_number = f"{ORDER_NUMBER_PREFIX}{parsed['uds_order_id']}"
    if order_number in existing_numbers:
        return False

    if not dry_run and parsed["payment_status"] not in ("PAID", "COMPLETED"):
        return False

    order_date = datetime.now(tz=timezone.utc)
    if parsed["date_created"]:
        try:
            order_date = datetime.fromisoformat(
                parsed["date_created"].replace("Z", "+00:00"))
        except ValueError:
            pass

    status_id = STATUS_MAP.get(parsed["state"], "21385")  # дефолт: Новый
    delivery_id = next((v for k, v in DELIVERY_MAP.items()
                        if k.lower() in parsed["delivery_name"].lower()), "")

    items = resolve_products(parsed["items"], name_map, uds_id_map)

    if dry_run:
        items_str = ", ".join(f"{i['name']} ×{i['qty']}" for i in parsed["items"])
        logger.info("[DRY-RUN] %s | %s | %s | %.0f₽ | %s",
            order_number, parsed["customer_name"], parsed["customer_phone"],
            parsed["total"], items_str or "(нет товаров)")
        return True

    client_id = await get_or_create_client(v2, parsed, client_cache)

    order_reqs: dict = {
        REQ_ORDER_NUMBER: order_number,
        REQ_ORDER_DATE: order_date.strftime("%Y-%m-%d"),
        REQ_ORDER_CLIENT: str(client_id),
        REQ_ORDER_SOURCE: SOURCE_UDS,
        REQ_ORDER_STATUS: status_id,
        REQ_ORDER_TOTAL: str(int(parsed["total"])),
    }
    if parsed["items_total"]:
        order_reqs[REQ_ORDER_ITEMS_TOTAL] = str(int(parsed["items_total"]))
    if parsed["delivery_cost"]:
        order_reqs[REQ_ORDER_DELIVERY_COST] = str(int(parsed["delivery_cost"]))
    if parsed["address"]:
        order_reqs[REQ_ORDER_ADDRESS] = parsed["address"]
    if delivery_id:
        order_reqs[REQ_ORDER_DELIVERY] = delivery_id
    if parsed["comment"]:
        order_reqs[REQ_ORDER_COMMENT] = parsed["comment"]

    order_id = await v2.create_object(TABLE_ORDERS, order_number, order_reqs)

    for item in items:
        item_reqs = {
            REQ_ITEM_ORDER: str(order_id),
            REQ_ITEM_QTY: str(item["qty"]),
            REQ_ITEM_PRICE: str(int(item["price"])),
            REQ_ITEM_SUM: str(int(item["qty"] * item["price"])),
        }
        if item["product_id"]:
            item_reqs[REQ_ITEM_PRODUCT] = str(item["product_id"])
        if item.get("uds_product_id"):
            item_reqs[REQ_ITEM_UDS_ID] = str(item["uds_product_id"])
        await v2.create_object(TABLE_ORDER_ITEMS, item["name"], item_reqs)

    existing_numbers.add(order_number)
    logger.info("✅ %s | %s | %.0f₽ | %d позиций",
        order_number, parsed["customer_name"], parsed["total"], len(items))
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(args: argparse.Namespace) -> None:
    since_dt = None
    if args.since:
        since_dt = datetime.fromisoformat(args.since).replace(tzinfo=timezone.utc)
        logger.info("Фильтр: заказы с %s", args.since)

    token = args.token or UDS_ADMIN_TOKEN
    if not token:
        logger.error("Bearer-токен не указан")
        sys.exit(1)

    uds = UDSClient(token=token)

    async with httpx.AsyncClient(timeout=30) as http:
        # Авторизация v2
        r = await http.post(f"{V2_URL}/api/v2/iam/login",
            json={"email": V2_EMAIL, "password": V2_PASSWORD})
        v2_token = r.json()["accessToken"]
        v2 = V2Client(http, v2_token)
        logger.info("Integram v2 авторизация OK")

        # Загрузить уже существующие заказы
        logger.info("Загрузка существующих заказов из v2...")
        existing_objs = await v2.list_objects(TABLE_ORDERS)
        existing_numbers = set()
        for obj in existing_objs:
            reqs = obj.get("requisites", {})
            num = reqs.get(REQ_ORDER_NUMBER) or obj.get("value", "")
            if num.startswith(ORDER_NUMBER_PREFIX):
                existing_numbers.add(num)
        logger.info("Уже в v2: %d UDS-SHOP заказов", len(existing_numbers))

        # Загрузить каталог товаров v2 (name → id)
        logger.info("Загрузка каталога товаров v2...")
        prod_objs = await v2.list_objects(TABLE_PRODUCTS)
        name_map = {obj["value"].lower().strip(): obj["id"]
                    for obj in prod_objs if obj.get("value")}
        # Точный маппинг UDS product ID → v2 product ID (из поля REQ_ITEM_UDS_ID в каталоге)
        uds_id_map: dict[int, int] = {}
        for obj in prod_objs:
            sku = (obj.get("requisites") or {}).get(REQ_ITEM_UDS_ID)
            if sku:
                try:
                    uds_id_map[int(sku)] = obj["id"]
                except (ValueError, TypeError):
                    pass
        logger.info("Товаров в v2: %d (UDS ID маппинг: %d)", len(name_map), len(uds_id_map))

        # Загрузить клиентов в кэш
        logger.info("Загрузка клиентов v2...")
        client_objs = await v2.list_objects(TABLE_CLIENTS)
        client_cache: dict[str, int] = {}
        for obj in client_objs:
            phone = (obj.get("requisites") or {}).get(REQ_CLIENT_PHONE, "")
            key = "".join(c for c in phone if c.isdigit())
            if key:
                client_cache[key] = obj["id"]
        logger.info("Клиентов в кэше: %d", len(client_cache))

        # Пагинация по UDS
        offset = 0
        total_uds = None
        processed = created = skipped = errors = 0

        try:
            while True:
                page = await uds.get_orders_page(offset=offset)
                if total_uds is None:
                    total_uds = page.get("total", 0)
                    logger.info("Всего заказов в UDS: %d", total_uds)

                rows = page.get("rows", [])
                if not rows:
                    break

                for row in rows:
                    order_id = row["id"]
                    date_str = (row.get("dateCreated") or "")[:10]

                    if since_dt and date_str:
                        try:
                            row_dt = datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)
                            if row_dt < since_dt:
                                skipped += 1
                                continue
                        except ValueError:
                            pass

                    order_number = f"{ORDER_NUMBER_PREFIX}{order_id}"
                    if order_number in existing_numbers:
                        skipped += 1
                        continue

                    try:
                        detail = await uds.get_order_detail(order_id)
                        await asyncio.sleep(SLEEP_DETAIL)
                        parsed = parse_order(detail)

                        ok = await sync_order(v2, parsed, client_cache,
                                              name_map, uds_id_map, existing_numbers, args.dry_run)
                        if ok:
                            created += 1
                        else:
                            skipped += 1
                        processed += 1

                    except Exception as e:
                        logger.error("Ошибка заказа %s: %s", order_id, e)
                        errors += 1

                    if args.limit and processed >= args.limit:
                        logger.info("Достигнут лимит --limit=%d", args.limit)
                        break

                if args.limit and processed >= args.limit:
                    break

                offset += len(rows)
                if offset >= (total_uds or 0):
                    break

                logger.info("Прогресс: %d/%d (создано: %d, пропущено: %d, ошибок: %d)",
                    offset, total_uds, created, skipped, errors)
                await asyncio.sleep(SLEEP_PAGE)

        finally:
            await uds.close()

    logger.info("=== Готово: создано %d, пропущено %d, ошибок %d (из %d в UDS) ===",
        created, skipped, errors, total_uds or "?")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Синхронизация UDS → Integram v2")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--since", default="2024-01-01")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--token", default=None)
    asyncio.run(main(parser.parse_args()))
