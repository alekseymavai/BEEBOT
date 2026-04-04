"""Одноразовая синхронизация UDS-магазина (goods-orders) → Integram CRM.

UDS goods-orders — это заказы из UDS-магазина с оплатой через YooKassa.
Они содержат полный состав товаров, адрес доставки и телефон клиента.
Отличаются от UDS operations (loyalty card), которые синхронизирует UDSPoller.

Запуск:
    cd /home/hive/BEEBOT
    python scripts/sync_uds_shop_orders.py [--dry-run] [--since 2024-01-01] [--limit 100]

Параметры:
    --dry-run    Не писать в CRM, только показать что будет сделано
    --since      Синхронизировать заказы начиная с этой даты (YYYY-MM-DD)
    --limit      Максимальное количество заказов для обработки
    --token      Bearer-токен UDS admin (если не указан — читает из .env или запрашивает)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx

# Добавить корень проекта в path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import config
from src.integram_client import IntegramClient
from src.phone_utils import normalize_phone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# UDS Admin API
# ---------------------------------------------------------------------------

UDS_ADMIN_BASE = "https://api.uds.app/admin"
UDS_COMPANY_ID = getattr(config, "UDS_COMPANY_ID", "") or "549756192009"

# Bearer-токен из localStorage браузера (game.auth_token).
# Действителен ~120 дней. Обновить при истечении.
UDS_ADMIN_TOKEN = getattr(config, "UDS_ADMIN_TOKEN", "") or \
    "MTM3NDM4OTk1MDM1MjowZDBjZDFhNi0wM2RkLTQ5NDUtOTQ3NS00MDFkYzEyMTc4Y2M6"

ORDER_NUMBER_PREFIX = "UDS-SHOP-"
SLEEP_BETWEEN_DETAIL_REQUESTS = 0.3   # сек между запросами деталей заказа
SLEEP_BETWEEN_PAGE_REQUESTS = 1.0     # сек между страницами списка
PAGE_SIZE = 50


class UDSAdminClient:
    """HTTP-клиент для UDS Admin API (Bearer auth)."""

    def __init__(self, token: str, company_id: str) -> None:
        self._token = token
        self._company_id = company_id
        self._http = httpx.AsyncClient(
            base_url=UDS_ADMIN_BASE,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/json",
            },
            timeout=30.0,
        )

    async def close(self) -> None:
        await self._http.aclose()

    async def get_orders_page(self, offset: int, limit: int = PAGE_SIZE) -> dict:
        r = await self._http.get(
            f"/companies/{self._company_id}/goods-orders",
            params={"max": limit, "offset": offset},
        )
        r.raise_for_status()
        return r.json()

    async def get_order_detail(self, order_id: int) -> dict:
        r = await self._http.get(
            f"/companies/{self._company_id}/goods-orders/{order_id}",
        )
        r.raise_for_status()
        return r.json()


# ---------------------------------------------------------------------------
# Sync logic
# ---------------------------------------------------------------------------


def _parse_order_detail(detail: dict) -> dict:
    """Извлечь нужные поля из детального ответа UDS."""
    customer = detail.get("customer") or {}
    delivery = detail.get("deliveryData") or {}
    purchase = detail.get("purchase") or {}
    goods_rows = (detail.get("goods") or {}).get("rows") or []

    # Адрес и имя получателя
    address = delivery.get("address") or ""
    receiver_name = delivery.get("receiverName") or customer.get("displayName") or ""
    receiver_phone = delivery.get("receiverPhone") or customer.get("phone") or ""
    user_comment = delivery.get("userComment") or ""
    delivery_case = (delivery.get("deliveryCase") or {})
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
            "uds_id": str(g.get("id") or ""),
        }
        for g in goods_rows
        if g.get("type") != "OPTION"  # пропустить опции/добавки
    ]

    return {
        "uds_order_id": detail.get("id"),
        "date_created": detail.get("dateCreated") or "",
        "state": detail.get("state") or "",
        "payment_status": detail.get("paymentStatus") or "",
        "total": float(detail.get("purchase", {}).get("total") or 0) or float(detail.get("price") or 0),
        "items_total": float(purchase.get("total") or 0) - delivery_cost,
        "delivery_cost": delivery_cost,
        "delivery_method": delivery_name,
        "customer_name": receiver_name,
        "customer_phone": normalize_phone(receiver_phone) or receiver_phone,
        "address": address,
        "comment": user_comment,
        "items": items,
    }


async def _get_or_create_client(
    integram: IntegramClient,
    parsed: dict,
    client_cache: dict,  # phone_digits → client_id
) -> int:
    """Найти или создать клиента в Integram используя кеш. Вернуть client_id.

    Кеш строится один раз в main() из всех клиентов CRM.
    Если клиента нет в кеше — создаём напрямую через API (без повторной загрузки всех клиентов).
    """
    from src.crm_constants import TABLE_CLIENTS, REQ_CLIENT_PHONE, REQ_CLIENT_SOURCE, SOURCE_IDS

    phone = parsed["customer_phone"]
    name = parsed["customer_name"] or "UDS-покупатель"
    phone_digits = "".join(c for c in phone if c.isdigit())

    # Сначала проверяем кеш
    if phone_digits and phone_digits in client_cache:
        return client_cache[phone_digits]

    # Клиента нет в кеше — создаём напрямую (он точно новый, т.к. кеш полный)
    reqs: dict = {}
    if phone:
        reqs[REQ_CLIENT_PHONE] = phone
    reqs[REQ_CLIENT_SOURCE] = SOURCE_IDS.get("UDS", "")

    obj_id = await integram._api.create_object(TABLE_CLIENTS, name, reqs)
    logger.info("Создан новый клиент '%s' (id=%d)", name, obj_id)

    if phone_digits:
        client_cache[phone_digits] = obj_id
    return obj_id


def _resolve_items_cached(items: list[dict], name_map: dict, short_map: dict) -> list[dict]:
    """Сопоставить товары UDS с каталогом Integram по названию (без IO)."""

    crm_items = []
    for item in items:
        name_lower = item["name"].lower().strip()
        product = name_map.get(name_lower) or short_map.get(name_lower)

        if not product:
            for pname, prod in name_map.items():
                if name_lower in pname or pname in name_lower:
                    product = prod
                    break

        crm_items.append({
            "product_id": product.id if product else 0,
            "name": item["name"],
            "quantity": item["qty"],
            "unit_price": item["price"],
        })

    return crm_items


async def _resolve_items(integram: IntegramClient, items: list[dict]) -> list[dict]:
    """Устаревшая версия — оставлена для совместимости."""
    all_products = await integram.get_products(in_stock_only=False)
    name_map = {p.name.lower().strip(): p for p in all_products}
    short_map = {(p.short_name or "").lower().strip(): p for p in all_products if p.short_name}
    return _resolve_items_cached(items, name_map, short_map)


async def sync_one_order(
    integram: IntegramClient,
    parsed: dict,
    client_cache: dict,
    name_map: dict,
    short_map: dict,
    dry_run: bool = False,
) -> bool:
    """Синхронизировать один заказ. Вернуть True если создан."""
    order_number = f"{ORDER_NUMBER_PREFIX}{parsed['uds_order_id']}"

    # Дата заказа
    order_date = datetime.now(tz=timezone.utc)
    if parsed["date_created"]:
        try:
            order_date = datetime.fromisoformat(
                parsed["date_created"].replace("Z", "+00:00")
            )
        except ValueError:
            pass

    # Статус в Integram
    state_map = {
        "COMPLETED": "Доставлен",
        "CANCELLED": "Отменён",
        "ACCEPTED": "В обработке",
        "NEW": "Новый",
        "NEED_ACK": "Новый",
        "WAITING_PAYMENT": "Ожидает оплаты",
    }
    status = state_map.get(parsed["state"], "Новый")

    if dry_run:
        items_str = ", ".join(
            f"{i['name']} ×{i['qty']}" for i in parsed["items"]
        )
        logger.info(
            "[DRY-RUN] %s | %s | %s | %.0f₽ | %s",
            order_number,
            parsed["customer_name"],
            parsed["customer_phone"],
            parsed["total"],
            items_str or "(нет товаров)",
        )
        return True

    # Создать клиента (с кешем)
    client_id = await _get_or_create_client(integram, parsed, client_cache)

    # Разобрать товары (из кеша)
    crm_items = _resolve_items_cached(parsed["items"], name_map, short_map)

    # Создать заказ
    await integram.create_order(
        client_id=client_id,
        items=crm_items,
        source="UDS",
        number=order_number,
        items_total=parsed["items_total"] or parsed["total"] - parsed["delivery_cost"],
        total=parsed["total"],
        status=status,
        date=order_date,
        address=parsed["address"],
        delivery_cost=parsed["delivery_cost"] if parsed["delivery_cost"] else None,
        comment=parsed["comment"] or None,
    )
    logger.info(
        "✅ Создан %s | %s | %.0f₽ | %d позиций",
        order_number,
        parsed["customer_name"],
        parsed["total"],
        len(crm_items),
    )
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
        logger.error("Bearer-токен не указан. Укажите --token или UDS_ADMIN_TOKEN в .env")
        sys.exit(1)

    uds = UDSAdminClient(token=token, company_id=UDS_COMPANY_ID)
    integram = IntegramClient()
    await integram.authenticate()

    # Загрузить существующие UDS-SHOP-* заказы из CRM
    logger.info("Загрузка существующих UDS-SHOP заказов из CRM...")
    existing = await integram.get_orders()
    existing_numbers = {o.number for o in existing if o.number and o.number.startswith(ORDER_NUMBER_PREFIX)}
    logger.info("В CRM уже есть %d UDS-SHOP заказов", len(existing_numbers))

    # Загрузить каталог товаров один раз
    logger.info("Загрузка каталога товаров из CRM...")
    all_products = await integram.get_products(in_stock_only=False)
    name_map = {p.name.lower().strip(): p for p in all_products}
    short_map = {(p.short_name or "").lower().strip(): p for p in all_products if p.short_name}
    logger.info("Загружено %d товаров", len(all_products))

    # Загрузить всех клиентов в кеш (phone_digits → client_id)
    logger.info("Загрузка клиентов в кеш...")
    all_clients = await integram.get_clients()
    client_cache: dict[str, int] = {}
    for c in all_clients:
        if c.phone:
            digits = "".join(ch for ch in c.phone if ch.isdigit())
            if digits:
                client_cache[digits] = c.id
    logger.info("Кеш клиентов: %d записей", len(client_cache))

    # Пагинация по списку заказов UDS
    offset = 0
    total_uds = None
    processed = 0
    created = 0
    skipped = 0
    errors = 0

    try:
        while True:
            page = await uds.get_orders_page(offset=offset)
            if total_uds is None:
                total_uds = page.get("total", 0)
                logger.info("Всего заказов в UDS-магазине: %d", total_uds)

            rows = page.get("rows", [])
            if not rows:
                break

            for row in rows:
                order_id = row["id"]
                order_number = f"{ORDER_NUMBER_PREFIX}{order_id}"
                date_str = (row.get("dateCreated") or "")[:10]

                # Фильтр по дате
                if since_dt and date_str:
                    try:
                        row_date = datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)
                        if row_date < since_dt:
                            logger.debug("Пропуск %s (дата %s < since)", order_number, date_str)
                            skipped += 1
                            continue
                    except ValueError:
                        pass

                # Уже в CRM?
                if order_number in existing_numbers:
                    logger.debug("Пропуск %s (уже в CRM)", order_number)
                    skipped += 1
                    continue

                # Получить детали
                try:
                    detail = await uds.get_order_detail(order_id)
                    await asyncio.sleep(SLEEP_BETWEEN_DETAIL_REQUESTS)

                    parsed = _parse_order_detail(detail)

                    # Пропустить неоплаченные (если не dry-run)
                    if not args.dry_run and parsed["payment_status"] not in ("PAID", "COMPLETED"):
                        logger.debug("Пропуск %s (статус оплаты: %s)", order_number, parsed["payment_status"])
                        skipped += 1
                        continue

                    success = await sync_one_order(
                        integram, parsed,
                        client_cache=client_cache,
                        name_map=name_map,
                        short_map=short_map,
                        dry_run=args.dry_run,
                    )
                    if success:
                        created += 1
                    processed += 1

                except Exception as e:
                    logger.error("Ошибка при обработке заказа %s: %s", order_id, e)
                    errors += 1

                if args.limit and processed >= args.limit:
                    logger.info("Достигнут лимит --limit=%d", args.limit)
                    break

            if args.limit and processed >= args.limit:
                break

            offset += len(rows)
            if offset >= (total_uds or 0):
                break

            logger.info(
                "Прогресс: %d/%d заказов обработано (создано: %d, пропущено: %d, ошибок: %d)",
                offset, total_uds, created, skipped, errors,
            )
            await asyncio.sleep(SLEEP_BETWEEN_PAGE_REQUESTS)

    finally:
        await uds.close()
        await integram.close()

    logger.info(
        "=== Готово: создано %d, пропущено %d, ошибок %d (из %d в UDS) ===",
        created, skipped, errors, total_uds or "?",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Синхронизация UDS-магазина → Integram CRM")
    parser.add_argument("--dry-run", action="store_true", help="Не писать в CRM, только показать")
    parser.add_argument("--since", default=None, help="Дата начала синхронизации YYYY-MM-DD")
    parser.add_argument("--limit", type=int, default=None, help="Лимит заказов")
    parser.add_argument("--token", default=None, help="Bearer-токен UDS admin")
    args = parser.parse_args()
    asyncio.run(main(args))
