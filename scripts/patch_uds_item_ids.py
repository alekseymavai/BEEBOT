"""Патч: заполнить поле «UDS ID товара» в таблице «Позиции заказа v2» (2166).

Для каждой позиции с пустым «UDS ID товара»:
  1. Из поля «Заказ» извлекаем UDS order ID (UDS-SHOP-{N})
  2. Вызываем UDS Admin API → goods-orders/{N} → список товаров с id
  3. Сопоставляем позицию CRM с UDS-товаром по названию
  4. Обновляем поле через Integram v2 API

Требования:
  .env: INTEGRAM_V2_EMAIL, INTEGRAM_V2_PASSWORD (или передать через --email/--password)
        UDS_ADMIN_TOKEN (или --token)

Запуск:
    cd /home/hive/BEEBOT
    python scripts/patch_uds_item_ids.py [--dry-run] [--limit N] [--verbose]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))
from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------

INTEGRAM_URL      = getattr(config, "INTEGRAM_V2_URL", "https://ai2o.online")
INTEGRAM_EMAIL    = getattr(config, "INTEGRAM_V2_EMAIL", "")
INTEGRAM_PASSWORD = getattr(config, "INTEGRAM_V2_PASSWORD", "")
INTEGRAM_WS       = getattr(config, "INTEGRAM_V2_WORKSPACE", "alekseymavai")

TABLE_ORDER_ITEMS_V2 = 2166
UDS_ITEM_ID_FIELD    = "UDS ID товара"

UDS_ADMIN_BASE   = "https://api.uds.app/admin"
UDS_COMPANY_ID   = getattr(config, "UDS_COMPANY_ID", "") or "549756192009"
UDS_ADMIN_TOKEN  = (
    getattr(config, "UDS_ADMIN_TOKEN", "")
    or "MTM3NDM4OTk1MDM1MjowZDBjZDFhNi0wM2RkLTQ5NDUtOTQ3NS00MDFkYzEyMTc4Y2M6"
)

SLEEP_UDS         = 0.3   # сек между запросами к UDS
LIST_BATCH        = 100   # записей за один list_objects вызов
TOKEN_TTL_SEC     = 850   # обновить токен до истечения (токен живёт ~900 с)


# ---------------------------------------------------------------------------
# Integram v2 client
# ---------------------------------------------------------------------------

class IntegramV2Client:
    def __init__(self, url: str, email: str, password: str, workspace: str) -> None:
        self._url       = url.rstrip("/")
        self._email     = email
        self._password  = password
        self._ws        = workspace
        self._token: str = ""
        self._refresh:  str = ""
        self._token_ts: float = 0.0
        self._http = httpx.AsyncClient(timeout=30.0)

    async def _login(self) -> None:
        r = await self._http.post(
            f"{self._url}/api/v2/iam/login",
            json={"email": self._email, "password": self._password},
        )
        r.raise_for_status()
        d = r.json()
        self._token   = d["accessToken"]
        self._refresh = d.get("refreshToken", "")
        self._token_ts = time.monotonic()
        logger.info("Integram v2: авторизован как %s", self._email)

    async def _ensure_token(self) -> None:
        if not self._token or time.monotonic() - self._token_ts > TOKEN_TTL_SEC:
            if self._refresh:
                try:
                    r = await self._http.post(
                        f"{self._url}/api/v2/iam/refresh",
                        json={"refreshToken": self._refresh},
                    )
                    d = r.json()
                    if d.get("accessToken"):
                        self._token   = d["accessToken"]
                        self._refresh = d.get("refreshToken", self._refresh)
                        self._token_ts = time.monotonic()
                        return
                except Exception:
                    pass
            await self._login()

    async def call(self, tool: str, **args) -> dict:
        await self._ensure_token()
        r = await self._http.post(
            f"{self._url}/api/v2/{self._ws}/ai/tool",
            headers={"Authorization": f"Bearer {self._token}"},
            json={"name": tool, "args": args, "skipHitl": True},
        )
        r.raise_for_status()
        return r.json().get("data", {})

    async def close(self) -> None:
        await self._http.aclose()


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _extract_uds_order_id(заказ_field: str) -> int | None:
    """Из 'UDS-SHOP-6196033 (id:2548)' → 6196033."""
    m = re.search(r"UDS-SHOP-(\d+)", заказ_field)
    return int(m.group(1)) if m else None


def _name_similarity(a: str, b: str) -> float:
    """Доля общих слов (≥3 символов) между двумя строками."""
    wa = set(w.lower() for w in re.findall(r"\w{3,}", a))
    wb = set(w.lower() for w in re.findall(r"\w{3,}", b))
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / max(len(wa), len(wb))


def _match_uds_item(crm_name: str, uds_items: list[dict]) -> dict | None:
    """Найти UDS-товар с наилучшим совпадением названия (порог 0.4)."""
    best, best_score = None, 0.0
    for u in uds_items:
        s = _name_similarity(crm_name, u.get("name", ""))
        if s > best_score:
            best_score, best = s, u
    return best if best_score >= 0.4 else None


# ---------------------------------------------------------------------------
# Загрузка всех записей таблицы 2166
# ---------------------------------------------------------------------------

SORT_STRATEGIES = [
    "id", "-id",              # нижние и верхние 100 по ID
    "name", "-name",           # по имени — захватывает середину
    "Количество", "-Количество",
    "Цена за единицу", "-Цена за единицу",
]


async def load_all_items(client: IntegramV2Client) -> list[dict]:
    """Загрузить все записи таблицы 2166.

    list_objects не поддерживает offset, поэтому используем 8 разных сортировок
    чтобы получить все 300 записей (каждая сортировка добавляет новые).
    """
    all_records: list[dict] = []
    seen_ids: set[int] = set()
    total_expected = None

    for sort_dir in SORT_STRATEGIES:
        data = await client.call(
            "list_objects",
            typeId=TABLE_ORDER_ITEMS_V2,
            limit=LIST_BATCH,
            sort=sort_dir,
        )
        rows = data.get("rows") or []
        if total_expected is None:
            total_expected = data.get("total", 0)
        new_count = 0
        for row in rows:
            rid = row.get("id")
            if rid and rid not in seen_ids:
                seen_ids.add(rid)
                all_records.append(row)
                new_count += 1
        logger.info("sort=%-20s +%d новых, итого %d / %d", sort_dir, new_count, len(all_records), total_expected or "?")
        if total_expected and len(all_records) >= total_expected:
            break

    logger.info("Загружено уникальных записей: %d / %d", len(all_records), total_expected or "?")
    return all_records


# ---------------------------------------------------------------------------
# UDS API
# ---------------------------------------------------------------------------

async def get_uds_order_items(uds_order_id: int, http: httpx.AsyncClient) -> list[dict]:
    """Вернуть [{id, name, qty, price, type}] из UDS-заказа."""
    resp = await http.get(
        f"/companies/{UDS_COMPANY_ID}/goods-orders/{uds_order_id}",
    )
    if resp.status_code == 404:
        logger.warning("UDS заказ %d не найден (404)", uds_order_id)
        return []
    resp.raise_for_status()
    d = resp.json()
    rows = (d.get("goods") or {}).get("rows") or []
    return [
        {"id": g.get("id"), "name": g.get("name") or "", "qty": g.get("qty"), "price": g.get("price")}
        for g in rows
        if g.get("type") != "OPTION"
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(args: argparse.Namespace) -> None:
    email    = args.email    or INTEGRAM_EMAIL
    password = args.password or INTEGRAM_PASSWORD
    uds_tok  = args.token    or UDS_ADMIN_TOKEN

    if not email or not password:
        logger.error("Укажите --email и --password или INTEGRAM_V2_EMAIL/PASSWORD в .env")
        sys.exit(1)

    integram = IntegramV2Client(INTEGRAM_URL, email, password, INTEGRAM_WS)
    await integram._login()

    # 1. Загрузить все записи
    logger.info("Загрузка позиций из таблицы %d...", TABLE_ORDER_ITEMS_V2)
    all_records = await load_all_items(integram)

    # 2. Отобрать записи без UDS ID
    need_patch = [r for r in all_records if not r.get(UDS_ITEM_ID_FIELD)]
    logger.info(
        "Записей без «%s»: %d / %d",
        UDS_ITEM_ID_FIELD, len(need_patch), len(all_records),
    )

    if not need_patch:
        logger.info("Все записи уже заполнены. Выход.")
        await integram.close()
        return

    # 3. Сгруппировать по UDS-заказу
    groups: dict[int, list[dict]] = {}
    skipped = 0
    for rec in need_patch:
        uds_order_id = _extract_uds_order_id(rec.get("Заказ") or "")
        if not uds_order_id:
            logger.debug("Пропуск id=%d: нет UDS-SHOP номера", rec["id"])
            skipped += 1
            continue
        groups.setdefault(uds_order_id, []).append(rec)

    logger.info("Уникальных UDS-заказов: %d (пропущено без номера: %d)", len(groups), skipped)

    # 4. Обработать каждый заказ
    uds_http = httpx.AsyncClient(
        base_url=UDS_ADMIN_BASE,
        headers={"Authorization": f"Bearer {uds_tok}", "Accept": "application/json"},
        timeout=30.0,
    )

    patched = 0
    no_match = 0
    errors = 0
    limit = args.limit or 9999999
    order_num = 0

    try:
        for uds_order_id, items in groups.items():
            if patched + no_match >= limit:
                logger.info("Лимит %d достигнут", limit)
                break
            order_num += 1
            logger.info(
                "[%d/%d] UDS-SHOP-%d (%d позиций)",
                order_num, len(groups), uds_order_id, len(items),
            )

            try:
                uds_items = await get_uds_order_items(uds_order_id, uds_http)
            except Exception as exc:
                logger.error("UDS API ошибка для %d: %s", uds_order_id, exc)
                errors += len(items)
                continue

            if args.verbose:
                logger.info("  UDS товары: %s", [(u["name"], u["id"]) for u in uds_items])

            for rec in items:
                crm_name = rec.get("name") or ""
                uds_match = _match_uds_item(crm_name, uds_items)
                if not uds_match or not uds_match.get("id"):
                    logger.warning("  ✗ id=%d «%s» — нет совпадения UDS", rec["id"], crm_name)
                    no_match += 1
                    continue

                uds_product_id = str(uds_match["id"])
                tag = "[DRY]" if args.dry_run else "PATCH"
                logger.info(
                    "  %s id=%d «%s» → UDS id=%s (≈«%s»)",
                    tag, rec["id"], crm_name, uds_product_id, uds_match["name"],
                )

                if not args.dry_run:
                    try:
                        result = await integram.call(
                            "update_object",
                            objId=rec["id"],
                            fields={UDS_ITEM_ID_FIELD: uds_product_id},
                        )
                        if result.get("success"):
                            patched += 1
                        else:
                            logger.error("  Ошибка update id=%d: %s", rec["id"], result)
                            errors += 1
                    except Exception as exc:
                        logger.error("  Исключение update id=%d: %s", rec["id"], exc)
                        errors += 1
                else:
                    patched += 1

            await asyncio.sleep(SLEEP_UDS)

    finally:
        await uds_http.aclose()
        await integram.close()

    logger.info(
        "=== Готово: пропатчено %d, без совпадения %d, ошибок %d, пропущено %d ===",
        patched, no_match, errors, skipped,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Патч UDS ID товара в Позиции заказа v2")
    parser.add_argument("--dry-run",  action="store_true",  help="Не писать, только показать")
    parser.add_argument("--limit",    type=int, default=None, help="Лимит позиций")
    parser.add_argument("--verbose",  action="store_true",  help="Вывести детали UDS API")
    parser.add_argument("--email",    default=None, help="Integram v2 email")
    parser.add_argument("--password", default=None, help="Integram v2 password")
    parser.add_argument("--token",    default=None, help="UDS Admin Bearer-токен")
    args = parser.parse_args()
    asyncio.run(main(args))
