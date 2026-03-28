"""Клиент для реального HTTP API Integram (ai2o.ru).

Integram использует собственный REST-подобный API:
  POST /{db}/auth?JSON              — авторизация (login, pwd)
  GET  /{db}/object/{table_id}?JSON — получить записи таблицы
  GET  /{db}/object/{table_id}/?F_I={id}&JSON — получить одну запись
  POST /{db}/object/{table_id}?JSON — создать запись

Конфигурация через .env:
  INTEGRAM_URL      — базовый URL (https://ai2o.ru)
  INTEGRAM_DB       — имя базы (bibot)
  INTEGRAM_LOGIN    — логин
  INTEGRAM_PASSWORD — пароль
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

import httpx

from src import config
from src.crm_constants import (  # noqa: F401 — re-export для обратной совместимости
    TABLE_ORDERS, TABLE_CLIENTS, TABLE_PRODUCTS, TABLE_ORDER_ITEMS,
    TABLE_CATEGORIES, TABLE_SOURCES, TABLE_STATUSES, TABLE_DELIVERY,
    TABLE_USERS, TABLE_STATUS_HISTORY,
    REQ_ORDER_DATE, REQ_ORDER_ADDRESS, REQ_ORDER_DELIVERY_COST,
    REQ_ORDER_ITEMS_TOTAL, REQ_ORDER_TOTAL, REQ_ORDER_TRACKING,
    REQ_ORDER_COMMENT, REQ_ORDER_CLIENT, REQ_ORDER_STATUS,
    REQ_ORDER_DELIVERY_METHOD, REQ_ORDER_SOURCE, REQ_ORDER_MESSENGER,
    REQ_ORDER_SHIPPED_DATE, REQ_ORDER_DELIVERED_DATE,
    REQ_HISTORY_ORDER, REQ_HISTORY_STATUS_FROM, REQ_HISTORY_STATUS_TO,
    REQ_HISTORY_DATE, REQ_HISTORY_COMMENT,
    REQ_CLIENT_PHONE, REQ_CLIENT_TG_ID, REQ_CLIENT_TG_USER,
    REQ_CLIENT_ADDRESS, REQ_CLIENT_CITY, REQ_CLIENT_COMMENT, REQ_CLIENT_SOURCE,
    REQ_ITEM_QTY, REQ_ITEM_PRICE, REQ_ITEM_SUM, REQ_ITEM_PRODUCT, REQ_ITEM_ORDER,
    REQ_PRODUCT_PRICE, REQ_PRODUCT_WEIGHT, REQ_PRODUCT_DESC, REQ_PRODUCT_INSTOCK,
    REQ_PRODUCT_SKU, REQ_PRODUCT_CATEGORY, REQ_PRODUCT_SHORT, REQ_PRODUCT_STOCK,
    REQ_USER_PASSWORD_HASH, REQ_USER_ROLE_OLD, REQ_USER_ROLE,
    REQ_USER_DISPLAY_NAME, REQ_USER_ACTIVE,
    REQ_ORDER_BATCH, REQ_ORDER_CDEK_CONFIRMED, REQ_ORDER_CLIENT_NOTIFIED, REQ_ORDER_STOCK_CHECKED,
    TABLE_BATCHES, REQ_BATCH_DATE, REQ_BATCH_DELIVERY, REQ_BATCH_COUNT, REQ_BATCH_NOTE,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Конфигурация (из src.config, который загружает .env)
# ---------------------------------------------------------------------------

_BASE_URL = (config.INTEGRAM_URL or "https://ai2o.ru").rstrip("/")
_DB = config.INTEGRAM_DB or "bibot"
_LOGIN = config.INTEGRAM_LOGIN or "bibot"
_PASSWORD = config.INTEGRAM_PASSWORD or ""


class IntegramAPIError(Exception):
    """Ошибка при работе с Integram API."""


# Названия месяцев → номер (для парсинга из имени заказа)
_MONTH_MAP = {
    "январ": 1, "феврал": 2, "март": 3, "марта": 3,
    "апрел": 4, "ма": 5, "мая": 5, "май": 5,
    "июн": 6, "июл": 7, "август": 8,
    "сентябр": 9, "октябр": 10, "ноябр": 11, "декабр": 12,
}


def _detect_month(order_name: str, date_str: str) -> str:
    """Определить месяц заказа в формате 'MM.YYYY'.

    Приоритет:
    1. Из имени заказа: '(Март 26)', 'февраль 26', etc.
    2. Из даты (DD.MM.YYYY ...) если есть.
    3. Пустая строка если не удалось определить.
    """
    if order_name:
        name_lower = order_name.lower()
        # Паттерн: "(Март 26)" или "(Октябрь 25)"
        m = re.search(r'\((\S+)\s+(\d{2,4})\)', name_lower)
        if m:
            month_num = _match_month(m.group(1))
            if month_num:
                year = _normalize_year(m.group(2))
                return f"{month_num:02d}.{year}"
        # Паттерн: "февраль 26" (всё имя — месяц + год)
        m = re.search(r'^(\S+)\s+(\d{2,4})$', name_lower.strip())
        if m:
            month_num = _match_month(m.group(1))
            if month_num:
                year = _normalize_year(m.group(2))
                return f"{month_num:02d}.{year}"

    # Fallback: из даты "DD.MM.YYYY ..."
    if date_str:
        parts = date_str.split(".")
        if len(parts) >= 3:
            try:
                mm = int(parts[1])
                yyyy = int(parts[2].split()[0])
                if 1 <= mm <= 12 and yyyy > 2000:
                    return f"{mm:02d}.{yyyy}"
            except (ValueError, IndexError):
                pass

    return ""


def _match_month(text: str) -> Optional[int]:
    """Найти номер месяца по началу слова."""
    text = text.lower().rstrip("ьяеи")
    for prefix, num in _MONTH_MAP.items():
        if text.startswith(prefix) or prefix.startswith(text):
            return num
    return None


def _normalize_year(y: str) -> int:
    """'26' → 2026, '2025' → 2025."""
    n = int(y)
    return n + 2000 if n < 100 else n


def _strip_html(text: str) -> str:
    """Убрать HTML-теги из значения реквизита."""
    if not text:
        return ""
    clean = re.sub(r'<[^>]+>', '', text)
    return clean.strip()


def _extract_ref_text(html: str) -> str:
    """Извлечь текст из HTML-ссылки типа <A HREF="...">Текст</A>."""
    if not html:
        return ""
    m = re.search(r'>([^<]+)<', html)
    return m.group(1).strip() if m else _strip_html(html)


def _extract_ref_id(html: str) -> Optional[int]:
    """Извлечь ID из ссылки типа <A HREF="/bibot/object/1023/?F_I=1137">."""
    if not html:
        return None
    m = re.search(r'F_I=(\d+)', html)
    if m:
        return int(m.group(1))
    # Fallback: plain numeric string (Integram не вернул HTML-ссылку)
    stripped = _strip_html(html).strip()
    if stripped.isdigit():
        return int(stripped)
    return None


class IntegramAPI:
    """Синхронный/асинхронный клиент для реального Integram API."""

    def __init__(self) -> None:
        self._token: Optional[str] = None
        self._xsrf: Optional[str] = None
        self._http: Optional[httpx.AsyncClient] = None

    async def _get_http(self) -> httpx.AsyncClient:
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(
                base_url=_BASE_URL,
                timeout=30.0,
            )
        return self._http

    async def close(self) -> None:
        if self._http and not self._http.is_closed:
            await self._http.aclose()

    async def authenticate(self) -> None:
        """Авторизация: POST /{db}/auth?JSON."""
        http = await self._get_http()
        resp = await http.post(
            f"/{_DB}/auth?JSON",
            data={"login": _LOGIN, "pwd": _PASSWORD},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        resp.raise_for_status()
        data = resp.json()
        if "token" not in data:
            raise IntegramAPIError(f"Ошибка авторизации: {data}")
        self._token = data["token"]
        self._xsrf = data.get("_xsrf", "")

        # Получить актуальный _xsrf токен
        xsrf_resp = await http.get(
            f"/{_DB}/xsrf?JSON",
            cookies={_DB: self._token},
        )
        if xsrf_resp.status_code == 200:
            xsrf_data = xsrf_resp.json()
            self._xsrf = xsrf_data.get("_xsrf", self._xsrf)
            self._token = xsrf_data.get("token", self._token)

        logger.info("Integram авторизация OK (user_id=%s)", data.get("id"))

    async def _request(
        self,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> httpx.Response:
        """HTTP-запрос с автоматической повторной авторизацией при 401/403."""
        http = await self._get_http()
        kwargs.setdefault("cookies", {_DB: self._token or ""})
        resp = await getattr(http, method)(url, **kwargs)

        if resp.status_code in (401, 403) and self._token:
            logger.warning("Integram сессия истекла, повторная авторизация...")
            await self.authenticate()
            kwargs["cookies"] = {_DB: self._token or ""}
            resp = await getattr(http, method)(url, **kwargs)

        resp.raise_for_status()
        return resp

    async def _get_table_page(
        self,
        table_id: int,
        pg: int = 1,
    ) -> dict:
        """Получить страницу таблицы (pg=1 — первая)."""
        url = f"/{_DB}/object/{table_id}/?JSON&F_U=1&pg={pg}"
        resp = await self._request("get", url)
        return resp.json()

    async def get_all_objects(self, table_id: int) -> list[dict[str, Any]]:
        """Получить все записи таблицы (все страницы).

        Возвращает список dict с ключами: id, val, reqs (dict реквизитов).
        """
        all_objects: list[dict[str, Any]] = []
        pg = 1
        page_size = 20
        while True:
            data = await self._get_table_page(table_id, pg=pg)

            # Объекты (id, val)
            objects = data.get("object", [])
            if not objects:
                break

            # Реквизиты: {"object_id": ["val1", "val2", ...], ...}
            reqs = data.get("&object_reqs", {})
            # Заголовки реквизитов (порядок полей)
            head_data = {}
            for key, val in data.items():
                if "&uni_obj_head" in key and "filter" not in key:
                    head_data = val
                    break
            req_ids = head_data.get("typ", [])

            for obj in objects:
                obj_id = str(obj.get("id", ""))
                obj_val = obj.get("val", "")
                obj_reqs_raw = reqs.get(obj_id, [])

                obj_reqs = {}
                for i, req_id in enumerate(req_ids):
                    if i < len(obj_reqs_raw):
                        obj_reqs[req_id] = obj_reqs_raw[i]

                all_objects.append({
                    "id": int(obj_id) if obj_id.isdigit() else 0,
                    "val": obj_val,
                    "reqs": obj_reqs,
                })

            if len(objects) < page_size:
                break
            pg += 1

        return all_objects

    # ------------------------------------------------------------------
    # Запись данных (create / update)
    # ------------------------------------------------------------------

    async def create_object(
        self,
        table_id: int,
        value: str,
        requisites: Optional[dict[str, str]] = None,
    ) -> int:
        """Создать объект в таблице. Возвращает ID нового объекта.

        Использует endpoint _m_new/{typeId} с форматом t{id}=value.

        Args:
            table_id:    ID таблицы (например TABLE_ORDERS=1024).
            value:       Название объекта (главное поле).
            requisites:  Реквизиты {req_id: значение}. Для reference — ID объекта.
        """
        form: dict[str, str] = {
            "_xsrf": self._xsrf or "",
            f"t{table_id}": value,
            "up": "1",
        }
        if requisites:
            for req_id, val in requisites.items():
                form[f"t{req_id}"] = str(val)

        resp = await self._request(
            "post",
            f"/{_DB}/_m_new/{table_id}?JSON",
            data=form,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        data = resp.json()
        obj_id = data.get("obj") or data.get("id")
        if not obj_id:
            raise IntegramAPIError(f"Не удалось создать объект: {data}")
        obj_id = int(obj_id)

        logger.info("Integram: создан объект id=%d в таблице %d", obj_id, table_id)
        return obj_id

    async def set_requisites(
        self,
        obj_id: int,
        table_id: int,
        requisites: dict[str, str],
    ) -> None:
        """Обновить реквизиты объекта через _m_save/{objectId}.

        Формат: t{typeId}=value, t{reqId}=reqValue.
        """
        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        # Получить текущее значение и тип объекта
        edit_resp = await self._request("get", f"/{_DB}/edit_obj/{obj_id}?JSON")
        edit_data = edit_resp.json()
        obj_meta = edit_data.get("obj", {})
        type_id = obj_meta.get("typ", str(table_id))
        obj_val = obj_meta.get("val", "")

        form: dict[str, str] = {
            "_xsrf": self._xsrf or "",
            f"t{type_id}": obj_val,
        }
        for req_id, val in requisites.items():
            form[f"t{req_id}"] = str(val)

        await self._request(
            "post",
            f"/{_DB}/_m_save/{obj_id}?JSON",
            data=form,
            headers=headers,
        )

    async def set_reference_field(
        self,
        obj_id: int,
        req_id: str,
        ref_value_id: str,
    ) -> None:
        """Обновить reference-поле объекта через _m_del (old links) + _m_set (new link).

        _m_save не обновляет reference-поля. Правильный механизм Integram:
        1. Удалить все существующие ссылки для поля (multiselect entries)
        2. Создать одну новую ссылку через _m_set

        Args:
            obj_id:       ID объекта (например, ID заказа)
            req_id:       ID реквизита (например REQ_ORDER_STATUS = "1073")
            ref_value_id: ID значения-ссылки (например STATUS_IDS["Доставлен"] = "1090")
        """
        http = await self._get_http()
        cookies = {_DB: self._token or ""}
        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        # 1. Получить текущие link-записи для данного поля
        edit_resp = await self._request("get", f"/{_DB}/edit_obj/{obj_id}?JSON")
        reqs = edit_resp.json().get("reqs", {})
        field_data = reqs.get(req_id, {})
        ms = field_data.get("multiselect", {}) if isinstance(field_data, dict) else {}
        link_ids = ms.get("id", [])

        # 2. Удалить все старые ссылки
        for lid in link_ids:
            await http.post(
                f"/{_DB}/_m_del/{lid}?JSON",
                data={"_xsrf": self._xsrf or ""},
                headers=headers,
                cookies=cookies,
            )

        # 3. Создать новую ссылку
        await http.post(
            f"/{_DB}/_m_set/{obj_id}?JSON",
            data={"_xsrf": self._xsrf or "", f"t{req_id}": str(ref_value_id)},
            headers=headers,
            cookies=cookies,
        )
        logger.debug(
            "set_reference_field: obj=%d req=%s val=%s (удалено %d старых links)",
            obj_id, req_id, ref_value_id, len(link_ids),
        )

    async def delete_object(self, obj_id: int) -> None:
        """Удалить объект по ID.

        Использует endpoint _m_del/{objectId}?JSON.
        """
        await self._request(
            "post",
            f"/{_DB}/_m_del/{obj_id}?JSON",
            data={"_xsrf": self._xsrf or ""},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        logger.info("Integram: удалён объект id=%d", obj_id)

    async def update_object_value(self, obj_id: int, new_value: str) -> None:
        """Обновить главное поле (val) объекта через _m_save."""
        await self._request(
            "post",
            f"/{_DB}/_m_save/{obj_id}?JSON",
            data={"_xsrf": self._xsrf or "", "t_val": new_value},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        logger.info("Integram: обновлён val объекта id=%d → '%s'", obj_id, new_value)

    async def get_order_items(self, order_id: Optional[int] = None) -> list[dict[str, Any]]:
        """Получить позиции заказов.

        Args:
            order_id: Если указан — только позиции этого заказа.
        """
        raw = await self.get_all_objects(TABLE_ORDER_ITEMS)
        items = []
        for obj in raw:
            r = obj["reqs"]
            ref_order_id = _extract_ref_id(r.get(REQ_ITEM_ORDER, ""))
            if order_id and ref_order_id != order_id:
                continue
            items.append({
                "id": obj["id"],
                "name": obj["val"],
                "order_id": ref_order_id,
                "product_id": _extract_ref_id(r.get(REQ_ITEM_PRODUCT, "")),
                "product_name": _extract_ref_text(r.get(REQ_ITEM_PRODUCT, "")),
                "quantity": int(_parse_number(r.get(REQ_ITEM_QTY, "")) or 0),
                "unit_price": _parse_number(r.get(REQ_ITEM_PRICE, "")) or 0,
                "total": _parse_number(r.get(REQ_ITEM_SUM, "")) or 0,
            })
        return items

    # ------------------------------------------------------------------
    # Высокоуровневые методы для веб-панели
    # ------------------------------------------------------------------

    async def get_orders(self) -> list[dict[str, Any]]:
        """Получить все заказы в удобном формате."""
        raw = await self.get_all_objects(TABLE_ORDERS)
        orders = []
        for obj in raw:
            r = obj["reqs"]
            order_name = obj["val"]
            orders.append({
                "id": obj["id"],
                "number": order_name,
                "month": _detect_month(order_name, _strip_html(r.get(REQ_ORDER_DATE, ""))),
                "date": _strip_html(r.get(REQ_ORDER_DATE, "")),
                "client_name": _extract_ref_text(r.get(REQ_ORDER_CLIENT, "")),
                "client_id": _extract_ref_id(r.get(REQ_ORDER_CLIENT, "")),
                "status": _extract_ref_text(r.get(REQ_ORDER_STATUS, "")),
                "delivery_method": _extract_ref_text(r.get(REQ_ORDER_DELIVERY_METHOD, "")),
                "source": _extract_ref_text(r.get(REQ_ORDER_SOURCE, "")),
                "delivery_address": _strip_html(r.get(REQ_ORDER_ADDRESS, "")),
                "delivery_cost": _parse_number(r.get(REQ_ORDER_DELIVERY_COST, "")),
                "items_total": _parse_number(r.get(REQ_ORDER_ITEMS_TOTAL, "")),
                "total": _parse_number(r.get(REQ_ORDER_TOTAL, "")),
                "tracking_number": _strip_html(r.get(REQ_ORDER_TRACKING, "")),
                "comment": _strip_html(r.get(REQ_ORDER_COMMENT, "")),
                "messenger": _strip_html(r.get(REQ_ORDER_MESSENGER, "")),
                "batch_id": _extract_ref_id(r.get(REQ_ORDER_BATCH, "")) or None,
                "cdek_confirmed": _strip_html(r.get(REQ_ORDER_CDEK_CONFIRMED, "")) not in ("", "0"),
                "client_notified": _strip_html(r.get(REQ_ORDER_CLIENT_NOTIFIED, "")) not in ("", "0"),
                "stock_checked": _strip_html(r.get(REQ_ORDER_STOCK_CHECKED, "")) not in ("", "0"),
            })
        return orders

    async def get_clients(self) -> list[dict[str, Any]]:
        """Получить всех клиентов."""
        raw = await self.get_all_objects(TABLE_CLIENTS)
        clients = []
        for obj in raw:
            r = obj["reqs"]
            clients.append({
                "id": obj["id"],
                "name": obj["val"],
                "phone": _strip_html(r.get(REQ_CLIENT_PHONE, "")),
                "telegram_id": _strip_html(r.get(REQ_CLIENT_TG_ID, "")),
                "telegram_username": _strip_html(r.get(REQ_CLIENT_TG_USER, "")),
                "address": _strip_html(r.get(REQ_CLIENT_ADDRESS, "")),
                "city": _strip_html(r.get(REQ_CLIENT_CITY, "")),
                "source": _extract_ref_text(r.get(REQ_CLIENT_SOURCE, "")),
                "comment": _strip_html(r.get(REQ_CLIENT_COMMENT, "")),
            })
        return clients

    async def get_products(self) -> list[dict[str, Any]]:
        """Получить все товары."""
        raw = await self.get_all_objects(TABLE_PRODUCTS)
        products = []
        for obj in raw:
            r = obj["reqs"]
            products.append({
                "id": obj["id"],
                "name": obj["val"],
                "price": _parse_number(r.get(REQ_PRODUCT_PRICE, "")),
                "weight": _parse_number(r.get(REQ_PRODUCT_WEIGHT, "")),
                "description": _strip_html(r.get(REQ_PRODUCT_DESC, "")),
                "in_stock": _strip_html(r.get(REQ_PRODUCT_INSTOCK, "")) != "",
                "sku_uds": _strip_html(r.get(REQ_PRODUCT_SKU, "")),
                "category": _extract_ref_text(r.get(REQ_PRODUCT_CATEGORY, "")),
                "short_name": _strip_html(r.get(REQ_PRODUCT_SHORT, "")),
                "stock": _parse_number(r.get(REQ_PRODUCT_STOCK, "")),
            })
        return products

    async def get_batches(self) -> list[dict[str, Any]]:
        """Получить все партии отправки."""
        raw = await self.get_all_objects(TABLE_BATCHES)
        result = []
        for obj in raw:
            r = obj["reqs"]
            result.append({
                "id": obj["id"],
                "name": obj["val"],
                "date": _strip_html(r.get(REQ_BATCH_DATE, "")),
                "delivery_method": _strip_html(r.get(REQ_BATCH_DELIVERY, "")),
                "count": int(_strip_html(r.get(REQ_BATCH_COUNT, "0")) or 0),
                "note": _strip_html(r.get(REQ_BATCH_NOTE, "")),
            })
        result.sort(key=lambda x: x.get("date", ""), reverse=True)
        return result

    async def get_dashboard_stats(self) -> dict[str, Any]:
        """Статистика для дашборда."""
        orders = await self.get_orders()
        clients = await self.get_clients()

        total_orders = len(orders)
        total_clients = len(clients)
        total_revenue = sum(o.get("total") or 0.0 for o in orders)
        avg_order = total_revenue / total_orders if total_orders else 0.0
        new_orders = sum(1 for o in orders if o.get("status") == "Новый")
        delivered_orders = sum(1 for o in orders if o.get("status") == "Доставлен")

        return {
            "total_orders": total_orders,
            "total_clients": total_clients,
            "total_revenue": total_revenue,
            "avg_order": round(avg_order),
            "new_orders": new_orders,
            "delivered_orders": delivered_orders,
        }


def _parse_number(val: str) -> Optional[float]:
    """Парсинг числа из строки (может быть пустая или с пробелами)."""
    if not val:
        return None
    clean = _strip_html(val).replace(" ", "").replace(",", ".")
    if not clean:
        return None
    try:
        return float(clean)
    except ValueError:
        return None
