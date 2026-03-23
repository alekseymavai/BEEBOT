"""Роутер экспорта CSV."""

import csv
import io
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from src.integram_api import IntegramAPIError
from src.integram_client import IntegramError
from src.web.deps import (
    CurrentUser,
    _client_to_dict,
    _get_crm,
    _order_to_dict,
    _product_to_dict,
    _require_role,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["export"])


def _to_csv(rows: list[dict], fields: list[str], headers: list[str]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(headers)
    for row in rows:
        writer.writerow([row.get(f, "") for f in fields])
    return buf.getvalue()


@router.get("/api/export/orders")
async def export_orders_csv(
    status: Optional[str] = None,
    _: CurrentUser = Depends(_require_role("admin")),
) -> StreamingResponse:
    try:
        crm = await _get_crm()
        try:
            orders = await crm.get_orders(status=status)
            rows = [_order_to_dict(o) for o in orders]
            content = _to_csv(
                rows,
                fields=["number", "date", "client_name", "status", "delivery_method",
                        "delivery_address", "items_total", "delivery_cost", "total", "source"],
                headers=["Номер", "Дата", "Клиент", "Статус", "Доставка",
                         "Адрес", "Сумма товаров", "Доставка ₽", "Итого", "Источник"],
            )
            return StreamingResponse(
                io.BytesIO(content.encode("utf-8-sig")),
                media_type="text/csv; charset=utf-8",
                headers={"Content-Disposition": "attachment; filename=orders.csv"},
            )
        finally:
            await crm.close()
    except (IntegramError, IntegramAPIError) as exc:
        logger.error("Ошибка Integram: %s", exc)
        raise HTTPException(502, "Ошибка CRM")


@router.get("/api/export/clients")
async def export_clients_csv(
    _: CurrentUser = Depends(_require_role("admin")),
) -> StreamingResponse:
    try:
        crm = await _get_crm()
        try:
            clients = await crm.get_clients()
            rows = [_client_to_dict(c) for c in clients]
            content = _to_csv(
                rows,
                fields=["id", "name", "phone", "city", "address", "source", "telegram_username"],
                headers=["ID", "ФИО", "Телефон", "Город", "Адрес", "Источник", "Telegram"],
            )
            return StreamingResponse(
                io.BytesIO(content.encode("utf-8-sig")),
                media_type="text/csv; charset=utf-8",
                headers={"Content-Disposition": "attachment; filename=clients.csv"},
            )
        finally:
            await crm.close()
    except (IntegramError, IntegramAPIError) as exc:
        logger.error("Ошибка Integram: %s", exc)
        raise HTTPException(502, "Ошибка CRM")


@router.get("/api/export/products")
async def export_products_csv(
    _: CurrentUser = Depends(_require_role("admin")),
) -> StreamingResponse:
    try:
        crm = await _get_crm()
        try:
            products = await crm.get_products(in_stock_only=False)
            rows = [_product_to_dict(p) for p in products]
            content = _to_csv(
                rows,
                fields=["id", "name", "category", "price", "weight", "stock", "in_stock"],
                headers=["ID", "Название", "Категория", "Цена", "Вес", "Остаток", "В наличии"],
            )
            return StreamingResponse(
                io.BytesIO(content.encode("utf-8-sig")),
                media_type="text/csv; charset=utf-8",
                headers={"Content-Disposition": "attachment; filename=products.csv"},
            )
        finally:
            await crm.close()
    except (IntegramError, IntegramAPIError) as exc:
        logger.error("Ошибка Integram: %s", exc)
        raise HTTPException(502, "Ошибка CRM")
