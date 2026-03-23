"""Роутер товаров и склада."""

import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException

from src.integram_api import IntegramAPIError
from src.integram_client import IntegramError
from src.web.deps import (
    CurrentUser,
    ProductCreate,
    ProductUpdate,
    StockUpdate,
    _DEFAULT_PAGE_SIZE,
    _get_crm,
    _paginate,
    _product_to_dict,
    _require_role,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["products"])


@router.get("/api/products")
async def list_products(
    in_stock_only: bool = False,
    search: Optional[str] = None,
    page: int = 1,
    per_page: int = _DEFAULT_PAGE_SIZE,
    _: CurrentUser = Depends(_require_role("admin", "warehouse")),
) -> dict[str, Any]:
    try:
        crm = await _get_crm()
        try:
            products = await crm.get_products(in_stock_only=in_stock_only)
            result = [_product_to_dict(p) for p in products]
            return _paginate(result, page, per_page, search,
                             search_fields=["name", "category", "description", "sku_uds"])
        finally:
            await crm.close()
    except (IntegramError, IntegramAPIError) as exc:
        logger.error("Ошибка Integram: %s", exc)
        raise HTTPException(status_code=502, detail="Ошибка CRM")


@router.post("/api/products")
async def create_product(
    body: ProductCreate,
    _: CurrentUser = Depends(_require_role("admin")),
) -> dict:
    try:
        crm = await _get_crm()
        try:
            product_id = await crm.create_product(
                body.name, price=body.price, weight=body.weight,
                description=body.description, in_stock=body.in_stock,
                sku_uds=body.sku_uds, category=body.category,
                short_name=body.short_name, stock=body.stock,
            )
            return {"ok": True, "product_id": product_id}
        finally:
            await crm.close()
    except (IntegramError, IntegramAPIError) as exc:
        logger.error("Ошибка Integram: %s", exc)
        raise HTTPException(502, "Ошибка CRM")


@router.patch("/api/products/{product_id}")
async def update_product(
    product_id: int,
    body: ProductUpdate,
    _: CurrentUser = Depends(_require_role("admin")),
) -> dict:
    try:
        crm = await _get_crm()
        try:
            kwargs = body.model_dump(exclude_none=True)
            if kwargs:
                await crm.update_product(product_id, **kwargs)
            return {"ok": True, "product_id": product_id}
        finally:
            await crm.close()
    except (IntegramError, IntegramAPIError) as exc:
        logger.error("Ошибка Integram: %s", exc)
        raise HTTPException(502, "Ошибка CRM")


@router.delete("/api/products/{product_id}")
async def delete_product(
    product_id: int,
    _: CurrentUser = Depends(_require_role("admin")),
) -> dict:
    try:
        crm = await _get_crm()
        try:
            await crm.delete_product(product_id)
            return {"ok": True, "product_id": product_id}
        finally:
            await crm.close()
    except (IntegramError, IntegramAPIError) as exc:
        logger.error("Ошибка Integram: %s", exc)
        raise HTTPException(502, "Ошибка CRM")


@router.patch("/api/products/{product_id}/stock")
async def update_product_stock(
    product_id: int,
    body: StockUpdate,
    _: CurrentUser = Depends(_require_role("admin", "warehouse")),
) -> dict:
    try:
        crm = await _get_crm()
        try:
            await crm.update_product_stock(product_id, body.stock)
            return {"ok": True, "product_id": product_id, "stock": body.stock}
        finally:
            await crm.close()
    except (IntegramError, IntegramAPIError) as exc:
        logger.error("Ошибка Integram: %s", exc)
        raise HTTPException(502, "Ошибка CRM")
