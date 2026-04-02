"""Роутер заказов и позиций заказа."""

import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request

from src.crm_constants import STATUS_IDS, DELIVERY_IDS as DELIVERY_METHOD_IDS
from src.integram_api import IntegramAPIError
from src.integram_client import IntegramError, IntegramNotFoundError
from src.web.deps import (
    CurrentUser,
    ChecklistUpdate,
    EDITABLE_STATUSES,
    ItemCreate,
    ItemUpdate,
    OrderCreate,
    OrderUpdate,
    StatusUpdate,
    TrackingUpdate,
    _DEFAULT_PAGE_SIZE,
    _WAREHOUSE_STATUS_TRANSITIONS,
    _get_crm,
    _order_to_dict,
    _paginate,
    _require_role,
    get_items_cache,
    get_orders_cache,
    invalidate_items_cache,
    invalidate_orders_cache,
    push_event,
)
from src.web.notifications import (
    notify_client_status_change,
    notify_client_tracking,
    notify_beekeeper_status_change,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["orders"])


@router.get("/api/orders")
async def list_orders(
    status: Optional[str] = None,
    source: Optional[str] = None,
    client_id: Optional[int] = None,
    search: Optional[str] = None,
    page: int = 1,
    per_page: int = _DEFAULT_PAGE_SIZE,
    user: CurrentUser = Depends(_require_role("admin", "warehouse")),
) -> dict[str, Any]:
    try:
        crm = await _get_crm()
        try:
            all_orders = await get_orders_cache(crm)
            result = [_order_to_dict(o) for o in all_orders]
            if status:
                result = [o for o in result if o.get("status") == status]
            if client_id:
                result = [o for o in result if o.get("client_id") == client_id]
            if source:
                result = [o for o in result if o.get("source") == source]
            return _paginate(result, page, per_page, search,
                             search_fields=["number", "client_name", "delivery_address", "comment"])
        finally:
            await crm.close()
    except (IntegramError, IntegramAPIError) as exc:
        logger.error("Ошибка Integram: %s", exc)
        raise HTTPException(status_code=502, detail="Ошибка CRM")


@router.post("/api/orders", status_code=201)
async def create_order_web(
    body: OrderCreate,
    request: Request,
    _: CurrentUser = Depends(_require_role("admin", "warehouse")),
) -> dict[str, Any]:
    try:
        order_svc = getattr(request.app.state, "order_service", None)
        crm = await _get_crm()
        try:
            items = [{"product_id": i.product_id, "quantity": i.quantity, "unit_price": i.unit_price}
                     for i in body.items]
            if order_svc:
                # Через OrderService (единая логика + уведомления)
                order = await order_svc.create_order_with_client(
                    telegram_id=0,
                    full_name=body.client_name,
                    phone=body.phone or "",
                    items=items,
                    delivery_method=body.delivery_method or "",
                    address=body.delivery_address,
                    delivery_cost=body.delivery_cost or 0,
                    source=body.source or "Сайт",
                )
            else:
                # Fallback: прямой CRM
                client = await crm.get_or_create_client(
                    telegram_id=0, full_name=body.client_name,
                    phone=body.phone, source=body.source or "Сайт",
                )
                order = await crm.create_order(
                    client_id=client.id, items=items,
                    delivery_method=body.delivery_method or "",
                    delivery_address=body.delivery_address,
                    delivery_cost=body.delivery_cost or 0,
                    source=body.source or "Сайт",
                )
            invalidate_orders_cache()
            return _order_to_dict(order)
        finally:
            await crm.close()
    except (IntegramError, IntegramAPIError) as exc:
        logger.error("Ошибка Integram при создании заказа: %s", exc)
        raise HTTPException(status_code=502, detail="Ошибка CRM")


@router.get("/api/orders/{order_id}")
async def get_order(
    order_id: int,
    _: CurrentUser = Depends(_require_role("admin", "warehouse")),
) -> dict[str, Any]:
    try:
        crm = await _get_crm()
        try:
            order = await crm.get_order(order_id)
            items_cache = await get_items_cache(crm)
            order.items = items_cache.get(order_id, [])
            d = _order_to_dict(order)
            d["editable"] = order.status in EDITABLE_STATUSES
            return d
        except IntegramNotFoundError:
            raise HTTPException(status_code=404, detail="Заказ не найден")
        finally:
            await crm.close()
    except (IntegramError, IntegramAPIError) as exc:
        logger.error("Ошибка Integram: %s", exc)
        raise HTTPException(status_code=502, detail="Ошибка CRM")


@router.get("/api/orders/{order_id}/history")
async def get_order_history(
    order_id: int,
    _: CurrentUser = Depends(_require_role("admin", "warehouse")),
) -> list[dict]:
    try:
        crm = await _get_crm()
        try:
            return await crm.get_order_history(order_id)
        finally:
            await crm.close()
    except (IntegramError, IntegramAPIError) as exc:
        logger.error("Ошибка Integram: %s", exc)
        raise HTTPException(502, "Ошибка CRM")


@router.patch("/api/orders/{order_id}/status")
async def update_order_status(
    order_id: int,
    body: StatusUpdate,
    user: CurrentUser = Depends(_require_role("admin", "warehouse")),
) -> dict:
    if body.status not in STATUS_IDS:
        raise HTTPException(400, f"Неизвестный статус: {body.status}")
    try:
        crm = await _get_crm()
        try:
            order = None
            prev_status: str | None = None
            if user.role == "warehouse":
                try:
                    order = await crm.get_order(order_id)
                except IntegramNotFoundError:
                    raise HTTPException(404, "Заказ не найден")
                if (order.status, body.status) not in _WAREHOUSE_STATUS_TRANSITIONS:
                    raise HTTPException(403, f"Склад не может менять статус с «{order.status}» на «{body.status}»")
                prev_status = order.status

            await crm.update_order_status(order_id, body.status, from_status=prev_status)

            notified = False
            try:
                order = await crm.get_order(order_id)
                if order.client_id:
                    tg_id = await crm.get_client_telegram_id(order.client_id)
                    if tg_id:
                        notified = await notify_client_status_change(
                            telegram_id=tg_id, order_number=order.number or str(order_id),
                            new_status=body.status, tracking_number=order.tracking_number,
                        )
            except Exception as e:
                logger.warning("Не удалось уведомить клиента: %s", e)

            try:
                await notify_beekeeper_status_change(
                    order_number=order.number if order else str(order_id),
                    new_status=body.status,
                    client_name=order.client_name if order and hasattr(order, "client_name") else "",
                    tracking_number=order.tracking_number if order else None,
                )
            except Exception as e:
                logger.warning("Не удалось уведомить пчеловода: %s", e)

            await push_event("order_status", {
                "order_id": order_id,
                "order_number": order.number if order else str(order_id),
                "status": body.status,
            })
            invalidate_orders_cache()
            return {"ok": True, "order_id": order_id, "status": body.status, "notified": notified}
        finally:
            await crm.close()
    except (IntegramError, IntegramAPIError) as exc:
        logger.error("Ошибка Integram: %s", exc)
        raise HTTPException(502, "Ошибка CRM")


@router.patch("/api/orders/{order_id}/checklist")
async def update_order_checklist(
    order_id: int,
    body: ChecklistUpdate,
    _: CurrentUser = Depends(_require_role("admin", "warehouse")),
) -> dict:
    try:
        crm = await _get_crm()
        try:
            await crm.update_order_checklist(
                order_id,
                cdek_confirmed=body.cdek_confirmed,
                client_notified=body.client_notified,
                stock_checked=body.stock_checked,
            )
            return {"ok": True, "order_id": order_id}
        finally:
            await crm.close()
    except (IntegramError, IntegramAPIError) as exc:
        logger.error("Ошибка Integram: %s", exc)
        raise HTTPException(502, "Ошибка CRM")


@router.patch("/api/orders/{order_id}/tracking")
async def update_order_tracking(
    order_id: int,
    body: TrackingUpdate,
    _: CurrentUser = Depends(_require_role("admin")),
) -> dict:
    try:
        crm = await _get_crm()
        try:
            await crm.update_order(order_id, tracking_number=body.tracking_number)
            notified = False
            order = None
            try:
                order = await crm.get_order(order_id)
                if order.client_id:
                    tg_id = await crm.get_client_telegram_id(order.client_id)
                    if tg_id:
                        notified = await notify_client_tracking(
                            telegram_id=tg_id, order_number=order.number or str(order_id),
                            tracking_number=body.tracking_number,
                        )
            except Exception as e:
                logger.warning("Не удалось уведомить клиента: %s", e)

            await push_event("order_tracking", {"order_id": order_id,
                                                 "tracking_number": body.tracking_number})
            return {"ok": True, "order_id": order_id,
                    "tracking_number": body.tracking_number, "notified": notified}
        finally:
            await crm.close()
    except (IntegramError, IntegramAPIError) as exc:
        logger.error("Ошибка Integram: %s", exc)
        raise HTTPException(502, "Ошибка CRM")


@router.patch("/api/orders/{order_id}")
async def update_order(
    order_id: int,
    body: OrderUpdate,
    _: CurrentUser = Depends(_require_role("admin")),
) -> dict:
    try:
        crm = await _get_crm()
        try:
            try:
                order = await crm.get_order(order_id)
            except IntegramNotFoundError:
                raise HTTPException(404, "Заказ не найден")
            if order.status not in EDITABLE_STATUSES:
                raise HTTPException(409, f"Заказ в статусе «{order.status}» нельзя редактировать")
            if body.delivery_method is not None and body.delivery_method not in DELIVERY_METHOD_IDS:
                raise HTTPException(400, f"Неизвестный способ доставки: {body.delivery_method}")

            kwargs: dict[str, Any] = {}
            if body.delivery_address is not None:
                kwargs["delivery_address"] = body.delivery_address
            if body.delivery_method is not None:
                kwargs["delivery_method"] = body.delivery_method
            if body.delivery_cost is not None:
                kwargs["delivery_cost"] = body.delivery_cost
                kwargs["total"] = (order.items_total or 0) + body.delivery_cost
            if body.comment is not None:
                kwargs["comment"] = body.comment
            if kwargs:
                await crm.update_order(order_id, **kwargs)
            return {"ok": True, "order_id": order_id}
        finally:
            await crm.close()
    except (IntegramError, IntegramAPIError) as exc:
        logger.error("Ошибка Integram: %s", exc)
        raise HTTPException(502, "Ошибка CRM")


# ---------------------------------------------------------------------------
# Позиции заказа
# ---------------------------------------------------------------------------

order_items_router = APIRouter(tags=["order-items"])


@order_items_router.get("/api/orders/{order_id}/items")
async def get_order_items(
    order_id: int,
    _: CurrentUser = Depends(_require_role("admin", "warehouse")),
) -> list[dict]:
    try:
        crm = await _get_crm()
        try:
            items_cache = await get_items_cache(crm)
            return [i.model_dump() for i in items_cache.get(order_id, [])]
        finally:
            await crm.close()
    except (IntegramError, IntegramAPIError) as exc:
        logger.error("Ошибка Integram: %s", exc)
        raise HTTPException(502, "Ошибка CRM")


@order_items_router.post("/api/orders/{order_id}/items")
async def add_order_item(
    order_id: int, body: ItemCreate, _: CurrentUser = Depends(_require_role("admin"))
) -> dict:
    try:
        crm = await _get_crm()
        try:
            try:
                order = await crm.get_order(order_id)
            except IntegramNotFoundError:
                raise HTTPException(404, "Заказ не найден")
            if order.status not in EDITABLE_STATUSES:
                raise HTTPException(409, f"Заказ в статусе «{order.status}» нельзя редактировать")
            item_id = await crm.add_order_item(order_id, body.product_id, body.quantity, body.unit_price)
            await crm.recalculate_order_totals(order_id)
            invalidate_items_cache()
            return {"ok": True, "item_id": item_id}
        finally:
            await crm.close()
    except (IntegramError, IntegramAPIError) as exc:
        logger.error("Ошибка Integram: %s", exc)
        raise HTTPException(502, "Ошибка CRM")


@order_items_router.patch("/api/orders/{order_id}/items/{item_id}")
async def update_order_item(
    order_id: int, item_id: int, body: ItemUpdate,
    _: CurrentUser = Depends(_require_role("admin"))
) -> dict:
    try:
        crm = await _get_crm()
        try:
            try:
                order = await crm.get_order(order_id)
            except IntegramNotFoundError:
                raise HTTPException(404, "Заказ не найден")
            if order.status not in EDITABLE_STATUSES:
                raise HTTPException(409, f"Заказ в статусе «{order.status}» нельзя редактировать")
            items = await crm.get_order_items(order_id)
            current = next((i for i in items if i.id == item_id), None)
            if not current:
                raise HTTPException(404, "Позиция не найдена")
            qty = body.quantity if body.quantity is not None else current.quantity
            price = body.unit_price if body.unit_price is not None else current.unit_price
            await crm.update_order_item(item_id, qty=qty, price=price)
            await crm.recalculate_order_totals(order_id)
            invalidate_items_cache()
            return {"ok": True, "item_id": item_id}
        finally:
            await crm.close()
    except (IntegramError, IntegramAPIError) as exc:
        logger.error("Ошибка Integram: %s", exc)
        raise HTTPException(502, "Ошибка CRM")


@order_items_router.delete("/api/orders/{order_id}/items/{item_id}")
async def delete_order_item(
    order_id: int, item_id: int, _: CurrentUser = Depends(_require_role("admin"))
) -> dict:
    try:
        crm = await _get_crm()
        try:
            try:
                order = await crm.get_order(order_id)
            except IntegramNotFoundError:
                raise HTTPException(404, "Заказ не найден")
            if order.status not in EDITABLE_STATUSES:
                raise HTTPException(409, f"Заказ в статусе «{order.status}» нельзя редактировать")
            await crm.delete_order_item(item_id)
            await crm.recalculate_order_totals(order_id)
            invalidate_items_cache()
            return {"ok": True, "item_id": item_id}
        finally:
            await crm.close()
    except (IntegramError, IntegramAPIError) as exc:
        logger.error("Ошибка Integram: %s", exc)
        raise HTTPException(502, "Ошибка CRM")
