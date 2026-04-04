"""Bus Handlers — маршрутизация событий из Redis Streams → Service Layer.

Backend подписывается на stream:requests и обрабатывает запросы от бота.
Ответы отправляются обратно через reply_to.

Также публикует события (Backend → Bot) в stream:events.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from src.bus import EventBus, make_event

logger = logging.getLogger(__name__)

STREAM_REQUESTS = "stream:requests"
STREAM_EVENTS = "stream:events"
GROUP_BACKEND = "backend"


class BusHandlers:
    """Обработчик событий из шины → сервисы."""

    def __init__(
        self,
        bus: EventBus,
        *,
        order_service=None,
        consult_service=None,
        analytics_service=None,
    ):
        self._bus = bus
        self._order_svc = order_service
        self._consult_svc = consult_service
        self._analytics_svc = analytics_service

        self._handlers: dict[str, Any] = {
            "consult": self._handle_consult,
            "create_order": self._handle_create_order,
            "update_order_status": self._handle_update_status,
            "get_orders": self._handle_get_orders,
            "get_order": self._handle_get_order,
            "get_order_items": self._handle_get_order_items,
            "analytics_query": self._handle_analytics,
            "ping": self._handle_ping,
        }

    async def start(self) -> None:
        """Подписаться на stream:requests."""
        await self._bus.subscribe(
            STREAM_REQUESTS, GROUP_BACKEND, self._dispatch,
        )
        logger.info("BusHandlers: подписан на %s", STREAM_REQUESTS)

    async def publish_event(self, event_type: str, payload: dict) -> None:
        """Отправить событие в stream:events (Backend → Bot)."""
        event = make_event(event_type, payload)
        await self._bus.publish(STREAM_EVENTS, event)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    async def _dispatch(self, event: dict) -> Optional[dict]:
        """Маршрутизация события к нужному обработчику."""
        event_type = event.get("type", "")
        handler = self._handlers.get(event_type)

        if not handler:
            logger.warning("BusHandlers: неизвестный тип события '%s'", event_type)
            return {"error": f"unknown event type: {event_type}"}

        try:
            result = await handler(event.get("payload", {}))
            return result
        except Exception as e:
            logger.exception("BusHandlers: ошибка обработки '%s'", event_type)
            return {"error": str(e)}

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    async def _handle_ping(self, payload: dict) -> dict:
        return {"pong": True}

    async def _handle_consult(self, payload: dict) -> dict:
        """Консультация: запрос к KB + LLM."""
        if not self._consult_svc:
            return {"error": "ConsultService не подключён"}
        response, chunks = self._consult_svc.answer(
            query=payload.get("query", ""),
            history=payload.get("history"),
            style=payload.get("style"),
        )
        return {
            "response": response,
            "chunks": [{"text": c.get("text", ""), "source": c.get("source", "")} for c in chunks],
        }

    async def _handle_create_order(self, payload: dict) -> dict:
        """Создать заказ."""
        if not self._order_svc:
            return {"error": "OrderService не подключён"}

        order = await self._order_svc.create_order(
            client_id=payload["client_id"],
            items=payload.get("items", []),
            delivery_method=payload.get("delivery_method"),
            delivery_address=payload.get("delivery_address"),
            delivery_cost=payload.get("delivery_cost", 0),
            source=payload.get("source", "Telegram"),
        )
        return {
            "order_id": order.id,
            "order_number": order.number,
            "total": order.total,
        }

    async def _handle_update_status(self, payload: dict) -> dict:
        """Обновить статус заказа."""
        if not self._order_svc:
            return {"error": "OrderService не подключён"}

        order = await self._order_svc.update_status(
            order_id=payload["order_id"],
            new_status=payload["status"],
            comment=payload.get("comment", ""),
            role=payload.get("role", "admin"),
        )
        return {
            "order_id": order.id,
            "status": order.status,
        }

    async def _handle_get_orders(self, payload: dict) -> dict:
        """Получить список заказов."""
        if not self._order_svc:
            return {"error": "OrderService не подключён"}

        orders = await self._order_svc.get_orders(
            client_id=payload.get("client_id"),
            status=payload.get("status"),
        )
        return {
            "orders": [
                {"id": o.id, "number": o.number, "status": o.status,
                 "total": o.total, "date": str(o.date)}
                for o in orders
            ],
        }

    async def _handle_get_order(self, payload: dict) -> dict:
        """Получить заказ по ID."""
        if not self._order_svc:
            return {"error": "OrderService не подключён"}

        order = await self._order_svc.get_order(payload["order_id"])
        return {
            "id": order.id,
            "number": order.number,
            "status": order.status,
            "total": order.total,
            "client_name": order.client_name,
            "date": str(order.date),
        }

    async def _handle_get_order_items(self, payload: dict) -> dict:
        """Получить позиции заказа."""
        if not self._order_svc:
            return {"error": "OrderService не подключён"}

        items = await self._order_svc.get_order_items(payload["order_id"])
        return {
            "items": [
                {"id": i.id, "product_name": i.product_name,
                 "quantity": i.quantity, "unit_price": i.unit_price}
                for i in items
            ],
        }

    async def _handle_analytics(self, payload: dict) -> dict:
        """Аналитический запрос."""
        if not self._analytics_svc:
            return {"error": "AnalyticsService не подключён"}
        query = payload.get("query", "общая статистика")
        report = await self._analytics_svc.handle_query(query)
        return {"report": report}
