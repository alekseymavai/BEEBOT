"""Роутер PDF-отчётов по продажам.

Эндпоинт:
  GET /api/reports/sales?period=30d|90d|365d

Требует роль admin (JWT).
Возвращает PDF-файл для скачивания.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response

from src.integram_api import IntegramAPIError
from src.integram_client import IntegramError
from src.pdf_report import generate_sales_report
from src.web.deps import CurrentUser, _get_crm, _require_role

logger = logging.getLogger(__name__)

router = APIRouter(tags=["reports"])

_PERIOD_DAYS: dict[str, int] = {
    "30d": 30,
    "90d": 90,
    "365d": 365,
}


def _filter_orders_by_days(orders: list, days: int) -> list:
    """Отфильтровать заказы по количеству дней назад."""
    cutoff = datetime.now() - timedelta(days=days)
    result = []
    for order in orders:
        try:
            if isinstance(order.date, datetime):
                dt = order.date
            else:
                raw = str(order.date)
                if "." in raw and len(raw) >= 8:
                    parts = raw.split(".")
                    dt = datetime(int(parts[2][:4]), int(parts[1]), int(parts[0]))
                else:
                    dt = datetime.fromisoformat(raw)
            if dt >= cutoff:
                result.append(order)
        except Exception:
            result.append(order)
    return result


@router.get("/api/reports/sales")
async def sales_pdf_report(
    period: str = Query(default="30d", pattern="^(30d|90d|365d)$"),
    _: CurrentUser = Depends(_require_role("admin")),
) -> Response:
    """Сгенерировать PDF-отчёт по продажам за выбранный период.

    Args:
        period: Период отчёта — 30d (месяц), 90d (квартал) или 365d (год).

    Returns:
        PDF-файл для скачивания.
    """
    crm = await _get_crm()
    try:
        days = _PERIOD_DAYS[period]

        # Загрузить заказы
        try:
            all_orders = await crm.get_orders()
        except Exception as exc:
            logger.error("Не удалось загрузить заказы для PDF-отчёта: %s", exc)
            raise HTTPException(502, "Ошибка загрузки заказов из CRM")

        orders = _filter_orders_by_days(all_orders, days)

        # Загрузить позиции заказов
        items_by_order: dict = {}
        try:
            all_items = await crm.get_order_items_bulk()
            for item in all_items:
                items_by_order.setdefault(item.order_id, []).append(item)
        except Exception as exc:
            logger.warning("Не удалось загрузить позиции для PDF-отчёта: %s", exc)

        # Загрузить товары для алертов остатка
        products: list = []
        try:
            products = await crm.get_products(in_stock_only=False)
        except Exception as exc:
            logger.warning("Не удалось загрузить товары для PDF-отчёта: %s", exc)

        # Генерировать PDF
        try:
            pdf_bytes = generate_sales_report(
                orders=orders,
                items_by_order=items_by_order,
                period=period,
                products=products,
            )
        except Exception as exc:
            logger.error("Ошибка генерации PDF: %s", exc)
            raise HTTPException(500, "Ошибка генерации PDF-отчёта")

        filename = f"beebot_report_{period}_{datetime.now().strftime('%Y%m%d')}.pdf"
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )

    except HTTPException:
        raise
    except (IntegramError, IntegramAPIError) as exc:
        logger.error("Ошибка Integram при генерации PDF-отчёта: %s", exc)
        raise HTTPException(502, "Ошибка CRM")
    finally:
        await crm.close()
