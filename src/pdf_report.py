"""Генератор PDF-отчётов по продажам.

Использует reportlab для создания PDF-документов с:
- Шапкой отчёта (название, период)
- Таблицей выручки по месяцам
- Топ-10 товаров по выручке
- ABC-анализом (A: 80% выручки, B: 80–95%, C: 95–100%)
- Алертами о низком остатке
"""

from __future__ import annotations

import io
import logging
from collections import Counter, defaultdict
from datetime import datetime
from typing import Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

logger = logging.getLogger(__name__)

_PAGE_WIDTH, _PAGE_HEIGHT = A4
_MARGIN = 1.5 * cm

_RU_MONTHS = {
    "01": "Янв", "02": "Фев", "03": "Мар", "04": "Апр",
    "05": "Май", "06": "Июн", "07": "Июл", "08": "Авг",
    "09": "Сен", "10": "Окт", "11": "Ноя", "12": "Дек",
}

# Соответствие числового периода (30d/90d/365d) русским меткам
_PERIOD_LABELS: dict[str, str] = {
    "30d": "за 30 дней",
    "90d": "за 90 дней",
    "365d": "за 365 дней",
    "week": "за неделю",
    "month": "за месяц",
    "all": "за всё время",
}


def _period_label(period: str) -> str:
    return _PERIOD_LABELS.get(period, period)


def _build_styles():
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=16,
        spaceAfter=6,
        textColor=colors.HexColor("#1a1a2e"),
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontSize=11,
        spaceAfter=14,
        textColor=colors.HexColor("#4a4a6a"),
    )
    section_style = ParagraphStyle(
        "SectionHeader",
        parent=styles["Heading2"],
        fontSize=12,
        spaceBefore=12,
        spaceAfter=6,
        textColor=colors.HexColor("#2c5282"),
    )
    body_style = ParagraphStyle(
        "BodyText",
        parent=styles["Normal"],
        fontSize=9,
        leading=13,
    )

    return {
        "title": title_style,
        "subtitle": subtitle_style,
        "section": section_style,
        "body": body_style,
    }


# ---------------------------------------------------------------------------
# Общий стиль таблицы
# ---------------------------------------------------------------------------

_TABLE_HEADER_BG = colors.HexColor("#ebf4ff")
_TABLE_BORDER = colors.HexColor("#bee3f8")
_TABLE_ROW_ALT = colors.HexColor("#f7fafc")

_BASE_TABLE_STYLE = TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), _TABLE_HEADER_BG),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#2c5282")),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, 0), 9),
    ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
    ("TOPPADDING", (0, 0), (-1, 0), 6),
    ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
    ("FONTSIZE", (0, 1), (-1, -1), 9),
    ("TOPPADDING", (0, 1), (-1, -1), 4),
    ("BOTTOMPADDING", (0, 1), (-1, -1), 4),
    ("GRID", (0, 0), (-1, -1), 0.5, _TABLE_BORDER),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, _TABLE_ROW_ALT]),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
])


# ---------------------------------------------------------------------------
# Вычисления
# ---------------------------------------------------------------------------

def _monthly_revenue(orders: list) -> list[tuple[str, int, float]]:
    """Вернуть список (месяц-YYYY-MM, кол-во заказов, выручка) отсортированный по дате."""
    monthly: defaultdict[str, dict] = defaultdict(lambda: {"count": 0, "revenue": 0.0})
    for o in orders:
        try:
            if isinstance(o.date, datetime):
                dt = o.date
            else:
                raw = str(o.date)
                if "." in raw and len(raw) >= 8:
                    parts = raw.split(".")
                    dt = datetime(int(parts[2][:4]), int(parts[1]), int(parts[0]))
                else:
                    dt = datetime.fromisoformat(raw)
            key = dt.strftime("%Y-%m")
        except Exception:
            continue
        monthly[key]["count"] += 1
        monthly[key]["revenue"] += o.total or 0
    return [(k, v["count"], v["revenue"]) for k, v in sorted(monthly.items())]


def _top_products(orders: list, items_by_order: dict, top_n: int = 10) -> list[tuple[str, int, float]]:
    """Вернуть топ товаров: [(название, кол-во, выручка)]."""
    qty: Counter = Counter()
    revenue: defaultdict[str, float] = defaultdict(float)
    for order in orders:
        for item in items_by_order.get(order.id, []):
            name = item.product_name or f"Товар #{item.product_id}"
            qty[name] += item.quantity
            revenue[name] += item.total
    result = []
    for name, count in qty.most_common(top_n):
        result.append((name, count, revenue[name]))
    return result


def _abc_segments(orders: list) -> dict[str, list[tuple[str, float]]]:
    """ABC-сегментация клиентов по выручке.

    Returns:
        {"A": [(name, revenue), ...], "B": [...], "C": [...]}
    """
    client_revenue: defaultdict[str, float] = defaultdict(float)
    for o in orders:
        name = o.client_name or f"Клиент #{o.client_id}"
        client_revenue[name] += o.total or 0

    if not client_revenue:
        return {"A": [], "B": [], "C": []}

    total_rev = sum(client_revenue.values())
    sorted_clients = sorted(client_revenue.items(), key=lambda x: x[1], reverse=True)

    segments: dict[str, list] = {"A": [], "B": [], "C": []}
    cumulative = 0.0
    for name, rev in sorted_clients:
        cumulative += rev / total_rev * 100 if total_rev else 0
        seg = "A" if cumulative <= 80 else ("B" if cumulative <= 95 else "C")
        segments[seg].append((name, rev))

    return segments


def _low_stock_products(products: list, threshold: int = 5) -> list[tuple[str, int]]:
    """Товары с остатком ниже порога."""
    result = []
    for p in products:
        stock = p.stock or 0
        if stock <= threshold:
            result.append((p.name, int(stock)))
    return sorted(result, key=lambda x: x[1])


# ---------------------------------------------------------------------------
# Построение секций PDF
# ---------------------------------------------------------------------------

def _build_header(styles: dict, period: str, generated_at: str) -> list:
    elements = []
    elements.append(Paragraph("BEEBOT — Отчёт по продажам", styles["title"]))
    elements.append(Paragraph(
        f"Период: {_period_label(period)} &nbsp;&nbsp;|&nbsp;&nbsp; Сформирован: {generated_at}",
        styles["subtitle"],
    ))
    elements.append(Spacer(1, 0.3 * cm))
    return elements


def _build_summary_table(orders: list, styles: dict) -> list:
    """Сводные показатели."""
    elements = []
    elements.append(Paragraph("Сводка", styles["section"]))

    total_revenue = sum((o.total or 0) for o in orders)
    avg_order = total_revenue / len(orders) if orders else 0

    data = [
        ["Показатель", "Значение"],
        ["Всего заказов", str(len(orders))],
        ["Общая выручка", f"{total_revenue:,.0f} руб."],
        ["Средний чек", f"{avg_order:,.0f} руб."],
    ]
    col_widths = [10 * cm, 7 * cm]
    t = Table(data, colWidths=col_widths)
    t.setStyle(_BASE_TABLE_STYLE)
    elements.append(t)
    return elements


def _build_monthly_table(orders: list, styles: dict) -> list:
    """Таблица выручки по месяцам."""
    elements = []
    elements.append(Spacer(1, 0.3 * cm))
    elements.append(Paragraph("Выручка по месяцам", styles["section"]))

    monthly = _monthly_revenue(orders)
    if not monthly:
        elements.append(Paragraph("Нет данных по месяцам.", styles["body"]))
        return elements

    data = [["Месяц", "Заказов", "Выручка, руб."]]
    for ym, cnt, rev in monthly:
        parts = ym.split("-")
        month_label = f"{_RU_MONTHS.get(parts[1], parts[1])} {parts[0]}"
        data.append([month_label, str(cnt), f"{rev:,.0f}"])

    col_widths = [6 * cm, 4 * cm, 7 * cm]
    t = Table(data, colWidths=col_widths)
    t.setStyle(_BASE_TABLE_STYLE)
    elements.append(t)
    return elements


def _build_top_products_table(
    orders: list, items_by_order: dict, styles: dict, top_n: int = 10
) -> list:
    """Топ-10 товаров по выручке."""
    elements = []
    elements.append(Spacer(1, 0.3 * cm))
    elements.append(Paragraph(f"Топ-{top_n} товаров по выручке", styles["section"]))

    top = _top_products(orders, items_by_order, top_n)
    if not top:
        elements.append(Paragraph("Нет данных о позициях заказов.", styles["body"]))
        return elements

    data = [["#", "Товар", "Кол-во, шт.", "Выручка, руб."]]
    for i, (name, qty, rev) in enumerate(top, start=1):
        data.append([str(i), name, str(qty), f"{rev:,.0f}"])

    col_widths = [1 * cm, 8.5 * cm, 3 * cm, 4.5 * cm]
    t = Table(data, colWidths=col_widths)
    t.setStyle(_BASE_TABLE_STYLE)
    elements.append(t)
    return elements


def _build_abc_table(orders: list, styles: dict) -> list:
    """ABC-анализ клиентов."""
    elements = []
    elements.append(Spacer(1, 0.3 * cm))
    elements.append(Paragraph("ABC-анализ клиентов", styles["section"]))

    if not orders:
        elements.append(Paragraph("Нет данных для ABC-анализа.", styles["body"]))
        return elements

    segments = _abc_segments(orders)
    seg_meta = {
        "A": ("Ключевые", "до 80% выручки"),
        "B": ("Перспективные", "80-95% выручки"),
        "C": ("Разовые", "95-100% выручки"),
    }

    data = [["Сегмент", "Описание", "Клиентов", "Выручка, руб."]]
    for seg, (title, desc) in seg_meta.items():
        clients = segments.get(seg, [])
        seg_rev = sum(r for _, r in clients)
        data.append([
            f"{seg} — {title}",
            desc,
            str(len(clients)),
            f"{seg_rev:,.0f}",
        ])

    col_widths = [4.5 * cm, 4.5 * cm, 3 * cm, 5 * cm]
    t = Table(data, colWidths=col_widths)
    style = TableStyle(list(_BASE_TABLE_STYLE._cmds))
    # Подсветка строк по сегментам
    seg_colors = [
        colors.HexColor("#c6f6d5"),  # A — зелёный
        colors.HexColor("#fefcbf"),  # B — жёлтый
        colors.HexColor("#fed7d7"),  # C — красный
    ]
    for row_idx, bg in enumerate(seg_colors, start=1):
        style.add("BACKGROUND", (0, row_idx), (-1, row_idx), bg)
    t.setStyle(style)
    elements.append(t)
    return elements


def _build_low_stock_section(products: list, styles: dict) -> list:
    """Алерты о низком остатке."""
    elements = []
    elements.append(Spacer(1, 0.3 * cm))
    elements.append(Paragraph("Низкий остаток на складе", styles["section"]))

    low = _low_stock_products(products)
    if not low:
        elements.append(Paragraph("Все товары в достаточном количестве (> 5 шт.).", styles["body"]))
        return elements

    data = [["Товар", "Остаток, шт."]]
    for name, stock in low:
        data.append([name, str(stock)])

    col_widths = [12 * cm, 5 * cm]
    t = Table(data, colWidths=col_widths)
    style = TableStyle(list(_BASE_TABLE_STYLE._cmds))
    # Строки с нулевым остатком выделить красным
    for row_idx, (_, stock) in enumerate(low, start=1):
        if stock == 0:
            style.add("BACKGROUND", (0, row_idx), (-1, row_idx), colors.HexColor("#fed7d7"))
        elif stock <= 2:
            style.add("BACKGROUND", (0, row_idx), (-1, row_idx), colors.HexColor("#feebc8"))
    t.setStyle(style)
    elements.append(t)
    return elements


# ---------------------------------------------------------------------------
# Главная функция
# ---------------------------------------------------------------------------

def generate_sales_report(
    orders: list,
    items_by_order: dict,
    period: str,
    products: Optional[list] = None,
) -> bytes:
    """Сгенерировать PDF-отчёт по продажам.

    Args:
        orders: Список заказов (Order-объекты с полями id, date, total, client_name, client_id).
        items_by_order: Словарь {order_id: [OrderItem, ...]}.
        period: Строка периода: "30d", "90d", "365d", "week", "month", "all".
        products: Список продуктов (Product-объекты с полями name, stock). Опционально.

    Returns:
        Байты PDF-документа.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=_MARGIN,
        rightMargin=_MARGIN,
        topMargin=_MARGIN,
        bottomMargin=_MARGIN,
        title="BEEBOT — Отчёт по продажам",
        author="BEEBOT",
    )

    styles = _build_styles()
    generated_at = datetime.now().strftime("%d.%m.%Y %H:%M")

    elements: list = []
    elements.extend(_build_header(styles, period, generated_at))
    elements.extend(_build_summary_table(orders, styles))
    elements.extend(_build_monthly_table(orders, styles))
    elements.extend(_build_top_products_table(orders, items_by_order, styles))
    elements.extend(_build_abc_table(orders, styles))

    if products is not None:
        elements.extend(_build_low_stock_section(products, styles))

    doc.build(elements)
    return buf.getvalue()
