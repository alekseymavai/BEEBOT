"""Общие утилиты проекта BEEBOT.

Централизованные функции для парсинга дат, форматирования месяцев и т.д.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

# ---------------------------------------------------------------------------
# Русские названия месяцев (единый источник)
# ---------------------------------------------------------------------------

RU_MONTHS = {
    "01": "Январь", "02": "Февраль", "03": "Март",
    "04": "Апрель", "05": "Май", "06": "Июнь",
    "07": "Июль", "08": "Август", "09": "Сентябрь",
    "10": "Октябрь", "11": "Ноябрь", "12": "Декабрь",
}

# Для парсинга названий месяцев из текста (префикс → номер)
MONTH_PREFIX_MAP = {
    "январ": 1, "феврал": 2, "март": 3, "марта": 3,
    "апрел": 4, "ма": 5, "мая": 5, "май": 5,
    "июн": 6, "июл": 7, "август": 8,
    "сентябр": 9, "октябр": 10, "ноябр": 11, "декабр": 12,
}


def parse_date(val: object) -> Optional[datetime]:
    """Парсить дату из различных форматов.

    Поддерживает:
    - datetime объект (passthrough)
    - ISO 8601: "2026-04-02T12:00:00"
    - DD.MM.YYYY: "02.04.2026"
    - DD.MM.YYYY HH:MM:SS: "02.04.2026 12:00:00"
    - MM/DD/YYYY: "04/02/2026"

    Returns:
        datetime или None если не удалось распарсить.
    """
    if isinstance(val, datetime):
        return val
    if not val:
        return None

    s = str(val).strip()
    if not s:
        return None

    # ISO 8601
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        pass

    # DD.MM.YYYY [HH:MM[:SS]]
    if "." in s:
        try:
            parts = s.split()
            day, month, year = parts[0].split(".")
            hour = minute = second = 0
            if len(parts) > 1 and ":" in parts[1]:
                tp = parts[1].split(":")
                hour = int(tp[0])
                minute = int(tp[1]) if len(tp) > 1 else 0
                second = int(tp[2]) if len(tp) > 2 else 0
            return datetime(int(year), int(month), int(day), hour, minute, second)
        except (ValueError, IndexError):
            pass

    # MM/DD/YYYY
    if "/" in s:
        try:
            parts = s.split()
            month, day, year = parts[0].split("/")
            hour = minute = 0
            if len(parts) > 1 and ":" in parts[1]:
                tp = parts[1].split(":")
                hour, minute = int(tp[0]), int(tp[1]) if len(tp) > 1 else 0
            return datetime(int(year), int(month), int(day), hour, minute)
        except (ValueError, IndexError):
            pass

    return None


def format_month(month_key: str) -> str:
    """Форматировать ключ месяца 'MM.YYYY' или 'YYYY.MM' в русское название.

    Примеры:
        '04.2026' → 'Апрель 2026'
        '2026.04' → 'Апрель 2026'
    """
    parts = month_key.split(".")
    if len(parts) != 2:
        return month_key

    # Определить порядок: MM.YYYY или YYYY.MM
    if len(parts[0]) == 4:  # YYYY.MM
        year, mm = parts[0], parts[1]
    else:  # MM.YYYY
        mm, year = parts[0], parts[1]

    name = RU_MONTHS.get(mm.zfill(2), mm)
    return f"{name} {year}"
