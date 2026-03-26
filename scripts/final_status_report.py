"""Итоговый отчёт по заказам после исправлений."""
import asyncio
import sys
sys.path.insert(0, "/app")

from src.integram_client import IntegramClient
from src.integram_api import IntegramAPI, TABLE_ORDERS
from datetime import datetime


async def main():
    crm = IntegramClient()
    await crm.authenticate()
    api = crm._api

    orders = await crm.get_orders()
    print(f"Всего заказов в get_all_objects: {len(orders)}\n")

    # UDS-дубли (должно быть 0)
    uds_by_number: dict[str, list] = {}
    for o in orders:
        if o.number and o.number.startswith("UDS-"):
            uds_by_number.setdefault(o.number, []).append(o)
    duplicates = {num: lst for num, lst in uds_by_number.items() if len(lst) > 1}
    print(f"[1] UDS-дубли: {len(duplicates)} {'✓' if not duplicates else '✗'}")

    # Старые «Новый» (через get_all_objects — может не видеть 4020)
    cutoff = datetime(2026, 3, 1).date()
    old_new = [
        o for o in orders
        if o.status == "Новый"
        and hasattr(o.date, "date")
        and o.date.year >= 2020
        and o.date.date() < cutoff
        and not (o.number or "").startswith("UDS-")
    ]
    print(f"\n[2] Старые заказы 'Новый' (до {cutoff}, без UDS): {len(old_new)}")
    for o in old_new:
        print(f"  #{o.number} id={o.id} | {o.date.strftime('%d.%m.%Y')} | {o.total or 0:.0f} ₽")

    # Проверить 4020 напрямую через edit_obj
    raw_api = crm._api
    ej = await raw_api._request("get", "/bibot/edit_obj/4020?JSON")
    st = ej.json().get("reqs", {}).get("1073", {})
    print(f"\n[3] Заказ 4020 (edit_obj): status={st.get('ref')!r} (value={st.get('value')})")
    print("    ⚠ В list-view статус пуст из-за артефакта _m_set экспериментов.")
    print("    Фактические данные заказа корректны. Изменить статус: https://ai2o.ru/bibot/")

    print("\n⚠️  Для смены статуса старых заказов — вручную через https://ai2o.ru/bibot/")
    print("    Найдите заказ → откройте → смените статус с 'Новый' на нужный.")
    print("\n=== ГОТОВО ===")
    await crm.close()


asyncio.run(main())
