"""
Скрипт исправления заказов:
1. Удаляет дублирующиеся UDS-заказы (оставляет с меньшим ID)
   - Сначала удаляет позиции заказа, затем сам заказ
2. Показывает старые "Новый" заказы (статус нельзя изменить через API)

Запуск:
  python scripts/fix_orders.py          # dry-run (только отчёт)
  python scripts/fix_orders.py --apply  # применить изменения
"""
import asyncio
import sys
from datetime import datetime

sys.path.insert(0, "/app")

DRY_RUN = "--apply" not in sys.argv

from src.integram_client import IntegramClient
from src.integram_api import IntegramAPI


async def delete_order_with_items(api: IntegramAPI, order_id: int, dry_run: bool) -> bool:
    """Удалить заказ: сначала позиции, затем сам заказ."""
    http = await api._get_http()

    # Получить позиции
    items = await api.get_order_items(order_id)
    print(f"    Позиций заказа: {len(items)}")

    if not dry_run:
        for item in items:
            resp = await http.post(
                f"/bibot/_m_del/{item['id']}?JSON",
                data={"_xsrf": api._xsrf or ""},
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            data = resp.json()
            if isinstance(data, list) and data and "error" in data[0]:
                print(f"    ✗ Не удалось удалить позицию {item['id']}: {data[0]['error']}")
            else:
                print(f"    ✓ Удалена позиция {item['id']}")

        # Удалить заказ
        resp2 = await http.post(
            f"/bibot/_m_del/{order_id}?JSON",
            data={"_xsrf": api._xsrf or ""},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        data2 = resp2.json()
        if isinstance(data2, list) and data2 and "error" in data2[0]:
            print(f"    ✗ Не удалось удалить заказ {order_id}: {data2[0]['error']}")
            return False
        print(f"    ✓ Удалён заказ id={order_id}")
        return True
    return True


async def main():
    mode = "DRY-RUN" if DRY_RUN else "APPLY"
    print(f"=== fix_orders.py [{mode}] ===\n")

    crm = IntegramClient()
    await crm.authenticate()
    api = crm._api

    orders = await crm.get_orders()
    print(f"Всего заказов: {len(orders)}")

    # ── 1. Дубли UDS ─────────────────────────────────────────────────────────
    uds_by_number: dict[str, list] = {}
    for o in orders:
        if o.number and o.number.startswith("UDS-"):
            uds_by_number.setdefault(o.number, []).append(o)

    duplicates = {num: lst for num, lst in uds_by_number.items() if len(lst) > 1}
    print(f"\n[1] Дублей UDS-заказов: {len(duplicates)}")

    deleted_ids: set[int] = set()
    for num, lst in duplicates.items():
        lst.sort(key=lambda x: x.id)
        keep = lst[0]
        to_delete = lst[1:]
        print(f"  {num}: оставляем id={keep.id} | удаляем {[o.id for o in to_delete]}")
        for o in to_delete:
            success = await delete_order_with_items(api, o.id, DRY_RUN)
            if success:
                deleted_ids.add(o.id)

    print(f"  Итого {'к удалению' if DRY_RUN else 'удалено'}: {len(deleted_ids)}")

    # ── 2. Старые "Новый" ─────────────────────────────────────────────────────
    cutoff = datetime(2026, 3, 1).date()
    old_new = [
        o for o in orders
        if o.status == "Новый"
        and o.id not in deleted_ids
        and hasattr(o.date, "date")
        and o.date.year >= 2020
        and o.date.date() < cutoff
        and not (o.number or "").startswith("UDS-")
    ]

    print(f"\n[2] Старых заказов 'Новый' (до {cutoff}, без UDS): {len(old_new)}")
    for o in old_new:
        date_str = o.date.strftime("%d.%m.%Y")
        print(f"  #{o.number} id={o.id} | {date_str} | {o.source or '—'} | {o.total or 0:.0f} ₽")

    print("""
⚠️  Статус заказов нельзя изменить через API (_m_save не обновляет reference-поля).
   Измените статус вручную через веб-панель Integram: https://ai2o.ru/bibot/
""")

    print("=== ГОТОВО ===")
    await crm.close()


asyncio.run(main())
