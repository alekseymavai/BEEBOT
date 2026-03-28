"""Онтология продуктов и симптомов из Integram.

Загружает Симптомы и Показания к применению один раз при старте,
кэширует в памяти. Используется в _node_beebot оркестратора:
если пользователь упомянул симптом → добавить рекомендации в LLM-контекст.

Таблицы Integram (созданы в bibot):
  Симптомы         (6137): Категория (6139), Ключевые слова (6141)
  Показания (6143): Товар-REF (6144), Симптом-REF (6145),
                    Рекомендация (6147), Примечание (6149)
"""

from __future__ import annotations

import logging
from typing import Optional

from src.crm_constants import (
    TABLE_DEV_ADVICE,
    REQ_ADVICE_TEXT, REQ_ADVICE_CATEGORY, REQ_ADVICE_PRIORITY, REQ_ADVICE_STATUS,
)
from src.integram_api import IntegramAPI, _extract_ref_text, _extract_ref_id, _strip_html

logger = logging.getLogger(__name__)

# IDs таблиц и полей (созданы через MySQL)
_TABLE_SYMPTOMS = 6137
_TABLE_INDICATIONS = 6143

_ATTACH_SYM_KEYWORDS = "6141"    # Ключевые слова симптома
_ATTACH_IND_PRODUCT = "6144"     # Товар REF
_ATTACH_IND_SYMPTOM = "6145"     # Симптом REF
_ATTACH_IND_RECTYPE = "6147"     # Тип рекомендации (Основное/Вспомогательное/С осторожностью)
_ATTACH_IND_NOTE = "6149"        # Примечание с дозировкой


class OntologyCache:
    """Кэш онтологических связей Симптом → Продукт из Integram."""

    def __init__(self) -> None:
        # [{id, name, keywords: [str]}]
        self._symptoms: list[dict] = []
        # [{product_name, symptom_id, rec_type, note}]
        self._indications: list[dict] = []
        # [{name, text, category, priority, status}]
        self._advice: list[dict] = []
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    async def load(self) -> None:
        """Загрузить данные из Integram. Вызывать при старте бота."""
        api = IntegramAPI()
        try:
            await api.authenticate()
            await self._load_symptoms(api)
            await self._load_indications(api)
            await self._load_advice(api)
            self._loaded = True
            logger.info(
                "Онтология загружена: %d симптомов, %d показаний, %d советов",
                len(self._symptoms), len(self._indications), len(self._advice),
            )
        except Exception as exc:
            logger.warning("Не удалось загрузить онтологию из Integram: %s", exc)
        finally:
            await api.close()

    async def _load_symptoms(self, api: IntegramAPI) -> None:
        objects = await api.get_all_objects(_TABLE_SYMPTOMS)
        for obj in objects:
            name = obj.get("val", "").strip()
            if not name:
                continue  # пропуск global REF-записи (пустое имя)
            keywords_raw = _strip_html(obj["reqs"].get(_ATTACH_SYM_KEYWORDS, ""))
            keywords = [k.strip().lower() for k in keywords_raw.split(",") if k.strip()]
            if keywords:
                self._symptoms.append({
                    "id": obj["id"],
                    "name": name,
                    "keywords": keywords,
                })

    async def _load_indications(self, api: IntegramAPI) -> None:
        objects = await api.get_all_objects(_TABLE_INDICATIONS)
        for obj in objects:
            product_html = obj["reqs"].get(_ATTACH_IND_PRODUCT, "")
            symptom_html = obj["reqs"].get(_ATTACH_IND_SYMPTOM, "")
            product_name = _extract_ref_text(product_html)
            symptom_id = _extract_ref_id(symptom_html)
            rec_type = _strip_html(obj["reqs"].get(_ATTACH_IND_RECTYPE, ""))
            note = _strip_html(obj["reqs"].get(_ATTACH_IND_NOTE, ""))

            if product_name and symptom_id:
                self._indications.append({
                    "product_name": product_name,
                    "symptom_id": symptom_id,
                    "rec_type": rec_type,
                    "note": note,
                })

    async def _load_advice(self, api: IntegramAPI) -> None:
        """Загрузить советы пчеловода из таблицы TABLE_DEV_ADVICE (7195)."""
        if not TABLE_DEV_ADVICE:
            return
        objects = await api.get_all_objects(TABLE_DEV_ADVICE)
        for obj in objects:
            name = obj.get("val", "").strip()
            if not name:
                continue
            reqs = obj.get("reqs", {})
            status = _strip_html(reqs.get(REQ_ADVICE_STATUS, "")).strip().lower()
            if status == "архив":
                continue
            self._advice.append({
                "name": name,
                "text": _strip_html(reqs.get(REQ_ADVICE_TEXT, "")).strip(),
                "category": _strip_html(reqs.get(REQ_ADVICE_CATEGORY, "")).strip(),
                "priority": _strip_html(reqs.get(REQ_ADVICE_PRIORITY, "")).strip().lower(),
                "status": status,
            })

    @property
    def advice_items(self) -> list[dict]:
        """Список советов (для /advice команды)."""
        return list(self._advice)

    def get_advice_prompt(self) -> str:
        """Сформировать блок советов для инжекции в системный промпт консультанта.

        Возвращает пустую строку если советов нет.
        """
        if not self._advice:
            return ""
        _priority_order = {"высокий": 0, "средний": 1, "справочный": 2}
        sorted_advice = sorted(
            self._advice,
            key=lambda a: _priority_order.get(a["priority"], 9),
        )
        lines = []
        for item in sorted_advice:
            text = item["text"] or item["name"]
            lines.append(f"- {text}")
        return "\n".join(lines)

    def match(self, text: str) -> Optional[str]:
        """Найти симптомы в тексте и вернуть подсказку для LLM-промпта.

        Returns:
            Строка вида "Симптом: Язва. Рекомендовано: Прополис (осн.) — 20 капель..."
            или None если симптомов не обнаружено.
        """
        if not self._loaded or not self._symptoms:
            return None

        text_lower = text.lower()
        matched_ids: set[int] = set()
        matched_names: list[str] = []

        for sym in self._symptoms:
            for kw in sym["keywords"]:
                if kw in text_lower:
                    if sym["id"] not in matched_ids:
                        matched_ids.add(sym["id"])
                        matched_names.append(sym["name"])
                    break

        if not matched_ids:
            return None

        recs = [
            ind for ind in self._indications
            if ind["symptom_id"] in matched_ids
        ]
        if not recs:
            return None

        # Форматируем компактно — вся рекомендация в одну строку для memory_facts
        sym_str = ", ".join(matched_names)
        rec_parts = []
        for rec in recs:
            part = rec["product_name"]
            if rec["rec_type"] and rec["rec_type"] not in ("", "—"):
                part += f" ({rec['rec_type'].lower()})"
            if rec["note"]:
                part += f" — {rec['note']}"
            rec_parts.append(part)

        return f"Симптом пользователя: {sym_str}. Рекомендовано: {'; '.join(rec_parts)}"
