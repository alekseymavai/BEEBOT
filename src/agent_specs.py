"""AgentSpecsCache — спецификации агентов из Integram (Фаза 9.5).

Александр может менять system_prompt, skills, triggers прямо в Integram —
без деплоя. Бот перечитывает при старте или по команде /agent_config reload.

Если таблица AGENT_SPECS не создана в Integram — работает на in-code defaults.

Схема таблицы AGENT_SPECS (создать в Integram):
  Поля: agent_id (SHORT), system_prompt (MEMO), skills (MEMO),
        triggers (MEMO), voice_style (SHORT)
  → После создания таблицы заполнить TABLE_AGENT_SPECS в crm_constants.py.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-code defaults (используются если Integram таблица ещё не создана)
# ---------------------------------------------------------------------------

_DEFAULT_SPECS: dict[str, dict] = {
    "beebot": {
        "agent_id": "beebot",
        "system_prompt": None,   # None = использовать SYSTEM_PROMPT из llm_client.py
        "skills": ["consult", "recommend", "recipe"],
        "triggers": ["как принимать", "что помогает", "расскажи о"],
        "voice_style": "основатель",
    },
    "logist": {
        "agent_id": "logist",
        "system_prompt": None,
        "skills": ["create_order", "edit_order"],
        "triggers": ["заказ", "доставка", "адрес"],
        "voice_style": None,
    },
    "inspector": {
        "agent_id": "inspector",
        "system_prompt": None,
        "skills": ["diagnose_hive"],
        "triggers": ["осмотр улья", "диагностика", "осмотри"],
        "voice_style": "наставник",
    },
    "worker": {
        "agent_id": "worker",
        "system_prompt": None,
        "skills": ["assembly_queue", "checklist"],
        "triggers": [],
        "voice_style": None,
    },
}


# ---------------------------------------------------------------------------
# AgentSpecsCache
# ---------------------------------------------------------------------------

class AgentSpecsCache:
    """Кэш спецификаций агентов. Загружает из Integram, fallback на defaults."""

    def __init__(self) -> None:
        # agent_id → spec dict
        self._specs: dict[str, dict] = dict(_DEFAULT_SPECS)
        self._loaded_from_crm = False

    @property
    def loaded_from_crm(self) -> bool:
        return self._loaded_from_crm

    async def load(self) -> None:
        """Загрузить спецификации из Integram (если таблица создана).

        При отсутствии таблицы — тихо остаётся на defaults, логирует warning.
        """
        from src.crm_constants import TABLE_AGENT_SPECS
        if not TABLE_AGENT_SPECS:
            logger.info("AgentSpecsCache: TABLE_AGENT_SPECS не задана — используются defaults.")
            return

        from src.integram_api import IntegramAPI
        api = IntegramAPI()
        try:
            await api.authenticate()
            rows = await api.get_objects(TABLE_AGENT_SPECS)
            if not rows:
                logger.info("AgentSpecsCache: таблица AGENT_SPECS пуста — используются defaults.")
                return

            loaded = 0
            for row in rows:
                spec = self._parse_row(row)
                if spec and spec.get("agent_id"):
                    self._specs[spec["agent_id"]] = spec
                    loaded += 1
            self._loaded_from_crm = True
            logger.info("AgentSpecsCache: загружено %d спецификаций из Integram.", loaded)
        except Exception as e:
            logger.warning("AgentSpecsCache: не удалось загрузить из Integram: %s", e)

    @staticmethod
    def _parse_row(row: dict) -> Optional[dict]:
        """Преобразовать строку Integram в spec dict."""
        from src.crm_constants import (
            REQ_AGENT_ID, REQ_AGENT_SYSTEM_PROMPT,
            REQ_AGENT_SKILLS, REQ_AGENT_TRIGGERS, REQ_AGENT_VOICE_STYLE,
        )
        try:
            reqs = {str(r.get("id") or r.get("req_id", "")): r for r in (row.get("reqs") or [])}

            def _val(req_id: str) -> str:
                r = reqs.get(str(req_id))
                if not r:
                    return ""
                return str(r.get("value") or r.get("text") or "").strip()

            agent_id = _val(REQ_AGENT_ID)
            if not agent_id:
                return None
            return {
                "agent_id": agent_id,
                "system_prompt": _val(REQ_AGENT_SYSTEM_PROMPT) or None,
                "skills": [s.strip() for s in _val(REQ_AGENT_SKILLS).split(",") if s.strip()],
                "triggers": [t.strip() for t in _val(REQ_AGENT_TRIGGERS).split(",") if t.strip()],
                "voice_style": _val(REQ_AGENT_VOICE_STYLE) or None,
            }
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Read access
    # ------------------------------------------------------------------

    def get(self, agent_id: str) -> Optional[dict]:
        """Вернуть спецификацию агента (или None)."""
        return self._specs.get(agent_id)

    def get_system_prompt(self, agent_id: str) -> Optional[str]:
        """Вернуть system_prompt агента из Integram (None = использовать in-code default)."""
        spec = self._specs.get(agent_id, {})
        return spec.get("system_prompt") or None

    def get_triggers(self, agent_id: str) -> list[str]:
        """Вернуть list[str] триггеров для быстрой классификации."""
        spec = self._specs.get(agent_id, {})
        return spec.get("triggers") or []

    def set(self, agent_id: str, field: str, value: str) -> None:
        """Обновить поле спецификации в памяти (без записи в Integram)."""
        if agent_id not in self._specs:
            self._specs[agent_id] = {"agent_id": agent_id}
        if field == "skills":
            self._specs[agent_id][field] = [s.strip() for s in value.split(",") if s.strip()]
        elif field == "triggers":
            self._specs[agent_id][field] = [t.strip() for t in value.split(",") if t.strip()]
        else:
            self._specs[agent_id][field] = value or None

    async def update_crm(self, agent_id: str) -> bool:
        """Записать текущую спецификацию агента в Integram (если таблица создана)."""
        from src.crm_constants import TABLE_AGENT_SPECS
        if not TABLE_AGENT_SPECS:
            logger.warning("AgentSpecsCache.update_crm: TABLE_AGENT_SPECS не задана.")
            return False

        spec = self._specs.get(agent_id)
        if not spec:
            return False

        from src.crm_constants import (
            REQ_AGENT_ID, REQ_AGENT_SYSTEM_PROMPT,
            REQ_AGENT_SKILLS, REQ_AGENT_TRIGGERS, REQ_AGENT_VOICE_STYLE,
        )
        from src.integram_api import IntegramAPI
        api = IntegramAPI()
        try:
            await api.authenticate()
            reqs = {
                REQ_AGENT_ID: spec.get("agent_id", ""),
                REQ_AGENT_SYSTEM_PROMPT: spec.get("system_prompt") or "",
                REQ_AGENT_SKILLS: ", ".join(spec.get("skills") or []),
                REQ_AGENT_TRIGGERS: ", ".join(spec.get("triggers") or []),
                REQ_AGENT_VOICE_STYLE: spec.get("voice_style") or "",
            }
            # Найти существующую запись или создать новую
            rows = await api.get_objects(TABLE_AGENT_SPECS)
            existing_id = None
            for row in (rows or []):
                row_reqs = {str(r.get("id") or r.get("req_id", "")): r for r in (row.get("reqs") or [])}
                r = row_reqs.get(str(REQ_AGENT_ID))
                if r and (r.get("value") or r.get("text", "")).strip() == agent_id:
                    existing_id = row.get("id")
                    break

            if existing_id:
                await api.update_object(TABLE_AGENT_SPECS, existing_id, reqs)
            else:
                await api.create_object(TABLE_AGENT_SPECS, reqs)
            return True
        except Exception as e:
            logger.error("AgentSpecsCache.update_crm: ошибка записи: %s", e)
            return False

    def list_agents(self) -> list[str]:
        """Список всех agent_id в кэше."""
        return list(self._specs.keys())
