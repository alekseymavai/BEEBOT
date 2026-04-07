"""AgentsPlugin — все LangGraph-агенты и оркестратор.

Зависимости: crm, knowledge

Публикует в контейнере:
  "orchestrator"     → Orchestrator
  "inspector"        → InspectorAgent
  "logist"           → LogistAgent
  "analyst"          → AnalystAgent
  "admin_chat_agent" → AdminChatAgent
  "consult_service"  → ConsultService
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.kernel.plugin import Plugin

if TYPE_CHECKING:
    from src.kernel.container import Container

logger = logging.getLogger(__name__)


class AgentsPlugin(Plugin):
    name = "agents"
    dependencies = ["crm", "knowledge"]

    async def setup(self, container: "Container") -> None:
        from src.config import BEEKEEPER_CHAT_ID
        from src.orchestrator import Orchestrator
        from src.agents.inspector import InspectorAgent
        from src.agents.logist import LogistAgent
        from src.agents.analyst import AnalystAgent
        from src.agents.admin_chat import AdminChatAgent
        from src.services.consult_service import ConsultService

        kb = container.require("kb")
        crm = container.get("crm")

        # --- Orchestrator (содержит LLM-клиент, память, KB-агент) ---
        orchestrator = Orchestrator()
        orchestrator.set_kb(kb)

        # Онтология (optional)
        try:
            await orchestrator.load_ontology()
        except Exception as e:
            logger.warning("Онтология недоступна: %s", e)

        container.set("orchestrator", orchestrator)

        # --- Агенты ---
        inspector = InspectorAgent()
        inspector.kb = kb

        logist = LogistAgent(beekeeper_chat_id=BEEKEEPER_CHAT_ID)

        analyst = AnalystAgent(
            groq_client=orchestrator._groq,
            groq_model=orchestrator._model,
        )

        admin_chat_agent = AdminChatAgent(
            groq_client=orchestrator._groq,
            model=orchestrator._model,
        )

        # Подключить CRM к агентам
        if crm:
            logist.set_crm(crm)
            analyst.set_crm(crm)
            admin_chat_agent.set_crm(crm)

        container.set("inspector", inspector)
        container.set("logist", logist)
        container.set("analyst", analyst)
        container.set("admin_chat_agent", admin_chat_agent)

        # --- ConsultService (тонкая обёртка KB + LLM) ---
        consult_service = ConsultService(
            kb=kb,
            llm=orchestrator._groq,
        )
        container.set("consult_service", consult_service)

        logger.info("AgentsPlugin: оркестратор + 4 агента инициализированы.")
