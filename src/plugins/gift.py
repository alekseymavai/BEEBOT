"""GiftPlugin — Gift Protocol: CrmAgent + AnamnesisCache + GiftBroker + AgentSpecs.

Зависимости: crm, agents

Публикует в контейнере:
  "crm_agent"      → CrmAgent
  "anamnesis"      → AnamnesisCache
  "gift_broker"    → GiftBroker
  "agent_specs"    → AgentSpecsCache
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.kernel.plugin import Plugin

if TYPE_CHECKING:
    from src.kernel.container import Container

logger = logging.getLogger(__name__)


class GiftPlugin(Plugin):
    name = "gift"
    dependencies = ["crm", "agents"]

    async def setup(self, container: "Container") -> None:
        from src.crm_agent import CrmAgent
        from src.anamnesis import AnamnesisCache
        from src.gift_protocol import GiftBroker
        from src.agent_specs import AgentSpecsCache

        crm = container.get("crm")
        orchestrator = container.require("orchestrator")

        crm_agent = CrmAgent(crm)
        orchestrator.set_crm_agent(crm_agent)

        anamnesis = AnamnesisCache(orchestrator._memory)

        gift_broker = GiftBroker(
            orchestrator=orchestrator,
            context_store=orchestrator._shared_ctx,
            anamnesis=anamnesis,
            crm_agent=crm_agent,
        )

        container.set("crm_agent", crm_agent)
        container.set("anamnesis", anamnesis)
        container.set("gift_broker", gift_broker)

        # AgentSpecs
        agent_specs = AgentSpecsCache()
        try:
            await agent_specs.load()
        except Exception as e:
            logger.warning("AgentSpecs недоступны: %s", e)
        orchestrator.set_agent_specs(agent_specs)
        container.set("agent_specs", agent_specs)

        logger.info(
            "GiftPlugin: GiftBroker инициализирован (CrmAgent.available=%s).",
            crm_agent.available,
        )
