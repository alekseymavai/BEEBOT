"""KnowledgePlugin — база знаний FAISS + keyword-буст из CRM.

Зависимости: crm

Публикует в контейнере:
  "kb" → KnowledgeBase
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.kernel.plugin import Plugin

if TYPE_CHECKING:
    from src.kernel.container import Container

logger = logging.getLogger(__name__)


class KnowledgePlugin(Plugin):
    name = "knowledge"
    dependencies = ["crm"]

    async def setup(self, container: "Container") -> None:
        from src.knowledge_base import KnowledgeBase

        try:
            kb = KnowledgeBase()
            kb.load()
            logger.info("Knowledge base загружена: %d чанков.", len(kb.chunks))
        except FileNotFoundError:
            logger.error(
                "Knowledge base не найдена! Запустите `python -m src.build_kb`."
            )
            raise

        # Keyword-буст из CRM-каталога товаров
        crm = container.get("crm")
        if crm:
            try:
                products = await crm.get_products()
                names = [p.name for p in products if p.name]
                added = kb.update_keywords_from_products(names)
                if added:
                    logger.info(
                        "KB keyword-буст: +%d ключей из CRM (%d товаров).",
                        added,
                        len(names),
                    )
            except Exception as e:
                logger.warning("KB keyword-буст: не удалось обновить из CRM: %s", e)

        container.set("kb", kb)
