"""Фабрика CRM-клиента: v1 (ai2o.ru) или v2 (ai2o.online) по feature flag.

Единая точка входа для получения CRM-клиента.
Все потребители должны использовать get_crm_client() вместо прямого импорта.

Использование::

    from src.crm_factory import get_crm_client

    client = get_crm_client()
    await client.authenticate()
    products = await client.get_products()
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src import config

if TYPE_CHECKING:
    from src.integram_client import IntegramClient
    from src.integram_v2_client import IntegramV2Client

logger = logging.getLogger(__name__)

# Тип-алиас для клиента (v1 и v2 имеют одинаковый публичный интерфейс)
CrmClient = "IntegramClient | IntegramV2Client"


def get_crm_client() -> "IntegramClient | IntegramV2Client":
    """Создать CRM-клиент по текущему feature flag.

    INTEGRAM_V2=true  → IntegramV2Client (ai2o.online)
    INTEGRAM_V2=false → IntegramClient (ai2o.ru)
    """
    if config.INTEGRAM_V2:
        from src.integram_v2_client import IntegramV2Client
        logger.info("CRM: IntegramV2Client (ai2o.online)")
        return IntegramV2Client()
    else:
        from src.integram_client import IntegramClient
        logger.info("CRM: IntegramClient (ai2o.ru)")
        return IntegramClient()
