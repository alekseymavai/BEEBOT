"""Конфигурация DEVBOT — автономного разработчика BEEBOT."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Telegram
DEVBOT_TOKEN = os.getenv("DEVBOT_TOKEN", "")
DEVBOT_ADMIN_CHAT_ID: int | None = (
    int(os.getenv("DEVBOT_ADMIN_CHAT_ID", "0")) or None
)

# HTTP API (приём задач от BEEBOT)
DEVBOT_API_PORT = int(os.getenv("DEVBOT_API_PORT", "8091"))

# LLM (polza.ai — OpenAI-compatible прокси → Claude)
POLZA_API_KEY = os.getenv("POLZA_AI_API_KEY", "")
POLZA_BASE_URL = os.getenv("POLZA_BASE_URL", "https://api.polza.ai/api/v1")
POLZA_MODEL = os.getenv("POLZA_MODEL", "anthropic/claude-sonnet-4.6")
POLZA_REFERER = os.getenv("POLZA_REFERER", "https://dev.drondoc.ru")

# Пути
BEEBOT_DIR = Path(__file__).resolve().parent.parent.parent
MEMORY_DIR = Path.home() / ".claude" / "projects" / "-home-new-BEEBOT" / "memory"

# Integram (те же настройки что у BEEBOT)
INTEGRAM_URL = os.getenv("INTEGRAM_URL", "https://ai2o.ru")
INTEGRAM_DB = os.getenv("INTEGRAM_DB", "bibot")
INTEGRAM_LOGIN = os.getenv("INTEGRAM_LOGIN", "bibot")
INTEGRAM_PASSWORD = os.getenv("INTEGRAM_PASSWORD", "")

# ID таблиц DEVBOT (созданы в Integram 27.03.2026)
TABLE_DEV_ADVICE = int(os.getenv("TABLE_DEV_ADVICE", "7195"))    # Советы пчеловода
TABLE_DEV_TASKS = int(os.getenv("TABLE_DEV_TASKS", "7196"))      # Задачи разработки
TABLE_DEV_MEMORY = int(os.getenv("TABLE_DEV_MEMORY", "7197"))    # Память разработчика
