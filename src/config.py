import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# Telegram
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
# SOCKS5 прокси для обхода блокировки api.telegram.org (если VPS не может достучаться)
# Настроить совместно с tg-socks.service + groq-tunnel.service на hive
TG_SOCKS_PROXY = os.getenv("TG_SOCKS_PROXY")  # e.g. socks5://localhost:9150

# Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL")  # Optional proxy URL

# Embeddings
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)

# Paths
DATA_DIR = BASE_DIR / "data"
PDFS_DIR = DATA_DIR / "pdfs"
SUBTITLES_DIR = DATA_DIR / "subtitles"
COMMENTS_DIR = DATA_DIR / "comments"
TEXTS_DIR = DATA_DIR / "texts"
PROCESSED_DIR = DATA_DIR / "processed"
FAISS_INDEX_PATH = PROCESSED_DIR / "index.faiss"
CHUNKS_PATH = PROCESSED_DIR / "chunks.json"

# Integram CRM (v1 — ai2o.ru, read-only архив)
INTEGRAM_URL = os.getenv("INTEGRAM_URL")
INTEGRAM_LOGIN = os.getenv("INTEGRAM_LOGIN")
INTEGRAM_PASSWORD = os.getenv("INTEGRAM_PASSWORD")
INTEGRAM_DB = os.getenv("INTEGRAM_DB")

# Integram CRM v2 (ai2o.online — основная)
INTEGRAM_V2 = os.getenv("INTEGRAM_V2", "").lower() in ("true", "1", "yes")
INTEGRAM_V2_URL = os.getenv("INTEGRAM_V2_URL", "https://ai2o.online")
INTEGRAM_V2_EMAIL = os.getenv("INTEGRAM_V2_EMAIL", "")
INTEGRAM_V2_PASSWORD = os.getenv("INTEGRAM_V2_PASSWORD", "")
INTEGRAM_V2_WORKSPACE = os.getenv("INTEGRAM_V2_WORKSPACE", "alekseymavai")

# Группы где бот отвечает на все сообщения без @упоминания
# Несколько ID через запятую: ACTIVE_GROUP_IDS=-100123,-100456
_groups_raw = os.getenv("ACTIVE_GROUP_IDS", "")
ACTIVE_GROUP_IDS: frozenset[int] = frozenset(
    int(x.strip()) for x in _groups_raw.split(",") if x.strip().lstrip("-").isdigit()
)

# Telegram ID пчеловода для уведомлений о новых заказах
_beekeeper_raw = os.getenv("BEEKEEPER_CHAT_ID")
BEEKEEPER_CHAT_ID: int | None = int(_beekeeper_raw) if _beekeeper_raw else None

# Telegram ID администратора(ов) для доступа к аналитике и /admin командам
# Поддерживает несколько ID через запятую: ADMIN_CHAT_ID=123456,789012
_admin_raw = os.getenv("ADMIN_CHAT_ID")
ADMIN_CHAT_ID: int | None = None
ADMIN_IDS: frozenset[int] = frozenset()
if _admin_raw:
    _parsed = [int(x.strip()) for x in _admin_raw.split(",") if x.strip().isdigit()]
    ADMIN_IDS = frozenset(_parsed)
    ADMIN_CHAT_ID = _parsed[0] if _parsed else None
elif BEEKEEPER_CHAT_ID:
    ADMIN_IDS = frozenset({BEEKEEPER_CHAT_ID})
    ADMIN_CHAT_ID = BEEKEEPER_CHAT_ID

# UDS (система лояльности)
UDS_API_KEY = os.getenv("UDS_API_KEY")
UDS_COMPANY_ID = os.getenv("UDS_COMPANY_ID")
UDS_BASE_URL = os.getenv("UDS_BASE_URL")  # Optional proxy URL

# Delivery (СДЭК + Почта России)
CDEK_CLIENT_ID = os.getenv("CDEK_CLIENT_ID")
CDEK_CLIENT_SECRET = os.getenv("CDEK_CLIENT_SECRET")
POCHTA_TOKEN = os.getenv("POCHTA_TOKEN")
POCHTA_KEY = os.getenv("POCHTA_KEY")
SENDER_CITY = os.getenv("SENDER_CITY", "Москва")

# YouTube Data API (опционально — для автообновления KB)
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
YOUTUBE_CHANNEL_HANDLE = os.getenv("YOUTUBE_CHANNEL_HANDLE", "@a.dmitrov")

# Bot settings
MAX_CONTEXT_CHUNKS = int(os.getenv("MAX_CONTEXT_CHUNKS", "5"))
MAX_RESPONSE_LENGTH = int(os.getenv("MAX_RESPONSE_LENGTH", "4096"))

# Долгосрочная память (SQLite)
MEMORY_DB_PATH = DATA_DIR / "memory.db"

# DEVBOT — автономный разработчик (на hive, доступен через SSH-туннель)
# Туннель: groq-tunnel -R 8091:localhost:8091 пробрасывает VPS:8091 → hive:8091
DEVBOT_API_URL = os.getenv("DEVBOT_API_URL", "http://localhost:8091")
DEVBOT_API_KEY = os.getenv("DEVBOT_API_KEY", "")  # Bearer-токен (12.3)

# Яндекс Диск — OAuth-токен для резервного копирования (Фаза 12.2)
# Получить: https://oauth.yandex.ru → приложение с правом cloud_api:disk.write
YADISK_TOKEN = os.getenv("YADISK_TOKEN")

# Redis (event bus между ботом и бэкендом)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# AgentBus — dronedoc2026 AgentBus (опционально, для мульти-агентной экосистемы)
# При наличии: BEEBOT регистрируется в шине и экспортирует инструменты (kb_search, order_status, ask)
AGENT_BUS_URL = os.getenv("AGENT_BUS_URL")

# Telegram ID работников склада (сборка заказов)
# Несколько ID через запятую: WORKER_CHAT_IDS=123456789,987654321
_workers_raw = os.getenv("WORKER_CHAT_IDS", "")
WORKER_CHAT_IDS: frozenset[int] = frozenset(
    int(x.strip()) for x in _workers_raw.split(",") if x.strip().isdigit()
)
