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

# Integram CRM
INTEGRAM_URL = os.getenv("INTEGRAM_URL")
INTEGRAM_LOGIN = os.getenv("INTEGRAM_LOGIN")
INTEGRAM_PASSWORD = os.getenv("INTEGRAM_PASSWORD")
INTEGRAM_DB = os.getenv("INTEGRAM_DB")

# Telegram ID пчеловода для уведомлений о новых заказах
_beekeeper_raw = os.getenv("BEEKEEPER_CHAT_ID")
BEEKEEPER_CHAT_ID: int | None = int(_beekeeper_raw) if _beekeeper_raw else None

# Telegram ID администратора для доступа к аналитике (агент «Аналитик»)
_admin_raw = os.getenv("ADMIN_CHAT_ID")
ADMIN_CHAT_ID: int | None = int(_admin_raw) if _admin_raw else BEEKEEPER_CHAT_ID

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
