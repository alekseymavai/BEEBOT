"""Единые константы Integram CRM — ID таблиц, реквизитов и lookup-справочники.

Единый источник истины для всего кода, работающего с CRM.
Импортировать отсюда, не из integram_api.py.
"""

# ---------------------------------------------------------------------------
# ID таблиц
# ---------------------------------------------------------------------------

TABLE_ORDERS = 1024
TABLE_CLIENTS = 1023
TABLE_PRODUCTS = 1022
TABLE_ORDER_ITEMS = 1025
TABLE_CATEGORIES = 1018
TABLE_SOURCES = 1019
TABLE_STATUSES = 1020
TABLE_DELIVERY = 1021

# Онтологические и операционные таблицы
TABLE_USERS = 4964
TABLE_STATUS_HISTORY = 6156
TABLE_HEALTH_PROFILE = 6165
TABLE_BATCHES = 6404          # Партии отправки

# DEVBOT таблицы (созданы 27.03.2026)
TABLE_DEV_ADVICE = 7195       # Советы пчеловода
TABLE_DEV_TASKS = 7196        # Задачи разработки
TABLE_DEV_MEMORY = 7197       # Память разработчика

# AGENT_SPECS таблица (Фаза 9.5 — нужно создать в Integram UI, затем заполнить ID)
# Поля: agent_id (SHORT), system_prompt (MEMO), skills (MEMO),
#        triggers (MEMO), voice_style (SHORT)
TABLE_AGENT_SPECS = None      # TODO: заполнить после создания таблицы в Integram

# Реквизиты AGENT_SPECS (заполнить после создания таблицы)
REQ_AGENT_ID = None
REQ_AGENT_SYSTEM_PROMPT = None
REQ_AGENT_SKILLS = None
REQ_AGENT_TRIGGERS = None
REQ_AGENT_VOICE_STYLE = None

# Реквизиты «Советы пчеловода» (7195)
REQ_ADVICE_TEXT = "7199"      # MEMO — текст совета
REQ_ADVICE_CATEGORY = "7201"  # SHORT — категория (клиент/crm/продукт/процесс)
REQ_ADVICE_PRIORITY = "7203"  # SHORT — приоритет (высокий/средний/справочный)
REQ_ADVICE_STATUS = "7205"    # SHORT — статус (активен/архив)

# Реквизиты «Задачи разработки» (7196)
REQ_TASK_DESC = "7207"        # MEMO — описание задачи
REQ_TASK_STATUS = "7209"      # SHORT — статус (новая/анализ/выполняется/готово/ошибка)
REQ_TASK_PRIORITY = "7211"    # SHORT — приоритет (срочно/обычный/когда-нибудь)
REQ_TASK_FILES = "7213"       # CHARS — файлы затронуты
REQ_TASK_PR = "7215"          # CHARS — PR-ссылка
REQ_TASK_COMMIT = "7217"      # SHORT — SHA коммита
REQ_TASK_LESSONS = "7219"     # MEMO — уроки

# Реквизиты «Память разработчика» (7197)
REQ_MEM_CONTEXT = "7221"      # MEMO — почему Александр попросил
REQ_MEM_SOLUTION = "7223"     # MEMO — что именно сделано
REQ_MEM_FILES = "7225"        # CHARS — список изменённых файлов
REQ_MEM_PR = "7227"           # SHORT — ссылка на PR
REQ_MEM_ANTIPATTERN = "7229"  # MEMO — что НЕ делать
REQ_MEM_CATEGORY = "7231"     # SHORT — категория (модель/api/frontend/kb/infra/crm)

# ---------------------------------------------------------------------------
# Реквизиты заказов
# ---------------------------------------------------------------------------

REQ_ORDER_DATE = "1048"
REQ_ORDER_ADDRESS = "1050"
REQ_ORDER_DELIVERY_COST = "1052"
REQ_ORDER_ITEMS_TOTAL = "1054"
REQ_ORDER_TOTAL = "1056"
REQ_ORDER_TRACKING = "1058"
REQ_ORDER_COMMENT = "1059"
REQ_ORDER_CLIENT = "1071"
REQ_ORDER_STATUS = "1073"
REQ_ORDER_DELIVERY_METHOD = "1075"
REQ_ORDER_SOURCE = "1076"
REQ_ORDER_MESSENGER = "1388"
REQ_ORDER_SHIPPED_DATE = "6360"    # Дата отправки
REQ_ORDER_DELIVERED_DATE = "6362"  # Дата доставки
REQ_ORDER_BATCH = "6414"           # Партия отправки (REF → TABLE_BATCHES)
REQ_ORDER_CDEK_CONFIRMED = "6416"  # Адрес СДЭК уточнён (bool)
REQ_ORDER_CLIENT_NOTIFIED = "6418" # Клиент оповещён (bool)
REQ_ORDER_STOCK_CHECKED = "6420"   # Наличие проверено (bool)

# История статусов
REQ_HISTORY_ORDER = "6157"
REQ_HISTORY_STATUS_FROM = "6158"
REQ_HISTORY_STATUS_TO = "6160"
REQ_HISTORY_DATE = "6162"
REQ_HISTORY_COMMENT = "6164"

# Профиль здоровья клиента
REQ_HEALTH_CLIENT = "6166"   # REF → Клиент
REQ_HEALTH_SYMPTOM = "6167"  # REF → Симптомы (опционально)
REQ_HEALTH_SOURCE = "6169"   # Текст факта (источник данных)
REQ_HEALTH_DATE = "6171"     # Дата записи

# Партии отправки
REQ_BATCH_DATE = "6407"        # Дата партии (DATETIME)
REQ_BATCH_DELIVERY = "6409"    # Способ доставки (SHORT — текст)
REQ_BATCH_COUNT = "6411"       # Кол-во заказов (NUMBER)
REQ_BATCH_NOTE = "6413"        # Примечание (CHARS)

# ---------------------------------------------------------------------------
# Реквизиты клиентов
# ---------------------------------------------------------------------------

REQ_CLIENT_PHONE = "1036"
REQ_CLIENT_TG_ID = "1038"
REQ_CLIENT_TG_USER = "1040"
REQ_CLIENT_ADDRESS = "1042"
REQ_CLIENT_CITY = "1044"
REQ_CLIENT_COMMENT = "1046"
REQ_CLIENT_SOURCE = "1069"

# ---------------------------------------------------------------------------
# Реквизиты позиций заказа
# ---------------------------------------------------------------------------

REQ_ITEM_QTY = "1061"
REQ_ITEM_PRICE = "1063"
REQ_ITEM_SUM = "1065"
REQ_ITEM_PRODUCT = "1078"
REQ_ITEM_ORDER = "1154"

# ---------------------------------------------------------------------------
# Реквизиты товаров
# ---------------------------------------------------------------------------

REQ_PRODUCT_PRICE = "1027"
REQ_PRODUCT_WEIGHT = "1029"
REQ_PRODUCT_DESC = "1031"
REQ_PRODUCT_INSTOCK = "1033"
REQ_PRODUCT_SKU = "1035"
REQ_PRODUCT_CATEGORY = "1067"
REQ_PRODUCT_SHORT = "1173"
REQ_PRODUCT_STOCK = "4850"

# ---------------------------------------------------------------------------
# Реквизиты пользователей веб-панели
# ---------------------------------------------------------------------------

REQ_USER_PASSWORD_HASH = "4966"
REQ_USER_ROLE_OLD = "4967"
REQ_USER_ROLE = "4983"
REQ_USER_DISPLAY_NAME = "4969"
REQ_USER_ACTIVE = "4971"

# ---------------------------------------------------------------------------
# Lookup-справочники (ID записей)
# ---------------------------------------------------------------------------

# Статусы заказов
STATUS_IDS = {
    "Новый": "1086",
    "Подтверждён": "1087",
    "В сборке": "1088",
    "Отправлен": "1089",
    "Доставлен": "1090",
    "Отменён": "1091",
}

# Способы доставки
DELIVERY_IDS = {
    "СДЭК": "1092",
    "Почта России": "1093",
    "Самовывоз": "1094",
}

# Источники заказов
SOURCE_IDS = {
    "Telegram": "1082",
    "UDS": "1083",
    "WhatsApp": "1084",
    "Ручной ввод": "1085",
    "ВК": "3176",
    "Instagram": "3177",
}

# Категории товаров
CATEGORY_IDS = {
    "Продукты пчеловодства": "1079",
    "Настойки": "1080",
    "Программы здоровья": "1081",
    "Мёд": "1167",
    "Наборы": "1168",
    "Упаковка": "1169",
    "Свечи": "1170",
    "Чаи и травы": "1171",
}

# Списки для справочников
ORDER_STATUSES = list(STATUS_IDS.keys())
DELIVERY_METHODS = list(DELIVERY_IDS.keys())
