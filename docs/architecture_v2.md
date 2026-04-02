# BEEBOT — Архитектура v2 (CRM-миграция)

> **Версия:** 2 апреля 2026
> **Ключевое изменение:** переход с ai2o.ru (Integram v1) на ai2o.online (Integram v2)

---

## 1. Общая картина: два CRM-контура

```mermaid
graph TB
    subgraph BEEBOT["BEEBOT — Telegram-помощник пчеловода"]

        subgraph TRANSPORT["Транспорт"]
            TG["Telegram-бот<br/>(aiogram 3)"]
            WEB["Веб-панель<br/>(FastAPI + Vue 3)"]
        end

        subgraph CORE["Бизнес-логика"]
            ORCH["Оркестратор<br/>(LangGraph)"]
            SVC_O["OrderService"]
            SVC_C["ConsultService"]
            SVC_N["NotificationService"]
        end

        subgraph ADAPTERS["CRM-адаптеры"]
            V1["IntegramClient<br/>(v1, ai2o.ru)"]
            V2["IntegramV2Client<br/>(v2, ai2o.online)"]
            FF{{"INTEGRAM_V2<br/>feature flag"}}
        end
    end

    subgraph EXTERNAL["Внешние системы"]
        CRM_OLD[("ai2o.ru/bibot<br/>🗄️ Архив<br/>1924 клиента<br/>1915 заказов")]
        CRM_NEW[("ai2o.online<br/>📦 Основная<br/>85 товаров<br/>чистые данные")]
        LLM["Groq API<br/>(llama-3.3-70b)"]
        FAISS["FAISS<br/>(276 чанков)"]
    end

    TG --> ORCH --> SVC_C
    TG --> SVC_O
    WEB --> SVC_O

    SVC_O --> FF
    SVC_C --> LLM & FAISS
    SVC_N --> TG

    FF -->|"true"| V2
    FF -->|"false"| V1

    V1 -->|"read-only"| CRM_OLD
    V2 -->|"read+write"| CRM_NEW

    style CRM_OLD fill:#fee2e2,stroke:#ef4444
    style CRM_NEW fill:#bbf7d0,stroke:#22c55e
    style FF fill:#fef3c7,stroke:#f59e0b
```

---

## 2. CRM v2: схема таблиц и связей

```mermaid
erDiagram
    CATEGORIES["Категории товаров (151)"] {
        int id PK
        string name "8 категорий"
    }

    SOURCES["Источники (15)"] {
        int id PK
        string name "Telegram, UDS, WhatsApp..."
    }

    STATUSES["Статусы заказов (152)"] {
        int id PK
        string name "Новый → Доставлен"
    }

    DELIVERY_METHODS["Способы доставки (150)"] {
        int id PK
        string name "СДЭК, Почта, Самовывоз"
    }

    PRODUCTS["Товары (581)"] {
        int id PK
        string name "Название"
        int category_id FK "→ Категории"
        float price "Цена"
        float weight "Вес, г"
        text description "Описание"
        bool in_stock "В наличии"
        string sku_uds "Артикул UDS"
        string short_name "Короткое"
        int stock "Остаток"
    }

    CLIENTS["Клиенты (52)"] {
        int id PK
        string full_name "ФИО"
        string phone "Телефон"
        int telegram_id "Telegram ID"
        string telegram_user "Username"
        text address "Адрес"
        string city "Город"
        text comment "Комментарий"
        int source_id FK "→ Источники"
    }

    ORDERS["Заказы (60)"] {
        int id PK
        datetime date "Дата"
        int client_id FK "→ Клиенты"
        int status_id FK "→ Статусы"
        int delivery_id FK "→ Способы доставки"
        int source_id FK "→ Источники"
        text address "Адрес доставки"
        float delivery_cost "Стоимость доставки"
        float items_total "Сумма товаров"
        float total "Итого"
        string tracking "Трек-номер"
        text comment "Комментарий"
        string messenger "Мессенджер"
        datetime shipped_date "Дата отправки"
        datetime delivered_date "Дата доставки"
    }

    ORDER_ITEMS["Позиции заказа (78)"] {
        int id PK
        int order_id FK "→ Заказы"
        int product_id FK "→ Товары"
        int quantity "Количество"
        float unit_price "Цена за шт."
        float sum "Сумма"
    }

    STATUS_HISTORY["История статусов (84)"] {
        int id PK
        int order_id FK "→ Заказы"
        string status_from "Из статуса"
        string status_to "В статус"
        datetime date "Дата"
        string comment "Комментарий"
    }

    CATEGORIES ||--o{ PRODUCTS : "группирует"
    SOURCES ||--o{ CLIENTS : "источник"
    SOURCES ||--o{ ORDERS : "канал"

    STATUSES ||--o{ ORDERS : "текущий статус"
    DELIVERY_METHODS ||--o{ ORDERS : "доставка"

    CLIENTS ||--o{ ORDERS : "размещает"
    ORDERS ||--o{ ORDER_ITEMS : "содержит"
    PRODUCTS ||--o{ ORDER_ITEMS : "в позиции"
    ORDERS ||--o{ STATUS_HISTORY : "история"
```

---

## 3. API v2: протокол взаимодействия

```mermaid
sequenceDiagram
    participant App as IntegramV2Client
    participant API as ai2o.online
    participant DB as PostgreSQL

    Note over App,API: Аутентификация (1 раз / ~1 час)
    App->>API: POST /api/v2/iam/login<br/>{email, password}
    API-->>App: {accessToken, refreshToken}

    Note over App,API: CRUD через AI Tool endpoint
    App->>API: POST /api/v2/{workspace}/ai/tool<br/>{name: "list_objects",<br/>args: {typeId: 581},<br/>skipHitl: true}
    API->>DB: SELECT * FROM products
    DB-->>API: rows
    API-->>App: {ok: true, data: {rows: [...]}}

    Note over App,API: Создание записи
    App->>API: POST /api/v2/{workspace}/ai/tool<br/>{name: "create_object",<br/>args: {typeId: 581,<br/>fields: {Название: "Перга"}}}
    API->>DB: INSERT INTO products
    DB-->>API: id: 591
    API-->>App: {ok: true, data: {id: 591}}
```

---

## 4. Файловая структура CRM-слоя

```mermaid
graph TB
    subgraph CONFIG["Конфигурация"]
        CFG["config.py<br/>─────────<br/>INTEGRAM_V2 = true/false<br/>INTEGRAM_V2_URL<br/>INTEGRAM_V2_EMAIL<br/>INTEGRAM_V2_PASSWORD"]
    end

    subgraph V1_LAYER["v1 — ai2o.ru (архив)"]
        C1["crm_constants.py<br/>TABLE_ORDERS = 1024<br/>REQ_ORDER_DATE = 1048"]
        A1["integram_api.py<br/>HTTP: /{db}/object/{table}"]
        CL1["integram_client.py<br/>IntegramClient"]
    end

    subgraph V2_LAYER["v2 — ai2o.online (основная)"]
        C2["integram_v2_constants.py<br/>TABLE_ORDERS = 60<br/>TABLE_PRODUCTS = 581"]
        CL2["integram_v2_client.py<br/>IntegramV2Client<br/>─────────<br/>HTTP: /api/v2/{ws}/ai/tool"]
    end

    subgraph CONSUMERS["Потребители"]
        SVC["OrderService<br/>ConsultService<br/>NotificationService"]
        BOT["bot.py<br/>agents/"]
        WEB_API["web/routers/"]
    end

    CFG -->|"INTEGRAM_V2?"| V1_LAYER
    CFG -->|"INTEGRAM_V2?"| V2_LAYER

    C1 --> A1 --> CL1
    C2 --> CL2

    CONSUMERS --> CL1
    CONSUMERS --> CL2

    style V1_LAYER fill:#fee2e2
    style V2_LAYER fill:#bbf7d0
    style CFG fill:#fef3c7
```

---

## 5. Справочники: ID записей

### Категории товаров (таблица 151)

```mermaid
graph LR
    CAT["Категории товаров<br/>TABLE = 151"]

    CAT --- C1["157: Продукты<br/>пчеловодства"]
    CAT --- C2["159: Настойки"]
    CAT --- C3["161: Программы<br/>здоровья"]
    CAT --- C4["163: Мёд"]
    CAT --- C5["165: Наборы"]
    CAT --- C6["167: Упаковка"]
    CAT --- C7["169: Свечи"]
    CAT --- C8["171: Чаи и травы"]

    style CAT fill:#e8f5e9,stroke:#4caf50
    style C1 fill:#f1f8e9
    style C2 fill:#f1f8e9
    style C3 fill:#f1f8e9
    style C4 fill:#f1f8e9
    style C5 fill:#f1f8e9
    style C6 fill:#f1f8e9
    style C7 fill:#f1f8e9
    style C8 fill:#f1f8e9
```

### Статусы заказов (таблица 152)

```mermaid
graph LR
    S1["179: Новый"] --> S2["181: Подтверждён"]
    S2 --> S3["183: В сборке"]
    S3 --> S4["185: Отправлен"]
    S4 --> S5["187: Доставлен"]

    S1 --> S6["189: Отменён"]
    S2 --> S6
    S3 --> S6

    style S1 fill:#e3f2fd
    style S2 fill:#e8f5e9
    style S3 fill:#fff3e0
    style S4 fill:#f3e5f5
    style S5 fill:#c8e6c9
    style S6 fill:#ffcdd2
```

### Способы доставки и Источники

```mermaid
graph TB
    subgraph DEL["Способы доставки (150)"]
        D1["191: СДЭК"]
        D2["193: Почта России"]
        D3["195: Самовывоз"]
    end

    subgraph SRC["Источники (15)"]
        S1["19: Telegram"]
        S2["21: UDS"]
        S3["20: WhatsApp"]
        S4["17: ВК"]
        S5["18: Instagram"]
        S6["22: Личное обращение"]
    end

    style DEL fill:#e3f2fd
    style SRC fill:#fff3e0
```

---

## 6. Колонки и реквизиты: полная карта

### Товары (таблица 581)

| Колонка | REQ ID | Тип | REF |
|---------|--------|-----|-----|
| Название | 582 | string | — |
| Категория | 583 | ref | → 151 (Категории) |
| Цена | 584 | number | — |
| Вес | 585 | number | — |
| Описание | 586 | memo | — |
| В наличии | 587 | bool | — |
| Артикул UDS | 588 | string | — |
| Короткое название | 589 | string | — |
| Остаток | 590 | number | — |

### Клиенты (таблица 52)

| Колонка | REQ ID | Тип | REF |
|---------|--------|-----|-----|
| ФИО | 206 | string | — |
| Телефон | 207 | string | — |
| Telegram ID | 208 | number | — |
| Telegram Username | 209 | string | — |
| Адрес | 210 | memo | — |
| Город | 211 | string | — |
| Комментарий | 212 | memo | — |
| Источник | 213 | ref | → 15 (Источники) |

### Заказы (таблица 60)

| Колонка | REQ ID | Тип | REF |
|---------|--------|-----|-----|
| Дата | 214 | datetime | — |
| Адрес доставки | 215 | memo | — |
| Стоимость доставки | 216 | number | — |
| Сумма товаров | 217 | number | — |
| Итого | 218 | number | — |
| Трек-номер | 219 | string | — |
| Комментарий | 220 | memo | — |
| Клиент | 221 | ref | → 52 (Клиенты) |
| Статус | 222 | ref | → 152 (Статусы) |
| Способ доставки | 223 | ref | → 150 (Доставка) |
| Источник | 224 | ref | → 15 (Источники) |
| Мессенджер | 225 | string | — |
| Дата отправки | 226 | datetime | — |
| Дата доставки | 227 | datetime | — |

### Позиции заказа (таблица 78)

| Колонка | REQ ID | Тип | REF |
|---------|--------|-----|-----|
| Количество | 228 | number | — |
| Цена за шт. | 229 | number | — |
| Сумма | 230 | number | — |
| Заказ | 232 | ref | → 60 (Заказы) |
| Товар | 628 | ref | → 581 (Товары) |

---

## 7. Инфраструктура: полная топология

```mermaid
graph TB
    subgraph USERS["Пользователи"]
        U1["👤 Подписчики<br/>(Telegram)"]
        U2["🧑‍💼 Пчеловод<br/>(Admin)"]
        U3["👷 Работники<br/>(Worker)"]
        U4["🖥️ Веб-панель<br/>(браузер)"]
    end

    subgraph VPS["VPS 185.233.200.13"]
        BOT["beebot<br/>Docker<br/>~762 MiB"]
        BACKEND["beebot-web<br/>FastAPI :8088"]
        REDIS["Redis :6379"]
    end

    subgraph HIVE["Hive (локальная)"]
        GROQ_P["groq-proxy :8990"]
        SOCKS["SOCKS5 :9150"]
        DEVBOT["DEVBOT :8091"]
    end

    subgraph CLOUD["Облачные сервисы"]
        TG_API["Telegram API"]
        GROQ_API["Groq API"]
        CRM_V1[("ai2o.ru<br/>v1 архив")]
        CRM_V2[("ai2o.online<br/>v2 основная")]
        CDEK_API["СДЭК API"]
        POCHTA_API["Почта России"]
    end

    U1 & U2 & U3 -->|Telegram| TG_API
    U4 -->|HTTP :8088| BACKEND

    TG_API -->|SOCKS5| SOCKS -->|SSH tunnel| BOT
    BOT -->|SSH tunnel| GROQ_P --> GROQ_API
    BOT --> REDIS --> BACKEND

    BOT -->|"v1 (read)"| CRM_V1
    BOT -->|"v2 (read+write)"| CRM_V2
    BACKEND -->|"v2"| CRM_V2
    BACKEND --> CDEK_API & POCHTA_API

    U2 -->|"/dev"| DEVBOT

    style CRM_V1 fill:#fee2e2
    style CRM_V2 fill:#bbf7d0
    style HIVE fill:#f3e5f5
    style VPS fill:#e3f2fd
```

---

## 8. Жизненный цикл заказа

```mermaid
stateDiagram-v2
    [*] --> Новый : Клиент оформляет

    Новый --> Подтверждён : Пчеловод проверяет
    Новый --> Отменён : Клиент отказался

    Подтверждён --> В_сборке : Работник берёт
    Подтверждён --> Отменён : Нет товара

    В_сборке --> Отправлен : Трек-номер получен
    В_сборке --> Отменён : Проблема при сборке

    Отправлен --> Доставлен : Трекер подтвердил

    state Новый {
        [*] --> Telegram_FSM
        [*] --> UDS_Poller
        [*] --> Веб_панель
    }

    state Отправлен {
        [*] --> СДЭК_трекинг
        [*] --> Почта_трекинг
    }
```

---

## 9. v1 vs v2: сравнительная таблица

| Параметр | v1 (ai2o.ru) | v2 (ai2o.online) |
|----------|-------------|-----------------|
| URL | `https://ai2o.ru` | `https://ai2o.online` |
| Auth | `/{db}/auth?JSON` (cookie) | `/api/v2/iam/login` (JWT) |
| CRUD | `/{db}/object/{table}?JSON` | `/api/v2/{ws}/ai/tool` |
| Формат полей | `reqId → value` (числовые ID) | `column_name → value` (по имени) |
| REF-поля | `set_reference_field(id, reqId, refId)` | `fields: {"Статус": 179}` |
| Клиент Python | `IntegramClient` | `IntegramV2Client` |
| Константы | `crm_constants.py` | `integram_v2_constants.py` |
| Workspace | `bibot` (база данных) | `alekseymavai` (workspace slug) |
| Данные | 1924 клиента, 1915 заказов (грязные) | 85 товаров (чистые), 0 заказов |
| Статус | **read-only архив** | **основная CRM** |

---

## 10. Переключение: feature flag

```mermaid
flowchart TD
    START["Запрос к CRM"]
    CHECK{"config.INTEGRAM_V2?"}

    V1_PATH["IntegramClient<br/>─────────<br/>ai2o.ru/bibot<br/>cookie auth<br/>reqId-based fields"]

    V2_PATH["IntegramV2Client<br/>─────────<br/>ai2o.online<br/>JWT auth<br/>name-based fields"]

    START --> CHECK
    CHECK -->|"False (default)"| V1_PATH
    CHECK -->|"True"| V2_PATH

    ENV[".env<br/>─────────<br/>INTEGRAM_V2=true<br/>INTEGRAM_V2_EMAIL=...<br/>INTEGRAM_V2_PASSWORD=...<br/>INTEGRAM_V2_WORKSPACE=alekseymavai"]

    ENV -.->|"загружает"| CHECK

    style V1_PATH fill:#fee2e2
    style V2_PATH fill:#bbf7d0
    style ENV fill:#fef3c7
```

---

*Связанные документы:*
- *[architecture.md](architecture.md) — архитектура бота (Hexagonal, Redis, Service Layer)*
- *[../analysis.md](../analysis.md) — анализ проблем*
- *[../plan.md](../plan.md) — план развития*
- *[../src/integram_v2_constants.py](../src/integram_v2_constants.py) — все ID таблиц и колонок*
