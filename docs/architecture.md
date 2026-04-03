# BEEBOT — Архитектурные диаграммы

> **Версия:** 3 апреля 2026

---

## 1. Общая архитектура

```mermaid
graph TB
    subgraph USERS["Пользователи"]
        U1["Подписчики<br/>(Telegram)"]
        U2["Пчеловод<br/>(Admin)"]
        U3["Работники<br/>(Worker)"]
        U4["Веб-панель<br/>(Браузер)"]
    end

    subgraph BOT["Telegram-бот (Docker)"]
        HANDLERS["Роутеры aiogram<br/>admin / user / fsm / inspect / worker"]
        ORCH["Оркестратор<br/>(LangGraph)"]
        AGENTS["6 агентов"]
    end

    subgraph WEB["Веб-панель (Docker)"]
        API["FastAPI :8088"]
        VUE["Vue 3 PWA<br/>14 страниц"]
    end

    subgraph INFRA["Инфраструктура"]
        CRM_V2[("ai2o.online<br/>CRM v2")]
        CRM_V1[("ai2o.ru<br/>CRM v1 архив")]
        KB["FAISS<br/>276 чанков"]
        LLM["Groq API<br/>llama-3.3-70b"]
        MEM["SQLite<br/>память"]
    end

    subgraph EXTERNAL["Внешние системы"]
        UDS_API["UDS App<br/>(магазин)"]
        CDEK_A["СДЭК API"]
        POCHTA_A["Почта России"]
    end

    U1 & U2 & U3 --> BOT
    U4 --> VUE --> API

    HANDLERS --> ORCH --> AGENTS
    AGENTS --> KB & LLM & MEM
    AGENTS --> CRM_V2
    API --> CRM_V2
    AGENTS -.->|"read-only"| CRM_V1

    UDS_API -->|"polling 5 мин"| BOT
    BOT -->|"авто-трекинг 2ч"| CDEK_A & POCHTA_A

    style CRM_V2 fill:#bbf7d0,stroke:#22c55e
    style CRM_V1 fill:#fee2e2,stroke:#ef4444
    style EXTERNAL fill:#f3e5f5
```

---

## 2. Оркестратор: маршрутизация интентов

```mermaid
flowchart TD
    MSG["Сообщение от пользователя"] --> FSM{"FSM-состояние?"}

    FSM -->|"OrderFSM"| ORDER_FSM["Шаг диалога заказа<br/>(logist.py)"]
    FSM -->|"InspectFSM"| INSPECT_FSM["Шаг диалога осмотра<br/>(inspector.py)"]
    FSM -->|"Нет"| MODE{"Режим?"}

    MODE -->|"WORKER"| WORKER["Очередь сборки<br/>(worker.py)"]
    MODE -->|"ADMIN /admin"| ADMIN["Ассистент + CRM<br/>(admin_chat.py)"]
    MODE -->|"Обычный"| ORCH["Оркестратор"]

    ORCH --> CLASSIFY{"Классификация intent"}

    CLASSIFY -->|"consult"| BEEBOT["BeebotAgent<br/>FAISS → LLM"]
    CLASSIFY -->|"order"| START_FSM["Запуск OrderFSM"]
    CLASSIFY -->|"stats"| ANALYST["AnalystAgent<br/>ABC / сезонность"]
    CLASSIFY -->|"greeting"| GREET["Приветствие"]
    CLASSIFY -->|"edit/track"| MENU["Меню заказа"]
    CLASSIFY -->|"inspect"| START_INSPECT["Запуск InspectFSM"]

    BEEBOT --> RESPONSE["Ответ пользователю"]
    ORDER_FSM & INSPECT_FSM & WORKER & ADMIN & START_FSM & ANALYST & GREET & MENU & START_INSPECT --> RESPONSE

    style ORCH fill:#e3f2fd
    style CLASSIFY fill:#fff3e0
```

---

## 3. Агенты: зависимости и возможности

```mermaid
graph LR
    subgraph AGENTS["Агенты"]
        BEEBOT["Консультант<br/>(beebot.py)"]
        LOGIST["Логист<br/>(logist.py)"]
        ANALYST["Аналитик<br/>(analyst.py)"]
        INSPECTOR["Инспектор<br/>(inspector.py)"]
        ADMIN_CHAT["Ассистент<br/>(admin_chat.py)"]
        WORKER_A["Работник<br/>(worker.py)"]
    end

    KB["FAISS KB"]
    LLM["Groq LLM"]
    CRM["CRM"]
    MEM["Память"]

    BEEBOT --> KB & LLM
    LOGIST --> CRM & LLM
    ANALYST --> CRM & LLM
    INSPECTOR --> KB & LLM
    ADMIN_CHAT --> CRM & LLM
    WORKER_A --> CRM

    style BEEBOT fill:#e8f5e9
    style LOGIST fill:#e3f2fd
    style ANALYST fill:#fff3e0
    style INSPECTOR fill:#f3e5f5
    style ADMIN_CHAT fill:#fce4ec
    style WORKER_A fill:#e0f2f1
```

### Сравнительная таблица агентов

| Агент | KB | CRM | LLM | Вход | Выход |
|---|---|---|---|---|---|
| Консультант | Чтение | — | Groq | consult | Текст + источники |
| Логист | — | Запись | Groq | order (FSM) | Заказ в CRM |
| Аналитик | — | Чтение | Groq | stats | Отчёт (текст) |
| Инспектор | Чтение | — | Groq | /inspect (FSM) | Рекомендация |
| Ассистент | — | CrmSnapshot | Groq | /admin | Диалог |
| Работник | — | Чтение+Запись | — | /start (worker) | Кнопки |

---

## 4. Жизненный цикл заказа

```mermaid
stateDiagram-v2
    [*] --> Новый : Telegram / UDS / Веб

    Новый --> Подтверждён : Пчеловод проверяет
    Новый --> Отменён : Отказ

    Подтверждён --> В_сборке : Работник берёт
    Подтверждён --> Отменён : Нет товара

    В_сборке --> Отправлен : Трек-номер
    В_сборке --> Отменён : Проблема

    Отправлен --> Доставлен : Трекер подтвердил
```

### Источники заказов

| Источник | Как попадает | Уведомления |
|----------|-------------|-------------|
| Telegram FSM | logist.py → CRM | Пчеловод + работники |
| UDS-магазин | uds.py → CRM | Пчеловод + работники |
| Веб-панель | orders.py → CRM | Только пчеловод |

---

## 5. CRM: две системы

```mermaid
graph TB
    subgraph APP["BEEBOT"]
        FF{{"INTEGRAM_V2<br/>feature flag"}}
        V1_CL["IntegramClient<br/>(v1)"]
        V2_CL["IntegramV2Client<br/>(v2)"]
    end

    subgraph V1["ai2o.ru (АРХИВ)"]
        V1_DB[("bibot<br/>1924 клиента<br/>1915 заказов<br/>76 товаров")]
    end

    subgraph V2["ai2o.online (ОСНОВНАЯ)"]
        V2_DB[("alekseymavai<br/>85 товаров<br/>4 справочника<br/>чистые данные")]
    end

    FF -->|"true"| V2_CL --> V2_DB
    FF -->|"false"| V1_CL --> V1_DB

    style V1 fill:#fee2e2
    style V2 fill:#bbf7d0
    style FF fill:#fef3c7
```

### Схема таблиц CRM v2

```mermaid
erDiagram
    CATEGORIES["Категории (151)"] ||--o{ PRODUCTS["Товары (581)"] : "группирует"
    SOURCES["Источники (15)"] ||--o{ CLIENTS["Клиенты (52)"] : "канал"
    SOURCES ||--o{ ORDERS["Заказы (60)"] : "канал"
    STATUSES["Статусы (152)"] ||--o{ ORDERS : "текущий"
    DELIVERY["Доставка (150)"] ||--o{ ORDERS : "способ"
    CLIENTS ||--o{ ORDERS : "размещает"
    ORDERS ||--o{ ORDER_ITEMS["Позиции (78)"] : "содержит"
    PRODUCTS ||--o{ ORDER_ITEMS : "товар"

    PRODUCTS {
        int id PK
        string name
        float price
        int stock
        ref category
    }

    CLIENTS {
        int id PK
        string full_name
        string phone
        int telegram_id
        string city
    }

    ORDERS {
        int id PK
        datetime date
        ref client
        ref status
        ref delivery
        float total
        string tracking
    }

    ORDER_ITEMS {
        int id PK
        ref order
        ref product
        int quantity
        float price
    }
```

---

## 6. Инфраструктура: туннели и деплой

```mermaid
graph TB
    subgraph VPS["VPS 185.233.200.13"]
        BOT_C["beebot<br/>~762 MiB"]
        WEB_C["beebot-web<br/>:8088"]
        REDIS_C["Redis<br/>:6379"]
    end

    subgraph HIVE["Hive (локальная)"]
        GROQ_P["groq-proxy :8990"]
        SOCKS["SOCKS5 :9150"]
        DEVBOT_C["DEVBOT :8091"]
    end

    subgraph CLOUD["Облачные сервисы"]
        TG_API["Telegram API"]
        GROQ_API["Groq API"]
        CRM_V2_C["ai2o.online"]
        UDS_C["UDS Partner API"]
        CDEK["СДЭК API"]
        POCHTA["Почта России"]
    end

    BOT_C -->|"SSH tunnel"| GROQ_P --> GROQ_API
    BOT_C -->|"SOCKS5"| SOCKS --> TG_API
    BOT_C --> REDIS_C --> WEB_C
    BOT_C & WEB_C --> CRM_V2_C
    BOT_C -->|"polling 5 мин"| UDS_C
    WEB_C --> CDEK & POCHTA

    style VPS fill:#e3f2fd
    style HIVE fill:#f3e5f5
```

### Docker-контейнеры

| Контейнер | Образ | RAM | Порт |
|-----------|-------|-----|------|
| redis | redis:7-alpine | ~20 MiB | 6379 |
| beebot | Python 3.12 + FAISS + Groq | ~762 MiB | — |
| beebot-web | Python 3.12 + Vue dist | ~50 MiB | 8088 |

---

## 7. Файловая структура: три слоя

```mermaid
graph TB
    subgraph TRANSPORT["Транспорт (вход)"]
        TG["telegram/<br/>bot.py + роутеры"]
        WEB_T["web/<br/>api.py + роутеры"]
    end

    subgraph LOGIC["Бизнес-логика"]
        ORCH_L["Оркестратор"]
        AGENTS_L["6 агентов"]
        SVC["OrderService<br/>NotificationService"]
        UDS_L["UDS Poller<br/>(sync каждые 5 мин)"]
    end

    subgraph INFRA_L["Инфраструктура (выход)"]
        CRM_L["CRM v1 / v2"]
        LLM_L["Groq LLM"]
        KB_L["FAISS KB"]
        DEL_L["СДЭК / Почта"]
        MEM_L["SQLite память"]
    end

    TG --> LOGIC
    WEB_T --> LOGIC
    LOGIC --> INFRA_L

    style TRANSPORT fill:#e3f2fd
    style LOGIC fill:#e8f5e9
    style INFRA_L fill:#fff3e0
```

---

## 8. Поток консультации: пользователь → ответ

```mermaid
sequenceDiagram
    participant User as Пользователь
    participant Bot as Telegram-бот
    participant Orch as Оркестратор
    participant Agent as BeebotAgent
    participant KB as FAISS (276 чанков)
    participant LLM as Groq (llama-3.3-70b)

    User->>Bot: "Чем полезна перга?"
    Bot->>Orch: route(query, user_id)
    Orch->>Orch: classify → "consult"
    Orch->>Agent: answer(query, history, style)
    Agent->>KB: search(query, top_k=5)
    KB-->>Agent: [chunk1, chunk2, chunk3]
    Agent->>LLM: system_prompt + chunks + query
    LLM-->>Agent: "Перга — это пыльца..."
    Agent-->>Orch: (response, chunks)
    Orch-->>Bot: response
    Bot-->>User: "Перга — это пыльца..."
```

---

## 9. Поток заказа: FSM 7 шагов

```mermaid
sequenceDiagram
    participant User as Пользователь
    participant Bot as Telegram-бот
    participant FSM as OrderFSM
    participant Logist as LogistAgent
    participant CRM as Integram CRM

    User->>Bot: /order
    Bot->>FSM: start
    FSM->>Logist: start_order()
    Logist->>CRM: get_products()
    CRM-->>Logist: [85 товаров]
    Logist-->>FSM: каталог + клавиатура

    loop 7 шагов
        FSM-->>User: Вопрос (товары/ФИО/телефон/адрес/доставка)
        User->>FSM: Ответ
    end

    FSM->>Logist: create_order(client, items, delivery)
    Logist->>CRM: create_order()
    Logist->>Bot: notify_beekeeper()
    Logist->>Bot: notify_workers()
    Bot-->>User: "Заказ #TG-20260402 создан!"
```

---

## 10. UDS-синхронизация: магазин → CRM

```mermaid
sequenceDiagram
    participant UDS as UDS Partner API
    participant Poller as UDSPoller<br/>(каждые 5 мин)
    participant Dedup as TransactionDeduplicator
    participant CRM as Integram CRM
    participant Bot as Telegram-бот
    participant Admin as Пчеловод

    Note over Poller: Старт бота
    Poller->>CRM: get_orders() — загрузить UDS-* заказы
    CRM-->>Dedup: {UDS-001, UDS-002, ...}

    Note over Poller: Catch-up с 01.01.2024
    Poller->>UDS: get_transactions_since(2024-01-01)
    UDS-->>Poller: [tx1, tx2, ..., txN]
    loop Каждая транзакция
        Poller->>Dedup: is_new(tx)?
        alt Новая
            Poller->>CRM: get_or_create_client(phone)
            Poller->>CRM: get_product_by_sku(sku)
            Poller->>CRM: create_order(source="UDS")
            Dedup->>Dedup: mark_seen(tx.id)
        end
    end

    Note over Poller: Обычный polling
    loop Каждые 5 минут
        Poller->>UDS: get_transactions(limit=50)
        UDS-->>Poller: [последние транзакции]
        Poller->>Dedup: is_new(tx)?
        alt Новая
            Poller->>CRM: sync_uds_transaction()
            Poller->>Bot: уведомление
            Bot->>Admin: 🛒 Новый заказ из UDS
        end
    end
```

### Компоненты UDS-интеграции

| Компонент | Файл | Назначение |
|-----------|------|-----------|
| UDSClient | src/integrations/uds.py | REST-клиент UDS Partner API v2 (Basic Auth, retry 3×) |
| UDSPoller | src/integrations/uds.py | Фоновый polling + catch-up + дедупликация |
| TransactionDeduplicator | src/integrations/uds.py | Хранит обработанные ID, загружает из CRM при старте |
| sync_uds_transaction() | src/integrations/uds.py | Транзакция → клиент → товары по SKU → заказ → уведомление |
| sync_uds_catalog() | src/integrations/uds.py | Сопоставление каталога UDS ↔ Integram по артикулу |

### Известные ограничения

- **SKU-матчинг** — если артикул UDS не совпадает с полем «Артикул UDS» в CRM, товар не находится → `product_id=0`
- **Дедупликация в RAM** — при рестарте заново загружается из CRM (надёжно, но медленно при большом числе заказов)
- **Нет обратной синхронизации** — изменения в CRM не отправляются обратно в UDS

---

## 11. Голос Улья: 5 стилей

| Стиль | Описание | Когда использовать |
|-------|---------|-------------------|
| Наставник | Тёплый, отеческий тон | По умолчанию |
| Практик | Конкретные советы, цифры | Опытные пчеловоды |
| Селекционер | Научный подход, исследования | Вопросы о генетике, породах |
| Зимовщик | Спокойный, вдумчивый | Зимний период, подготовка |
| Эколог | Природа, экосистема | Вопросы о среде обитания |

---

*Связанные документы: [analysis.md](../analysis.md) | [plan.md](../plan.md) | [README.md](../README.md)*
