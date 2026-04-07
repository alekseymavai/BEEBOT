# BEEBOT — Архитектурные диаграммы

> **Версия:** 6 апреля 2026 — Микроядерная архитектура (BeeBotApp + 11 плагинов)
> **CRM:** v2 (ai2o.online, singleflight) + v1 архив (ai2o.ru)

---

## 0. Микроядерная архитектура (Plugin System)

```mermaid
graph TD
    subgraph KERNEL["src/kernel/ — Ядро"]
        APP["BeeBotApp\n(топосортировка + lifecycle)"]
        CONTAINER["Container\n(DI-реестр сервисов)"]
        PLUGIN["Plugin ABC\n(setup / register_routers\n/ get_bg_tasks / teardown)"]
        APP --> CONTAINER
        APP --> PLUGIN
    end

    subgraph PLUGINS["src/plugins/ — Плагины"]
        P_CRM["CrmPlugin\n→ crm, auth"]
        P_KB["KnowledgePlugin\n→ kb"]
        P_AGENTS["AgentsPlugin\n→ orchestrator, logist\ninspector, analyst\nadmin_chat, consult"]
        P_ORDERS["OrdersPlugin\n→ order_service\nnotification_service"]
        P_ANALYTICS["AnalyticsPlugin\n→ analytics_service\ndashboard_service"]
        P_DELIVERY["DeliveryPlugin\n→ delivery_service"]
        P_WORKERS["WorkersPlugin\n→ worker_service"]
        P_GIFT["GiftPlugin\n→ gift_broker, crm_agent\nagent_specs"]
        P_MON["MonitoringPlugin\n→ bg_manager\n5 BgTask-ов"]
        P_TG["TelegramPlugin\n→ 7 aiogram Router-ов"]
        P_WEB["WebPlugin\n→ FastAPI inject"]
    end

    P_CRM --> CONTAINER
    P_KB --> CONTAINER
    P_AGENTS --> CONTAINER
    P_ORDERS --> CONTAINER
    P_ANALYTICS --> CONTAINER
    P_DELIVERY --> CONTAINER
    P_WORKERS --> CONTAINER
    P_GIFT --> CONTAINER
    P_MON --> CONTAINER
    P_TG --> CONTAINER
    P_WEB --> CONTAINER

    P_KB -.зависит.-> P_CRM
    P_AGENTS -.зависит.-> P_CRM & P_KB
    P_ORDERS -.зависит.-> P_CRM & P_AGENTS
    P_ANALYTICS -.зависит.-> P_CRM & P_AGENTS
    P_WORKERS -.зависит.-> P_CRM
    P_GIFT -.зависит.-> P_CRM & P_AGENTS
    P_MON -.зависит.-> P_CRM
    P_TG -.зависит.-> P_CRM & P_AGENTS & P_ORDERS & P_GIFT & P_WORKERS
    P_WEB -.зависит.-> P_CRM & P_ORDERS & P_ANALYTICS & P_DELIVERY & P_WORKERS
```

**bot.py** (95 строк) — только цепочка `app.register(...).register(...).start()` + `gather(polling, uvicorn)`.
Добавить новый модуль = создать `Plugin`-подкласс + одна строка `app.register(NewPlugin())`.

---

## 1. Общая архитектура: один процесс

```mermaid
graph TB
    subgraph CLIENTS["Клиенты"]
        TG_USERS["Подписчики / Пчеловод / Работники\n(Telegram)"]
        BROWSER["Браузер\n(Vue 3 PWA)"]
    end

    subgraph BEEBOT["beebot — один контейнер, один процесс"]
        subgraph TRANSPORT["Транспорт"]
            POLLING["aiogram polling\n7 роутеров"]
            FASTAPI["FastAPI :8088\n9 веб-роутеров + Vue dist/"]
        end

        subgraph AGENTS["Агенты (тонкие обёртки)"]
            ORCH["Оркестратор\n(LangGraph)"]
            BEEBOT_A["BeebotAgent\nFAISS → LLM"]
            ANALYST_A["AnalystAgent\nABC/сезонность"]
            LOGIST_A["LogistAgent\nFSM 7 шагов"]
            WORKER_A["WorkerAgent\nОчередь сборки"]
            ADMIN_A["AdminChatAgent\nLLM + CrmSnapshot"]
            INSPECT_A["InspectorAgent\nОсмотр улья"]
        end

        subgraph SVC["Service Layer"]
            ORDER_SVC["OrderService\nCRUD + уведомления"]
            NOTIFY["NotificationService\nTelegram push"]
        end

        subgraph BG["Фоновые задачи"]
            SNAP["CrmSnapshot\n5 мин"]
            TRACK["OrderTracker\n2 часа"]
            UDS_P["UDSPoller\n5 мин"]
            TUN["TunnelMonitor\n60 сек"]
            BACK["BackupManager\nежедневно"]
        end

        STARTUP["startup.py\ncreate_services()"]
    end

    subgraph INFRA["Хранилища"]
        CRM_V2[("CRM v2\nai2o.online")]
        CRM_V1[("CRM v1 архив\nai2o.ru")]
        REDIS[("Redis :6379")]
        KB["FAISS\n276 чанков"]
        LLM["Groq API\nllama-3.3-70b"]
        SQLITE["SQLite\nmemory.db"]
    end

    subgraph HIVE["Hive (локальная машина)"]
        GROQ_P["groq-proxy :8990"]
        SOCKS["SOCKS5 :9150"]
        DEVBOT_H["DEVBOT :8091"]
    end

    TG_USERS -->|Telegram API| POLLING
    BROWSER -->|HTTP /api/*| FASTAPI

    STARTUP --> AGENTS & SVC & BG

    POLLING --> ORCH --> AGENTS
    AGENTS --> SVC
    FASTAPI --> SVC

    SVC --> CRM_V2
    AGENTS --> KB & LLM & SQLITE
    BG --> CRM_V2 & CRM_V1 & REDIS

    BEEBOT -->|SSH tunnel| GROQ_P --> LLM
    BEEBOT -->|SOCKS5| SOCKS

    style BEEBOT fill:#f0f9ff,stroke:#1976d2
    style SVC fill:#e8f5e9,stroke:#22c55e
    style AGENTS fill:#fff3e0
    style BG fill:#fff8e1,stroke:#f9a825
    style HIVE fill:#f3e5f5
```

**Ключевой принцип: Bot → Service Layer ← Frontend**

Два клиента — Telegram и браузер. Оба обращаются к Service Layer через разный транспорт. Один процесс — нет IPC, -400 MiB RAM.

---

## 2. Маршрутизация сообщений: от пользователя к агенту

```mermaid
flowchart TD
    MSG["Сообщение пользователя"] --> FSM_CHECK{"FSM-состояние?"}

    FSM_CHECK -->|"OrderFSM активна"| ORDER_STEP["Следующий шаг заказа\n(LogistAgent)"]
    FSM_CHECK -->|"InspectFSM активна"| INSPECT_STEP["Следующий шаг осмотра\n(InspectorAgent)"]
    FSM_CHECK -->|"EditFSM активна"| EDIT_STEP["Редактирование состава\n(LogistAgent)"]
    FSM_CHECK -->|"Нет"| MODE_CHECK{"Режим пользователя?"}

    MODE_CHECK -->|"WORKER_CHAT_IDS"| WORKER["WorkerAgent\nОчередь сборки"]
    MODE_CHECK -->|"ADMIN_MODE + /admin"| ADMIN["AdminChatAgent\nLLM + CrmSnapshot"]
    MODE_CHECK -->|"Обычный"| ORCH["Оркестратор\n(LangGraph)"]

    ORCH --> CLASSIFY{"Классификация intent\n(llama-3.3-70b)"}

    CLASSIFY -->|"consult"| BEEBOT_A["BeebotAgent\nFAISS → LLM → ответ"]
    CLASSIFY -->|"stats"| ANALYST_A["AnalystAgent\nABC-анализ"]
    CLASSIFY -->|"greeting"| GREET["Приветствие\nс кнопками"]
    CLASSIFY -->|"order → END"| START_ORDER["/order → OrderFSM"]
    CLASSIFY -->|"edit → END"| START_EDIT["/edit → EditFSM"]
    CLASSIFY -->|"track → END"| TRACK_INFO["Трекинг заказа"]
    CLASSIFY -->|"inspect → END"| START_INSPECT["/inspect → InspectFSM"]

    ORDER_STEP & INSPECT_STEP & EDIT_STEP & WORKER & ADMIN & BEEBOT_A & ANALYST_A & GREET & START_ORDER & START_EDIT & TRACK_INFO & START_INSPECT --> RESPONSE["Ответ\nпользователю"]

    style ORCH fill:#e3f2fd
    style CLASSIFY fill:#fff3e0
    style BEEBOT_A fill:#e8f5e9
    style ANALYST_A fill:#e8f5e9
```

---

## 3. Поток консультации: вопрос → ответ

```mermaid
sequenceDiagram
    participant User as Пользователь
    participant Router as user_router
    participant Orch as Оркестратор
    participant Agent as BeebotAgent
    participant KB as FAISS (276 чанков)
    participant LLM as Groq llama-3.3-70b

    User->>Router: "Чем полезна перга?"
    Router->>Orch: route(message, user_id, history)
    Orch->>Orch: classify_intent() → "consult"
    Orch->>Agent: answer(query, history, voice_style)
    Agent->>KB: search(query, top_k=5)
    Note over KB: 70% семантика (FAISS)<br/>30% стилометрия<br/>keyword-буст из CRM
    KB-->>Agent: [chunk1, chunk2, chunk3]
    Agent->>LLM: system_prompt + chunks + query + history
    LLM-->>Agent: "Перга — это законсервированная пыльца..."
    Agent-->>Orch: (response, source_chunks)
    Orch-->>Router: response
    Router-->>User: "Перга — это..."
```

---

## 4. Поток заказа: FSM → OrderService → Events

```mermaid
sequenceDiagram
    participant User as Пользователь
    participant FSM as OrderFSM (7 шагов)
    participant Logist as LogistAgent
    participant OS as OrderService
    participant CRM as Integram CRM v2
    participant Notify as NotificationService
    participant SSE as SSE (браузер)

    User->>FSM: /order
    FSM->>Logist: start_order()
    Logist->>CRM: get_products()
    CRM-->>Logist: [85 товаров]

    loop 7 шагов FSM
        FSM-->>User: Вопрос (товары → ФИО → тел. → адрес → доставка → подтверждение)
        User->>FSM: Ответ
    end

    FSM->>Logist: create_order(client_data, items, delivery)
    Logist->>OS: create_order_with_client()
    OS->>CRM: get_or_create_client(phone)
    OS->>CRM: create_order(items, delivery)
    OS->>Notify: notify_beekeeper(order)
    OS->>Notify: notify_workers(order)
    OS->>SSE: push("order.created", order_id)
    SSE-->>Browser: real-time обновление
    FSM-->>User: "Заказ #TG-20260402 создан! ✅"
```

---

## 5. CRM v2: архитектура клиента и singleflight

```mermaid
graph TB
    subgraph APP["beebot — запросы к CRM"]
        PARALLEL["Параллельные запросы\n(5 пользователей одновременно)"]
        SF{"singleflight\n_orders_lock"}
        CACHE{"Кэш заказов\n_orders_cache\nTTL: 90 сек"}
    end

    subgraph CLIENT["IntegramV2Client"]
        AUTH["JWT авторизация\nauto re-auth"]
        GET_ORDERS["get_orders()\nREST пагинация"]
        GET_ITEMS["get_order_items()\nparentId фильтр"]
        PAGINATION["_fetch_all_order_ids_rest()\nlast_n_pages поддержка"]
    end

    subgraph CRM["ai2o.online — Integram v2"]
        direction LR
        ORDERS_TBL[("Таблица заказов\n1915+ записей")]
        ITEMS_TBL[("Позиции заказов\nparentId связь")]
        CLIENTS_TBL[("Клиенты\n1924+ записей")]
        PRODUCTS_TBL[("Товары\n85 позиций")]
    end

    PARALLEL --> SF
    SF -->|"Если уже есть запрос — ждать"| SF
    SF -->|"Если кэш свежий"| CACHE
    SF -->|"Иначе — один запрос"| GET_ORDERS

    GET_ORDERS --> AUTH --> PAGINATION
    PAGINATION --> ORDERS_TBL
    GET_ITEMS --> ITEMS_TBL
    GET_ORDERS --> CACHE

    style SF fill:#fef3c7,stroke:#f59e0b
    style CACHE fill:#e8f5e9,stroke:#22c55e
    style APP fill:#e3f2fd
```

**Singleflight:** при 5 одновременных запросах `get_orders()` — только 1 HTTP-запрос к CRM. Остальные 4 ждут результат первого. Кэш 90 сек предотвращает лавину при частых запросах.

---

## 6. CRM v1 vs v2 — сравнение

```mermaid
graph LR
    subgraph FACTORY["crm_factory.py"]
        FLAG{{"INTEGRAM_V2\nenv flag"}}
    end

    subgraph V1["ai2o.ru — CRM v1 (АРХИВ)"]
        V1_CL["IntegramClient\n(849 строк)"]
        V1_AUTH["Cookie-based auth"]
        V1_DB[("bibot DB\n1924 клиента\n1915 заказов")]
    end

    subgraph V2["ai2o.online — CRM v2 (ОСНОВНАЯ)"]
        V2_CL["IntegramV2Client\n(1002 строки, 27 тестов)"]
        V2_AUTH["JWT auth\nauto re-auth"]
        V2_DB[("alekseymavai workspace\n85 товаров\n+ заказы/клиенты)"]
    end

    FLAG -->|"false (сейчас)"| V1_CL --> V1_AUTH --> V1_DB
    FLAG -->|"true (цель A.7)"| V2_CL --> V2_AUTH --> V2_DB

    style V1 fill:#fee2e2,stroke:#ef4444
    style V2 fill:#bbf7d0,stroke:#22c55e
    style FLAG fill:#fef3c7,stroke:#f59e0b
```

| Характеристика | v1 (ai2o.ru) | v2 (ai2o.online) |
|---------------|-------------|-----------------|
| Аутентификация | Cookie-based | JWT (auto re-auth) |
| Адресация полей | По REQ_ID (числа) | По имени колонки |
| Пагинация | Отдельный REST API | REST + AI Tools |
| Параллельность | Нет защиты | Singleflight + кэш |
| Тесты | Есть | 27 unit-тестов |
| Статус в prod | ✅ Используется | ⏳ Ожидает A.7 |
| Данные | 1924 клиента, 1915 заказов | 85 товаров (растёт) |

---

## 7. Жизненный цикл заказа

```mermaid
stateDiagram-v2
    [*] --> Новый : Telegram FSM / UDS поллер / Веб-панель

    Новый --> Подтверждён : Пчеловод нажал «Подтвердить»
    Новый --> Отменён : Отказ клиента / нет товара

    Подтверждён --> В_сборке : Работник берёт заказ
    Подтверждён --> Отменён : Пересмотр

    В_сборке --> Отправлен : Трек-номер введён
    В_сборке --> Отменён : Проблема со сборкой

    Отправлен --> Доставлен : OrderTracker подтвердил
    Отправлен --> Возврат : Не забрали / отказ
```

### Три источника заказов

| Источник | Путь | Уведомления |
|----------|------|-------------|
| Telegram FSM | LogistAgent → OrderService → CRM | Пчеловод + работники + SSE |
| UDS-магазин | UDSPoller → OrderService → CRM | Пчеловод + работники |
| Веб-панель | POST /api/orders → OrderService → CRM | Пчеловод + SSE |

---

## 8. UDS-синхронизация

```mermaid
sequenceDiagram
    participant UDS as UDS Partner API
    participant Poller as UDSPoller (каждые 5 мин)
    participant Dedup as TransactionDeduplicator
    participant OS as OrderService
    participant CRM as Integram CRM
    participant Bot as Telegram → Пчеловод

    Note over Poller: Старт: catch-up с 01.01.2024
    Poller->>CRM: get_orders() — загрузить все UDS-* заказы
    CRM-->>Dedup: {UDS-001, UDS-002, ...} → seen set

    loop Polling каждые 5 мин
        Poller->>UDS: get_transactions(limit=50)
        UDS-->>Poller: [tx1, tx2, ...]
        loop Каждая транзакция
            Poller->>Dedup: is_new(tx.id)?
            alt Новая транзакция
                Poller->>CRM: get_or_create_client(phone)
                Poller->>CRM: get_product_by_sku(sku)
                Poller->>OS: create_order(source="UDS")
                OS->>CRM: create_order()
                OS->>Bot: уведомление пчеловоду
                Dedup->>Dedup: mark_seen(tx.id)
            end
        end
    end
```

---

## 9. Память агентов: пять механизмов

```mermaid
graph TB
    subgraph AGENTS["Агенты"]
        CONSULTANT["Консультант"]
        LOGIST["Логист"]
        ADMIN_CHAT["Ассистент"]
        DEVBOT_A["DEVBOT"]
        OTHER["Инспектор\nАналитик\nРаботник"]
    end

    subgraph HOT["HOT tier — локально на VPS"]
        RAM_CTX["SharedContextStore\nДиалог 5 пар\nRAM, TTL 30 мин"]
        SQLITE_MEM["UserMemory (SQLite)\nФакты пользователей\nmemory.db"]
        ANAMNESIS["AnamnesisCache\nФакты + история CRM"]
        ADMIN_RAM["AdminChatAgent._history\n10 пар диалога\nRAM"]
        CRM_SNAP["CrmSnapshot\nRAM кэш CRM\nTTL 5 мин"]
    end

    subgraph COLD["COLD tier — Integram облако"]
        DEV_TASKS[("DEV_TASKS\nЗадачи разработки")]
        DEV_MEM[("DEV_MEMORY\nУроки и антипаттерны")]
        DEV_ADV[("DEV_ADVICE\nСоветы")]
    end

    CONSULTANT --> RAM_CTX & SQLITE_MEM & ANAMNESIS
    LOGIST -.->|"нет памяти"| OTHER
    ADMIN_CHAT --> ADMIN_RAM & CRM_SNAP
    DEVBOT_A --> DEV_TASKS & DEV_MEM & DEV_ADV
    OTHER -.->|"нет памяти"| OTHER

    style HOT fill:#e8f5e9,stroke:#22c55e
    style COLD fill:#e3f2fd,stroke:#1976d2
```

**Разрыв:** только Консультант имеет полноценную память. LangGraph Checkpointer не используется — вся история в RAM теряется при рестарте.

---

## 10. Веб-панель: стек и страницы

```mermaid
graph TB
    subgraph BROWSER["Браузер (Vue 3 + PrimeVue 4)"]
        subgraph VIEWS["14 страниц"]
            DASH["DashboardView\n6 карточек + 4 графика\nТоп-5 + Требуют внимания"]
            ORDERS["OrdersView\nТаблица + Канбан\nBatch-статус + drag&drop"]
            DETAIL["OrderDetailView\nЧеклист + история статусов"]
            NEW["NewOrderView\nСоздание заказа"]
            CLIENTS["ClientsView\nИстория заказов"]
            PRODUCTS["ProductsView\nCRUD каталога"]
            PACKING["PackingView\nOffline PWA"]
            STOCK["StockView\nOffline PWA"]
            OTHER_V["Batches / Monthly / Users / Login"]
        end

        subgraph STORES["Pinia stores"]
            AUTH_S["auth.js\nJWT токен"]
            OFFLINE_S["offline.js\nIndexedDB + sync queue"]
        end

        API_JS["api.js\naxios + JWT interceptor"]
        UTILS_JS["utils.js\nformatDate, formatMoney"]
    end

    subgraph BACKEND["FastAPI :8088"]
        ROUTER_AUTH["POST /auth/login"]
        ROUTER_ORDERS["GET/POST/PUT /api/orders"]
        ROUTER_DASH["GET /api/dashboard/stats\nGET /api/dashboard/alerts"]
        ROUTER_CLIENTS["GET/PUT /api/clients"]
        ROUTER_PRODUCTS["GET/PUT /api/products"]
        ROUTER_SSE["GET /api/events (SSE)"]
        STATIC["Раздача Vue dist/"]
    end

    VIEWS --> API_JS --> BACKEND
    ROUTER_SSE -->|"real-time push"| ORDERS & DASH
    PACKING & STOCK --> OFFLINE_S
    OFFLINE_S -->|"sync при сети"| API_JS

    style BROWSER fill:#f0f9ff
    style BACKEND fill:#e8f5e9,stroke:#22c55e
```

---

## 11. Инфраструктура деплоя

```mermaid
graph TB
    subgraph VPS["VPS 185.233.200.13 (ai-agent)"]
        subgraph DOCKER["Docker Compose (network_mode: host)"]
            BEEBOT_C["beebot\npython -m src.bot\n~762 MiB"]
            REDIS_C["Redis :6379\n~20 MiB"]
        end
    end

    subgraph HIVE["Hive (локальная машина — SPOF)"]
        GROQ_P["groq-proxy.service\n:8990 → api.groq.com"]
        SOCKS_P["tg-socks.service\n:9150 → Telegram API"]
        UDS_PROX["uds_proxy.py\n:8991 → api.uds.app"]
        DEVBOT_S["devbot.service\n:8091"]
    end

    subgraph CLOUD["Облачные сервисы"]
        TG_API["Telegram API"]
        GROQ_API["Groq API\nllama-3.3-70b"]
        CRM_V2_C["Integram v2\nai2o.online"]
        CRM_V1_C["Integram v1\nai2o.ru (архив)"]
        UDS_API_C["UDS Partner API"]
        CDEK_C["СДЭК API v2"]
        POCHTA_C["Почта России"]
        YADISK_C["Яндекс.Диск\n(бэкапы)"]
    end

    subgraph GITHUB["GitHub"]
        REPO["alekseymavai/BEEBOT\nmain branch"]
        CI["GitHub Actions\nruff + mypy + bandit\npytest + deploy"]
    end

    BEEBOT_C -->|SSH tunnel :8990| GROQ_P --> GROQ_API
    BEEBOT_C -->|SOCKS5 :9150| SOCKS_P --> TG_API
    BEEBOT_C -->|HTTPS| CRM_V2_C & CRM_V1_C
    BEEBOT_C -->|через :8991 proxy| UDS_PROX --> UDS_API_C
    BEEBOT_C --> CDEK_C & POCHTA_C & YADISK_C
    BEEBOT_C --> REDIS_C

    CI -->|"git reset --hard origin/main\ndocker compose up -d --build"| VPS

    style VPS fill:#e3f2fd,stroke:#1976d2
    style HIVE fill:#f3e5f5,stroke:#9c27b0
    style GITHUB fill:#e8f5e9,stroke:#22c55e
```

| Сервис | Машина | Порт | RAM | Управление |
|--------|--------|------|-----|-----------|
| beebot | VPS | 8088 | ~762 MiB | docker compose |
| redis | VPS | 6379 | ~20 MiB | docker compose |
| groq-proxy | hive | 8990 | ~10 MiB | systemd |
| tg-socks | hive | 9150 | ~5 MiB | systemd |
| uds_proxy | hive | 8991 | ~5 MiB | manual |
| devbot | hive | 8091 | ~50 MiB | systemd |

---

## 12. CI/CD pipeline

```mermaid
flowchart LR
    PUSH["git push\nto fork/branch"] --> PR["PR →\nalekseymavai/BEEBOT"]
    PR --> CI["GitHub Actions"]

    subgraph CI_STEPS["CI Steps"]
        direction TB
        RUFF["ruff check\n(lint + import order)"]
        MYPY["mypy\n(type check, soft)"]
        BANDIT["bandit\n(security scan)"]
        PYTEST["pytest\n(39 файлов, 8715 строк)"]
    end

    CI --> CI_STEPS
    CI_STEPS -->|"✅ все зелёные"| MERGE["squash-мерж\nв main"]
    MERGE --> DEPLOY["Deploy:\ngit reset --hard origin/main\ndocker compose up -d --build"]

    style CI fill:#f0f9ff
    style MERGE fill:#bbf7d0
    style DEPLOY fill:#e8f5e9,stroke:#22c55e
```

---

## 13. DEVBOT: автономный разработчик

```mermaid
stateDiagram-v2
    [*] --> IDLE : Бот запущен

    IDLE --> ANALYZING : /dev <задача>
    ANALYZING --> IDLE : Ошибка анализа

    ANALYZING --> CONFIRMING : Claude API → план
    CONFIRMING --> IDLE : Пчеловод отклонил

    CONFIRMING --> EXECUTING : Подтверждение
    EXECUTING --> EXECUTING : --resume (auto-continue)
    EXECUTING --> FEEDBACK : Выполнено

    FEEDBACK --> IDLE : Результат сохранён в DEV_MEMORY
```

**Цепочка:** `/dev задача` (Telegram) → HTTP POST → DEVBOT API (hive:8091) → Claude API (analyzer) → plan → Anthropic CLI executor → git push → deploy.

---

## 14. Поток события: заказ создан → браузер обновился

```mermaid
sequenceDiagram
    participant API as FastAPI\n(POST /api/orders)
    participant OS as OrderService
    participant CRM as Integram CRM
    participant Notify as NotificationService
    participant SSE as SSE Bridge
    participant Browser as Vue PWA
    participant TG as Telegram

    API->>OS: create_order(data)
    OS->>CRM: create_order()
    CRM-->>OS: order_id = 1916
    OS->>Notify: notify_beekeeper("Новый заказ #1916")
    Notify->>TG: send_message(ADMIN_CHAT_ID)
    OS->>SSE: emit("order.created", {id: 1916})
    SSE-->>Browser: EventSource message
    Browser->>Browser: invalidate orders cache
    Browser->>API: GET /api/orders (auto-refresh)
    API-->>Browser: [updated list]
```

---

## 15. Gift Protocol: передача контекста между агентами

```mermaid
graph LR
    subgraph USER["Пользователь отправляет сообщение"]
        MSG["сообщение"]
    end

    subgraph GIFT["GiftBroker (gift_protocol.py)"]
        ASSEMBLE["Собрать контекст:\n- UserMemory (SQLite факты)\n- SharedContextStore (диалог)\n- CrmSnapshot (заказы клиента)\n- AnamnesisCache"]
        CONTEXT["SharedContext\n(user_profile + episodes\n+ crm_data + dialog)"]
    end

    subgraph ORCH["Оркестратор"]
        CLASSIFY2["Классификация intent"]
        BEEBOT_G["BeebotAgent\n(получает контекст)"]
        LOGIST_G["LogistAgent\n(получает контекст)"]
    end

    MSG --> ASSEMBLE --> CONTEXT --> CLASSIFY2
    CLASSIFY2 --> BEEBOT_G
    CLASSIFY2 --> LOGIST_G

    style GIFT fill:#f3e5f5,stroke:#9c27b0
```

---

## 16. Структура: четыре слоя

```mermaid
graph TB
    subgraph L1["1. Транспорт (вход/выход)"]
        TG_L["bot.py + 7 роутеров\n(Telegram)"]
        WEB_L["api.py + 9 роутеров\n(FastAPI)"]
    end

    subgraph L2["2. Агенты (координация)"]
        ORCH_L["Orchestrator\n(LangGraph)"]
        AG6["6 агентов\n(beebot, logist, analyst,\ninspector, admin_chat, worker)"]
    end

    subgraph L3["3. Service Layer (бизнес-логика)"]
        OS_L["OrderService\nCRUD заказов + уведомления"]
        NS_L["NotificationService\nTelegram push"]
    end

    subgraph L4["4. Инфраструктура (данные)"]
        CRM_L["IntegramV2Client\n(CRM v2, ai2o.online)"]
        KB_L["FAISS KB\n(276 чанков)"]
        LLM_L["Groq LLM\n(llama-3.3-70b)"]
        MEM_L["SQLite\n(UserMemory)"]
        DEL_L["delivery/\n(СДЭК + Почта)"]
    end

    L1 --> L2
    L2 --> L3
    L3 --> L4
    L1 -->|"веб-роутеры"| L3

    style L1 fill:#e3f2fd
    style L2 fill:#fff3e0
    style L3 fill:#e8f5e9,stroke:#22c55e
    style L4 fill:#f5f5f5
```

---

## Сравнительные таблицы

### Агенты

| Агент | Файл | Строк | Триггер | Сервис | Состояние |
|-------|------|-------|---------|--------|-----------|
| BeebotAgent | beebot.py | 134 | consult intent | FAISS + LLM | ✅ RAM (SharedContext) |
| LogistAgent | logist.py | 491 | /order FSM | OrderService | ✅ FSM state |
| AnalystAgent | analyst.py | 56 | stats intent | orchestrator | ⚠️ Stub |
| InspectorAgent | inspector.py | 158 | /inspect FSM | FAISS + LLM | ✅ FSM state |
| AdminChatAgent | admin_chat.py | 282 | /admin mode | CrmSnapshot | ⚠️ Двойная загрузка |
| WorkerAgent | worker.py | 169 | WORKER_CHAT_IDS | CRM напрямую | ⚠️ RAM (теряется) |

### Веб-роутеры

| Роутер | Методы | Строк | CRM операции |
|--------|--------|-------|-------------|
| orders.py | GET/POST/PUT/batch | 441 | list, get, create, update, batch |
| dashboard.py | GET stats, alerts | 84 | stats, alerts, top products |
| clients.py | GET/PUT | 169 | list, get, merge |
| products.py | GET/PUT | 123 | list, get, update |
| batches.py | GET/POST/PUT | 114 | list, create, ship |
| auth.py | POST login | 67 | — |
| export.py | GET csv | 114 | list orders |
| report.py | GET pdf | 124 | list orders |
| sse.py | GET events | ~50 | — |

---

*Связанные документы: [analysis.md](../analysis.md) | [plan.md](../plan.md) | [README.md](../README.md)*
