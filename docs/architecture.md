# BEEBOT — Архитектурные диаграммы

> **Версия:** 2 апреля 2026
> **Ключевое изменение:** Hexagonal Architecture (steps 0–5)

---

## 0. Было → Стало: общий обзор

### БЫЛО: два монолита с общей CRM

```mermaid
graph LR
    subgraph BOT["beebot (Docker)"]
        B_BOT[Telegram-бот<br/>1900 строк]
        B_AGENTS[5 агентов]
        B_CRM1[CRM-клиент]
        B_LLM[LLM-клиент]
        B_KB[FAISS + Ontology]
        B_MEM[SQLite память]
        B_BG[Tracker · UDS · Backup]
    end

    subgraph WEB["beebot-web (Docker)"]
        W_API[FastAPI роуты]
        W_CRM2[CRM-клиент]
        W_VUE[Vue 3 PWA]
    end

    CRM[(Integram CRM)]

    B_CRM1 --> CRM
    W_CRM2 --> CRM

    style B_CRM1 fill:#fee2e2
    style W_CRM2 fill:#fee2e2
```

**Проблемы:** бизнес-логика в двух местах, CRM доступ отовсюду, бот = толстый монолит, нет общения между процессами.

### СТАЛО: три процесса + Redis Streams + Service Layer

```mermaid
graph TB
    subgraph BOT["beebot (тонкий клиент)"]
        T_BOT[Telegram handlers<br/>UI + FSM]
        T_CLIENT[BotServiceClient<br/>publish/subscribe]
    end

    REDIS[(Redis Streams)]

    subgraph BACKEND["beebot-backend"]
        S_BUS[BusHandlers<br/>маршрутизация событий]
        S_ORDER[OrderService]
        S_CONSULT[ConsultService]
        S_NOTIFY[NotificationService]
        S_BG[BackgroundTaskManager<br/>Tracker · UDS · Backup]
        I_CRM[CRM-адаптер<br/>единственный]
        I_LLM[LLM-адаптер]
        I_KB[KB-адаптер]
    end

    subgraph WEB_PANEL["Vue 3 PWA"]
        W_VUE[Веб-панель :8088]
    end

    CRM[(Integram CRM)]

    T_BOT --> T_CLIENT
    T_CLIENT -->|request| REDIS
    REDIS -->|response| T_CLIENT
    REDIS -->|events| T_CLIENT

    REDIS --> S_BUS
    S_BUS --> S_ORDER & S_CONSULT

    W_VUE -->|REST/JWT| S_ORDER

    S_ORDER --> I_CRM
    S_ORDER --> S_NOTIFY
    S_CONSULT --> I_LLM & I_KB
    S_BG --> I_CRM

    I_CRM --> CRM

    style I_CRM fill:#bbf7d0
    style REDIS fill:#bfdbfe
```

**Что изменилось:**
- Бот не знает про CRM/LLM/KB — только кнопки + Redis
- Один OrderService для бота, веба и UDS
- CRM доступ только из infrastructure/crm/
- Фоновые задачи с авто-рестартом и мониторингом

---

## 1. Структура файлов: три слоя

```mermaid
graph TB
    subgraph TRANSPORT["transport/ — тонкие адаптеры входа"]
        TG[telegram/<br/>bot.py · bot_client.py<br/>handlers/]
        WEB[web/<br/>app.py · routers/<br/>bus_handlers.py · bg_tasks.py]
    end

    subgraph SERVICES["services/ — бизнес-логика"]
        ORD[OrderService<br/>создание · статусы · позиции]
        CON[ConsultService<br/>KB + LLM]
        NOT[NotificationService<br/>клиент · пчеловод · работники]
        ANA[AnalyticsService]
        WRK[WorkerService]
        DEL[DeliveryService]
    end

    subgraph INFRA["infrastructure/ — адаптеры выхода"]
        CRM_I[crm/<br/>integram_api · integram_client<br/>constants · snapshot]
        LLM_I[llm/<br/>groq_client]
        KB_I[kb/<br/>knowledge_base · ontology]
        MEM_I[memory/<br/>sqlite_memory]
        DEL_I[delivery/<br/>cdek · pochta · tracker]
    end

    subgraph DOMAIN["domain/ — модели и правила"]
        MOD[models.py<br/>Order · Client · Product]
        EVT[events.py<br/>OrderCreated · StatusChanged]
        EXC[exceptions.py<br/>CRMUnavailable · InvalidStatus]
    end

    TG -->|вызывает| SERVICES
    WEB -->|вызывает| SERVICES
    SERVICES -->|использует| INFRA
    SERVICES -->|оперирует| DOMAIN
    INFRA -->|оперирует| DOMAIN

    style TRANSPORT fill:#e3f2fd
    style SERVICES fill:#e8f5e9
    style INFRA fill:#fff3e0
    style DOMAIN fill:#f3e5f5
```

---

## 2. Redis Streams: протокол событий

### Bot → Backend (запросы)

```mermaid
sequenceDiagram
    participant Bot as Бот (тонкий клиент)
    participant Redis as Redis Streams
    participant Backend as Backend (сервисы)

    Bot->>Redis: publish stream:requests<br/>{type: "consult", payload: {query, user_id}}
    Redis->>Backend: consume (consumer group: backend)
    Backend->>Backend: ConsultService.answer()
    Backend->>Redis: publish replies:{correlation_id}
    Redis->>Bot: read response
    Bot->>Bot: отправить ответ пользователю
```

### Backend → Bot (события)

```mermaid
sequenceDiagram
    participant Backend as Backend
    participant Redis as Redis Streams
    participant Bot as Бот

    Note over Backend: Tracker обнаружил: заказ доставлен
    Backend->>Redis: publish stream:events<br/>{type: "order_status_changed"}
    Redis->>Bot: consume (consumer group: bot)
    Bot->>Bot: отправить уведомление клиенту
```

### Типы событий

| Направление | type | payload |
|---|---|---|
| Bot → Backend | `consult` | user_id, query, history, style |
| Bot → Backend | `create_order` | client_id, items, delivery |
| Bot → Backend | `update_order_status` | order_id, status, role |
| Bot → Backend | `get_orders` | client_id, status |
| Bot → Backend | `analytics_query` | query, admin_id |
| Bot → Backend | `ping` | — |
| Backend → Bot | `order_status_changed` | order_id, status, client_tg_id |
| Backend → Bot | `delivery_update` | order_id, tracking_status |
| Backend → Bot | `new_order_from_web` | order_id, order_number |

---

## 3. OrderService: единый источник правды

```mermaid
graph TB
    subgraph CALLERS["Кто вызывает"]
        C1[Telegram бот<br/>через Redis]
        C2[Веб-панель<br/>через FastAPI]
        C3[UDS Poller<br/>фоновая задача]
    end

    subgraph SERVICE["OrderService"]
        CREATE[create_order<br/>+ create_order_with_client]
        STATUS[update_status<br/>валидация + история]
        ITEMS[add_item · update_item<br/>delete_item · recalculate]
        READ[get_orders · get_order<br/>get_order_items]
    end

    subgraph DEPS["Зависимости"]
        CRM_DEP[IntegramClient<br/>через DI]
        NOTIFY_DEP[NotificationService<br/>через DI]
    end

    C1 --> CREATE & STATUS & READ
    C2 --> CREATE & STATUS & ITEMS & READ
    C3 --> CREATE

    CREATE --> CRM_DEP & NOTIFY_DEP
    STATUS --> CRM_DEP & NOTIFY_DEP
    ITEMS --> CRM_DEP

    style SERVICE fill:#e8f5e9
```

**Было:** 3 разных реализации создания заказа (logist.py, orders.py, uds.py).
**Стало:** один `OrderService.create_order()` с единой логикой уведомлений.

---

## 4. BackgroundTaskManager

```mermaid
graph TB
    MGR[BackgroundTaskManager]

    MGR --> T1[CRM Snapshot<br/>каждые 5 мин]
    MGR --> T2[OrderTracker<br/>каждые 2 часа]
    MGR --> T3[UDS Poller<br/>каждые 5 мин]
    MGR --> T4[TunnelMonitor<br/>каждые 60 сек]
    MGR --> T5[BackupManager<br/>ежедневно]

    T1 & T2 & T3 & T4 & T5 -->|crash| MGR
    MGR -->|auto-restart<br/>+ алерт пчеловоду| T1 & T2 & T3 & T4 & T5

    HEALTH[GET /api/health] --> MGR
    MGR -->|status()| HEALTH
```

**Было:** `asyncio.create_task()` — fire-and-forget, падение незаметно.
**Стало:** авто-рестарт при падении, экспоненциальный backoff, алерты, `/api/health`.

---

## 5. Docker: три контейнера

```mermaid
graph LR
    subgraph DOCKER["docker-compose.yml"]
        REDIS_C[redis:7-alpine<br/>~20 MB RAM]
        BOT_C[beebot<br/>aiogram + redis-py<br/>~50 MB RAM]
        BACKEND_C[beebot-backend<br/>FastAPI + FAISS + Groq<br/>~700 MB RAM]
    end

    BOT_C -->|streams| REDIS_C
    BACKEND_C -->|streams| REDIS_C
    BOT_C -.->|fallback при Redis down| BOT_C

    style REDIS_C fill:#bfdbfe
```

**Независимый деплой:**
- `docker compose stop beebot` → веб-панель работает
- `docker compose stop beebot-backend` → бот отвечает «Сервис недоступен»

---

## 6. CRM: схема данных

```mermaid
erDiagram
    CLIENTS {
        int id PK
        string full_name
        string phone
        int telegram_id
        string city
        string address
    }

    ORDERS {
        int id PK
        string number
        int client_id FK
        string status
        string source
        string delivery_method
        float total
        datetime date
        string tracking
        int batch_id FK
    }

    ORDER_ITEMS {
        int id PK
        int order_id FK
        int product_id FK
        int quantity
        float unit_price
    }

    PRODUCTS {
        int id PK
        string name
        float price
        int stock
        string category
        string sku_uds
    }

    STATUS_HISTORY {
        int id PK
        int order_id FK
        string status_from
        string status_to
        datetime date
    }

    HEALTH_PROFILE {
        int id PK
        int client_id FK
        string symptom
        string source_text
    }

    CLIENTS ||--o{ ORDERS : "размещает"
    ORDERS ||--o{ ORDER_ITEMS : "содержит"
    PRODUCTS ||--o{ ORDER_ITEMS : "включён в"
    ORDERS ||--o{ STATUS_HISTORY : "история"
    CLIENTS ||--o{ HEALTH_PROFILE : "здоровье"
```

---

## 7. Поток запроса: Telegram → ответ

```mermaid
flowchart TD
    A[Сообщение Telegram] --> B{FSM-состояние?}
    B -->|Да: OrderFSM| C[Шаг диалога заказа]
    B -->|Да: InspectFSM| D[Шаг осмотра улья]
    B -->|Нет| E{Режим?}

    E -->|WORKER| F[Очередь сборки]
    E -->|ADMIN /admin| G[Ассистент + CrmSnapshot]
    E -->|Обычный| H[Orchestrator]

    H --> I{Быстрая классификация}
    I -->|greeting| J[Приветствие]
    I -->|order/edit/track| K[Запуск FSM / меню]
    I -->|stats| L[Аналитик]
    I -->|None → LLM| M[Определение интента]
    M --> N[Консультант: FAISS → LLM]

    N --> O[Ответ пользователю]
    C & D & F & G & J & K & L --> O
```

---

## 8. Агенты: сравнительная таблица

| Агент | KB | CRM | LLM | Вход | Особенности |
|---|---|---|---|---|---|
| Консультант | FAISS | — | Groq | consult | Голос Улья (5 стилей) |
| Логист | — | Запись | Groq | order | FSM 7 шагов |
| Аналитик | — | Чтение | Groq | stats | ABC, сезонность, прогноз |
| Инспектор | FAISS | — | Groq | /inspect | 3 вопроса → рекомендация |
| Ассистент | — | CrmSnapshot | Groq | /admin | Свободный диалог |
| Worker | — | Чтение+Запись | — | /start | inbox + DEFERRED |
| CrmAgent | — | Единственный | — | внутренний | Через GiftBroker |
| DEVBOT | — | DEV-таблицы | Claude | /dev | Только hive |

---

## 9. Источники заказов

| Источник | Через что попадает | Позиции | Уведомления |
|---|---|---|---|
| Telegram (FSM) | OrderService.create_order_with_client | Полные | Пчеловод + работники |
| UDS-магазин | OrderService.create_order | По sku_uds | Пчеловод + работники |
| Веб-панель | OrderService.create_order | Ручной ввод | Пчеловод + работники |
| ВК / Instagram | Ручной ввод через веб | Ручной | — |

---

## 10. Инфраструктура: туннели

```mermaid
graph LR
    subgraph VPS["VPS 185.233.200.13"]
        BOT[beebot]
        WEB[beebot-backend :8088]
        REDIS[Redis :6379]
    end

    subgraph HIVE["Hive"]
        PROXY[groq-proxy :8990]
        SOCKS[SOCKS5 :9150]
        DEVBOT[DEVBOT :8091]
    end

    subgraph EXT["Внешние API"]
        GROQ[Groq API]
        TG[Telegram API]
        CRM_EXT[(Integram CRM)]
        CDEK[СДЭК]
        POCHTA[Почта]
    end

    BOT -->|SSH tunnel| PROXY --> GROQ
    BOT -->|SOCKS5| SOCKS --> TG
    BOT & WEB --> CRM_EXT
    WEB --> CDEK & POCHTA
```

---

*Связанные документы: [analysis.md](../analysis.md) · [plan.md](../plan.md)*
