# BEEBOT — Архитектурные диаграммы

> **Версия:** 31 марта 2026 · Добавлен раздел 11: Новые компоненты фаз 9–12

---

## 1. Общая архитектура системы

```mermaid
graph TB
    subgraph Пользователи
        TG_USER[📱 Подписчики<br/>Telegram]
        WORKER[🏭 Работник склада<br/>Telegram]
        ADMIN[🐝 Пчеловод<br/>Telegram + Веб]
        DEV[👨‍💻 Разработчик<br/>DEVBOT + Claude Code]
    end

    subgraph VPS["VPS 185.233.200.13 (Docker)"]
        BOT[🤖 Telegram-бот<br/>aiogram 3 · Router-модули]
        WEB_API[🌐 FastAPI<br/>REST API + JWT + SSE]
        WEB_FRONT[💻 Vue 3 PWA<br/>порт 8088]
        TRACKER[⏱ Авто-трекинг<br/>каждые 2 часа]
        UDS_POLL[🔄 UDS Poller<br/>каждые 5 минут]
        CRM_SNAP[📷 CrmSnapshot<br/>кэш 5 мин + low-stock алерт]
        TUNNEL_MON[🔌 TunnelMonitor<br/>порт 8990 · 60 сек]
        BACKUP_MGR[💾 BackupManager<br/>daily + weekly · Яндекс Диск]
    end

    subgraph Агенты["Агенты (src/agents/)"]
        ORCH[🧠 Оркестратор<br/>LangGraph StateGraph]
        CONSUL[🐝 Консультант<br/>FAISS → LLM + FAQ fallback]
        LOGIST[📦 Логист<br/>FSM 7 шагов]
        ANALYST[📊 Аналитик<br/>CRM → отчёты]
        INSPECT[🔍 Инспектор<br/>Осмотр улья]
        ADMINCHAT[🤖 Ассистент<br/>LLM + CrmSnapshot]
        WORKER_AG[🏭 WorkerAgent<br/>очередь + inbox + DEFERRED]
    end

    subgraph GiftLayer["Gift Protocol"]
        GIFT_BROKER[GiftBroker<br/>suggest_interface]
        SHARED_CTX[(SharedContext<br/>per-user · TTL 30мин)]
        ANAMNESIS_C[AnamnesisCache<br/>SQLite + CRM история]
        CRM_AGENT[CrmAgent<br/>единый владелец CRM]
        AGENT_BUS[AgentBus Client<br/>dronedoc2026 · опц.]
    end

    subgraph KB["База знаний"]
        FAISS[(FAISS<br/>276 чанков + comment:×1.2)]
        DOCS[📄 PDF / TXT<br/>26 YouTube + Q&A комментарии]
        MEMORY[(SQLite<br/>Факты пользователей)]
        ONTOLOGY[(Integram<br/>74 симптома + 77 показаний)]
    end

    subgraph External["Внешние сервисы"]
        GROQ[⚡ Groq API<br/>llama-3.3-70b]
        CRM[(Integram CRM<br/>ai2o.ru/bibot)]
        CDEK[🚚 СДЭК API v2]
        POCHTA[📮 Почта России]
        UDS_API[💳 UDS API<br/>система лояльности]
        TG_API[💬 Telegram API]
    end

    subgraph HIVE["Hive (локальная машина)"]
        PROXY[groq-proxy<br/>порт 8990]
        TG_SOCKS[SOCKS5-прокси<br/>порт 9150]
        DEVBOT[🤖 DEVBOT<br/>порт 8091 · Bearer auth]
        CLAUDE[⚡ Claude Code CLI<br/>executor]
    end

    subgraph Backup["Резервные копии"]
        YADISK[(Яндекс Диск<br/>/BEEBOT/daily/ · weekly/)]
    end

    TG_USER -->|сообщения| TG_API
    WORKER -->|/start очередь| TG_API
    ADMIN -->|/admin /stats| TG_API
    ADMIN -->|веб-панель :8088| WEB_FRONT
    DEV -->|/dev задача| TG_API

    TG_API -->|polling SOCKS5| BOT
    BOT --> ORCH
    BOT --> INSPECT
    BOT --> WORKER_AG
    BOT --> ADMINCHAT
    ADMINCHAT --> CRM_SNAP

    ORCH -->|consult| CONSUL
    ORCH -->|order| LOGIST
    ORCH -->|stats| ANALYST
    ORCH -->|greeting| BOT

    CONSUL --> FAISS
    CONSUL --> GROQ

    LOGIST --> CRM
    LOGIST --> CDEK
    LOGIST --> POCHTA

    ANALYST --> CRM

    WORKER_AG --> CRM

    CRM_SNAP --> CRM

    FAISS --> DOCS
    BOT --> MEMORY
    BOT --> ONTOLOGY

    WEB_FRONT -->|REST/JWT| WEB_API
    WEB_API --> CRM
    WEB_API -->|SSE| WEB_FRONT

    TRACKER --> CRM
    TRACKER --> CDEK
    TRACKER --> POCHTA
    UDS_POLL --> UDS_API
    UDS_POLL --> CRM

    BOT --> GIFT_BROKER
    GIFT_BROKER <--> SHARED_CTX
    GIFT_BROKER --> ANAMNESIS_C
    ANAMNESIS_C --> CRM_AGENT
    CRM_AGENT --> CRM

    BOT -->|DEVBOT_API_URL Bearer| DEVBOT
    DEVBOT --> CLAUDE
    CLAUDE -->|git + deploy| VPS

    BOT --> TUNNEL_MON
    TUNNEL_MON -->|is_healthy| CONSUL

    BACKUP_MGR -->|memory.db + CRM CSV| YADISK
    BACKUP_MGR --> MEMORY

    BOT --> AGENT_BUS

    BOT -->|SOCKS5| TG_SOCKS
    GROQ -->|groq-proxy:8990| PROXY
    TG_SOCKS -->|SSH -R 9150| VPS
    PROXY -->|SSH -L 8990| VPS
```

---

## 2. Поток запроса через оркестратор

```mermaid
flowchart TD
    A[📱 Сообщение Telegram] --> B{Это FSM-состояние?}
    B -->|Да — LogistFSM| C[LogistAgent FSM<br/>шаг диалога]
    B -->|Да — InspectFSM| D[InspectorAgent FSM<br/>шаг диалога]
    B -->|Нет| E{Режим пользователя?}

    E -->|WORKER| F[WorkerAgent<br/>очередь сборки]
    E -->|ADMIN /admin| G[AdminChatAgent<br/>CrmSnapshot → LLM]
    E -->|Обычный| H[Orchestrator.route]

    H --> I{_fast_classify}
    I -->|greeting/order/edit/track| J[Немедленный результат<br/>без LLM]
    I -->|None| K[_classify_intent<br/>Groq ~5 токенов]

    J --> L{Intent}
    K --> L

    L -->|consult| M[BeebotAgent<br/>FAISS search → LLM]
    L -->|order| N[passthrough →<br/>bot.py запускает FSM]
    L -->|edit| O[passthrough →<br/>bot.py показывает меню]
    L -->|track| P[passthrough →<br/>bot.py ищет трек]
    L -->|stats| Q[AnalystAgent<br/>CRM → отчёт]
    L -->|greeting| R[Приветствие по имени]

    M --> S[Загрузить SQLite факты<br/>+ онтологическую рекомендацию]
    S --> T[LLM ответ в Голосе Улья]
    T --> U[FAQ-счётчик++<br/>extract_fact → SQLite]
    U --> V[📤 Ответ пользователю]

    C --> V
    D --> V
    F --> V
    G --> V
    N --> V
    O --> V
    P --> V
    Q --> V
    R --> V
```

---

## 3. FSM оформления заказа (LogistAgent)

```mermaid
stateDiagram-v2
    [*] --> select_products : /order или intent=order

    select_products --> enter_name : выбраны товары
    select_products --> [*] : /cancel

    enter_name --> enter_phone : введено ФИО
    enter_name --> [*] : /cancel

    enter_phone --> confirm_phone : телефон введён
    confirm_phone --> enter_address : ✅ Верно
    confirm_phone --> enter_phone : ❌ Изменить

    enter_address --> select_delivery : введён адрес
    enter_address --> [*] : /cancel

    select_delivery --> confirm_order : выбран способ (СДЭК/Почта/Самовывоз)
    select_delivery --> [*] : /cancel

    confirm_order --> [*] : ✅ Подтвердить → CRM + уведомление пчеловоду
    confirm_order --> select_products : 🔄 Изменить
    confirm_order --> [*] : ❌ Отменить

    note right of select_delivery
        СДЭК: OAuth2 → тарификация
        Почта: tariff.pochta.ru
        Самовывоз: бесплатно
    end note

    note right of confirm_order
        CRM: create_order() + create_order_items()
        Уведомления:
        - пчеловоду (Notifier.new_order)
        - работникам (notify_workers_new_order)
        - группам (ACTIVE_GROUP_IDS)
    end note
```

---

## 4. Поток UDS → CRM

```mermaid
flowchart TD
    A[UDS Poller старт] --> B[load_existing_from_crm<br/>загрузить UDS-* заказы из CRM]
    B --> C[_initial_sync<br/>cursor-пагинация с 01.01.2024]
    C --> D[Бесконечный polling<br/>каждые 5 мин]

    D --> E[get_transactions limit=50]
    E --> F{is_new?}
    F -->|дубль| D
    F -->|новая| G[sync_uds_transaction]

    G --> H[_get_or_create_client_by_phone<br/>поиск/создание клиента]
    H --> I[_build_order_items<br/>⚠️ по имени товара, не SKU]
    I --> J[create_order source=UDS]
    J --> K{notify?}

    K -->|исторический catch-up| L[Без уведомлений]
    K -->|новый в prod| M[notify_beekeeper<br/>+ notify_workers_new_order<br/>+ ACTIVE_GROUP_IDS]

    M --> N[mark_seen]
    L --> N
    N --> D
```

---

## 5. DEVBOT — жизненный цикл задачи

```mermaid
stateDiagram-v2
    [*] --> IDLE

    IDLE --> ANALYZING : /dev <задача>

    ANALYZING --> CONFIRMING : analyze_task() — Claude API<br/>+ dev_memory + советы
    ANALYZING --> IDLE : ошибка анализа

    CONFIRMING --> EXECUTING : ✅ Выполнить
    CONFIRMING --> ANALYZING : уточнение → пересчёт плана
    CONFIRMING --> IDLE : ❌ Отменить

    EXECUTING --> FEEDBACK : exit_code=0<br/>record_completion() в Integram
    EXECUTING --> IDLE : exit_code≠0 → отчёт об ошибке

    FEEDBACK --> EXECUTING : фидбек → feedback loop
    FEEDBACK --> IDLE : /ok или таймаут 10 мин

    note right of ANALYZING
        analyzer.py:
        Claude API → план изменений
        + оценка сложности
        + риски
    end note

    note right of EXECUTING
        executor.py:
        claude --output-format stream-json
        --model claude-sonnet-4.6
        --append-system-prompt rules
        auto-continue через --resume
    end note

    note right of FEEDBACK
        Память (2 уровня):
        1. Integram DEV_TASKS + DEV_MEMORY
        2. .claude/memory/ файлы
    end note
```

---

## 6. WorkerAgent — поток сборки заказов

```mermaid
flowchart TD
    A[Работник: /start] --> B{WORKER_CHAT_IDS?}
    B -->|не в списке| C[Обычный /start → приветствие]
    B -->|в списке| D[get_worker_queue<br/>статусы: Новый/Подтверждён/В сборке]

    D --> E[build_queue_keyboard<br/>список заказов]
    E --> F[Нажать заказ]

    F --> G[format_order_card<br/>+ build_order_keyboard]
    G --> H{Статус заказа}

    H -->|Новый/Подтверждён| I[Показать состав<br/>кнопка: Взять в работу]
    H -->|В сборке| J[Чеклист позиций<br/>toggle каждой позиции]

    I --> K[update_order_status → В сборке<br/>⚠️ _checked в RAM]
    K --> J

    J --> L{Все позиции отмечены?}
    L -->|Нет| J
    L -->|Да| M[Кнопка: Заказ собран!]

    M --> N[update_order_status → Подтверждён<br/>notify_workers_assembled → пчеловоду]
    N --> D

    subgraph Push-уведомления
        P[Новый заказ из TG/UDS/Веб] --> Q[notify_workers_new_order<br/>все WORKER_CHAT_IDS]
        Q --> R[кнопка Открыть очередь]
    end
```

---

## 7. Инфраструктура: туннели и прокси

```mermaid
graph LR
    subgraph VPS["VPS 185.233.200.13"]
        BOT_DOCKER[beebot container<br/>network_mode: host]
        WEB_DOCKER[beebot-web<br/>:8088]
    end

    subgraph HIVE["Hive (локальная машина)"]
        TG_SOCKS_SVC[tg_socks_proxy.py<br/>SOCKS5 :9150]
        GROQ_PROXY_SVC[groq_proxy.py<br/>HTTP-прокси :8990]
        DEVBOT_SVC[devbot<br/>FastAPI :8091]
    end

    subgraph External["Внешние API"]
        TG_API[Telegram API<br/>api.telegram.org]
        GROQ_API[Groq API<br/>api.groq.com]
    end

    BOT_DOCKER -->|SOCKS5 9150| TG_SOCKS_SVC
    TG_SOCKS_SVC -->|HTTPS| TG_API

    BOT_DOCKER -->|HTTP :8990| GROQ_PROXY_SVC
    GROQ_PROXY_SVC -->|HTTPS| GROQ_API

    BOT_DOCKER -->|HTTP localhost:8091<br/>SSH-туннель| DEVBOT_SVC

    subgraph systemd["systemd (hive)"]
        GROQ_TUNNEL[groq-tunnel.service<br/>SSH -L 8990:VPS -R 9150:hive]
        TG_SOCKS_SVC_SD[tg-socks.service]
        GROQ_PROXY_SD[groq-proxy.service]
    end
```

---

## 8. CRM: схема данных Integram

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
        string comment
    }

    HEALTH_PROFILE {
        int id PK
        int client_id FK
        string symptom
        string source_text
        datetime date
    }

    DEV_TASKS {
        int id PK
        string description
        string status
        string pr_url
        string sha
        string lessons
    }

    DEV_MEMORY {
        int id PK
        string topic
        string context
        string solution
        string files
        string antipattern
    }

    CLIENTS ||--o{ ORDERS : "размещает"
    ORDERS ||--o{ ORDER_ITEMS : "содержит"
    PRODUCTS ||--o{ ORDER_ITEMS : "включён в"
    ORDERS ||--o{ STATUS_HISTORY : "история"
    CLIENTS ||--o{ HEALTH_PROFILE : "профиль здоровья"
```

---

## 9. Сравнительные таблицы

### 9.1 Агенты: возможности и ограничения

| Агент | Доступ к KB | Доступ к CRM | Интент | Ограничения |
|-------|-------------|--------------|--------|-------------|
| Консультант (Beebot) | ✅ FAISS + comment:×1.2 | ❌ | consult | FAQ fallback при is_healthy=False |
| Логист | ❌ | ✅ Запись (через CrmAgent) | order | Предзаполняет адрес из истории |
| Аналитик | ❌ | ✅ Чтение | stats | Только ADMIN_CHAT_ID |
| Инспектор | ✅ FAISS поиск | ❌ | /inspect + оркестратор | inspect-интент в оркестраторе |
| Ассистент пчеловода | ❌ | ✅ CrmSnapshot | /admin | Только ADMIN_CHAT_ID |
| WorkerAgent | ❌ | ✅ Чтение+Запись | /start (worker) | inbox + DEFERRED · suggest_interface() |
| CrmAgent | ❌ | ✅ Единственный владелец | внутренний | Через GiftBroker |
| DEVBOT | ❌ | ✅ DEV-таблицы | /dev | Только hive · Bearer auth |

### 9.2 Источники заказов и их обработка

| Источник | Как попадает в CRM | Позиции | Клиент | Уведомления |
|----------|-------------------|---------|--------|-------------|
| Telegram-бот (FSM) | LogistAgent.confirm → CRM | ✅ Полные | Telegram ID | Пчеловод + Работники |
| UDS система лояльности | UDSPoller → sync_uds_transaction | ⚠️ По имени (SKU=0) | Телефон (telegram_id=0) | Пчеловод + Работники |
| Веб-панель | POST /api/orders | ✅ Ручной ввод | Выбор из CRM | Пчеловод + Работники |
| WhatsApp/ВК/Instagram | Ручной ввод | Ручной ввод | Ручной ввод | Нет |
| DEVBOT (/dev) | Нет — это задачи разработки | — | — | — |

### 9.3 Способы доставки

| Способ | Расчёт стоимости | Трекинг | Авто-уведомление |
|--------|-----------------|---------|-----------------|
| СДЭК | OAuth2 + tariff API | ✅ По трек-номеру | ✅ Каждые 2 ч |
| Почта России | tariff.pochta.ru | ✅ По трек-номеру | ✅ Каждые 2 ч |
| Самовывоз | 0 ₽ (фиксированно) | ❌ | ❌ |

### 9.4 Права доступа

| Роль | Как определяется | Возможности |
|------|-----------------|------------|
| Пользователь | Любой chat_id | consult, /order, /products, /inspect, /voice, /help |
| Работник | `WORKER_CHAT_IDS` в .env | Очередь сборки (/start), статусы В сборке |
| Администратор | `ADMIN_CHAT_ID` / `ADMIN_IDS` | /admin, /stats, /faq, /yt_check, /yt_update, /status, /orders, /clients |
| Веб-пользователь | JWT (WEB_PASSWORD) | Полная веб-панель |

### 9.5 LLM использование по компонентам

| Компонент | LLM-вызов | Токенов/вызов | Цель |
|-----------|-----------|---------------|------|
| Classify intent | Groq | ~100 | Маршрутизация запроса |
| Консультант | Groq | ~2000 | Ответ в стиле автора |
| AdminChatAgent | Groq | ~4000 | Диалог с пчеловодом |
| Аналитик | Groq | ~3000 | Анализ CRM-статистики |
| DEVBOT Analyzer | Claude API (Sonnet) | ~2000 | Анализ задачи → план |
| DEVBOT Executor | Claude Code CLI | ~20000+ | Реализация задачи |

---

*Связанные документы: [analysis.md](../analysis.md) · [plan.md](../plan.md)*

---

## 11. Новые компоненты (фазы 9–12, 31.03.2026)

> Добавлены на основе реализованных фаз. Отражают **текущее состояние** production-кода.

### 11.1 Gift Protocol: SharedContext + GiftBroker + AnamnesisCache

```mermaid
graph TB
    subgraph GiftLayer["Gift Protocol (src/)"]
        SC[("SharedContext\n─────────────\nUserContext per user:\n· history (5 пар)\n· interface_mode\n· TTL 30 мин")]
        BROKER[GiftBroker\n─────────────\nsend() → собирает анамнез\n→ orchestrator.route()\n→ обновляет SharedContext\nsuggests_interface()]
        ANAMNESIS[AnamnesisCache\n─────────────\nSQLite факты\n+ история заказов CRM\nformat_for_llm()]
        CRMAGENT[CrmAgent\n─────────────\nget_client_by_tg_id()\nget_orders_for_user()\nadd_health_fact()\nget_products()]
    end

    MSG[📱 Сообщение] --> BROKER
    BROKER <--> SC
    BROKER --> ANAMNESIS
    ANAMNESIS --> CRMAGENT
    BROKER --> ORCHESTRATOR[Orchestrator\nLangGraph]
    ORCHESTRATOR -->|consult| BEEBOT[BeebotAgent]
    ORCHESTRATOR -->|order/edit/track| PASSTHROUGH[Роутеры bot_admin/user]

    style SC fill:#ddd6fe
    style BROKER fill:#bfdbfe
    style ANAMNESIS fill:#fef3c7
    style CRMAGENT fill:#bbf7d0
```

### 11.2 Worker→Client переключение интерфейса

```mermaid
flowchart TD
    A[Работник: /start] --> B{suggest_interface()}
    B -->|interface_mode=default| C[Показать очередь сборки]
    B -->|interface_mode=client| D[Показать меню клиента]

    C --> E[Очередь пуста?]
    E -->|Нет| C
    E -->|Да| F[Inline кнопка: Режим клиента]
    F --> G[cb_worker_client_mode\nset_interface_mode → client]
    G --> D

    D --> H[Стандартный /start\nприветствие + меню]
    H --> I[Следующий /start]
    I --> B
```

### 11.3 TunnelMonitor + FAQ Fallback

```mermaid
flowchart LR
    subgraph Monitor["TunnelMonitor (каждые 60 сек)"]
        TCP[TCP connect\nlocalhost:8990]
        TCP -->|OK| HEALTHY[is_healthy = True]
        TCP -->|timeout/OSError| UNHEALTHY[is_healthy = False]
        TCP -->|ConnectionRefused| DEVMODE[Dev mode\nis_healthy = True]
    end

    subgraph Alert["Алертинг"]
        HEALTHY & UNHEALTHY --> DIFF{Состояние\nизменилось?}
        DIFF -->|down→up| UP_ALERT[🟢 Туннель восстановлен]
        DIFF -->|up→down| DOWN_ALERT[🔴 Туннель упал]
        DIFF -->|без изменений| SILENCE[Тихо]
    end

    subgraph Fallback["BeebotAgent fallback"]
        Q[Запрос пользователя] --> CHECK{is_healthy?}
        CHECK -->|True| LLM[LLM ответ]
        CHECK -->|False| FAQ[kb.search → топ-3 чанка\nАвтономный режим]
    end
```

### 11.4 AgentBus (dronedoc2026)

```mermaid
sequenceDiagram
    participant BUS as AgentBus\ndronedoc2026:8081
    participant BOT as AgentBusClient\n(BEEBOT)

    Note over BOT: При старте (если AGENT_BUS_URL задан)
    BOT->>BUS: POST /register {agent_id: beebot, tools: [kb_search, order_status, ask]}
    BUS-->>BOT: 200 OK

    loop Каждые 30 сек
        BOT->>BUS: POST /heartbeat {agent_id: beebot}
    end

    loop Каждые 10 сек
        BOT->>BUS: GET /inbox/{beebot}
        BUS-->>BOT: [{tool, params, reply_to}]
        BOT->>BOT: _handle_message(tool, params)
        BOT->>BUS: POST /respond {reply_to, result}
    end

    Note over BOT: Инструменты: kb_search / order_status / ask
```

### 11.5 BackupManager (Яндекс Диск)

```mermaid
flowchart TD
    A[BackupManager.run()\nцикл раз в час] --> B{YADISK_TOKEN\nзадан?}
    B -->|Нет| Z[Логирование → exit]
    B -->|Да| C[_ensure_dirs\n/BEEBOT/daily/ + /BEEBOT/weekly/]

    C --> D{Сегодня уже\nбэкапили?}
    D -->|Нет| E[_do_daily\ndata/memory.db → Яндекс Диск\n+ _cleanup старше 30 дней]
    D -->|Да| F{Воскресенье?}
    E --> F

    F -->|Да и не делали| G[_do_weekly\nCRM → CSV → Яндекс Диск]
    F -->|Нет| A

    G --> A

    H[backup_now()] -->|вручную из /admin| E
```

### 11.6 PDF-отчёты и веб-панель

| Компонент | Путь | Что делает |
|-----------|------|-----------|
| `PdfReportGenerator` | `src/pdf_report.py` | reportlab: сводка, выручка по месяцам, топ-10 товаров, ABC-клиенты, низкий остаток |
| `GET /api/reports/sales` | `src/web/routers/report.py` | Только admin · period=30d/90d/365d → `application/pdf` |
| Кнопки дашборда | `web/src/views/DashboardView.vue` | 3 кнопки PDF с индикатором загрузки |
| `CrmSnapshot.alert_fn` | `src/crm_snapshot.py` | Алерт при остатке < 5 шт. · дебаунс 24ч |

---

## 10. Эволюция архитектуры: было → станет

> Раздел описывает переход к **Gift Protocol + CRM-агент + SharedContext**.
> Сформулировано 30 марта 2026 на основе анализа роста проекта и принципов dronedoc2026 AgentBus.

---

### 10.1 Было: CRM как общий ресурс

Сейчас `IntegramClient` вызывается напрямую из 10+ мест. Нет единого владельца CRM-домена.

```mermaid
graph TB
    subgraph bot_py["bot.py (1899 строк) — центр всего"]
        BOT[Хэндлеры<br/>FSM-логика<br/>WorkerAgent-вызовы<br/>AdminChat-вызовы]
    end

    subgraph agents["Агенты"]
        ORCH[Orchestrator<br/>роутер intent→агент]
        BEEBOT[BeebotAgent]
        LOGIST[LogistAgent FSM]
        WORKER[WorkerAgent<br/>_checked dict в RAM]
        ADMIN[AdminChatAgent<br/>CrmSnapshot]
        TRACKER[Tracker]
        UDS[UDSPoller]
    end

    MSG[📱 Telegram] --> BOT
    BOT -->|direct| ORCH
    BOT -->|direct| WORKER
    BOT -->|direct| ADMIN

    ORCH -->|consult| BEEBOT
    ORCH -->|order → passthrough| BOT
    ORCH -->|edit/track → passthrough| BOT

    BEEBOT -->|health fact| IC1[IntegramClient]
    LOGIST -->|create_order| IC2[IntegramClient]
    WORKER -->|get_orders<br/>update_status| IC3[IntegramClient]
    ADMIN -->|snapshot| IC4[IntegramClient]
    TRACKER -->|update_tracking| IC5[IntegramClient]
    UDS -->|sync_transaction| IC6[IntegramClient]

    IC1 & IC2 & IC3 & IC4 & IC5 & IC6 -->|все дёргают| CRM[(Integram CRM<br/>ai2o.ru/bibot)]

    style IC1 fill:#fee2e2
    style IC2 fill:#fee2e2
    style IC3 fill:#fee2e2
    style IC4 fill:#fee2e2
    style IC5 fill:#fee2e2
    style IC6 fill:#fee2e2
    style CRM fill:#fca5a5
```

**Проблемы текущей схемы:**
- Integram недоступен → падает всё и везде одновременно
- Бизнес-логика CRM размазана по 6+ файлам
- `bot.py` — единая точка отказа и 1899 строк монолита
- `WorkerAgent._checked` — dict в RAM, теряется при рестарте
- `_node_logist` в оркестраторе возвращает `""` — FSM живёт в bot.py

---

### 10.2 Станет: Gift Protocol + CRM-агент

Единый контур с Gift Broker в центре. CRM — один агент, владеющий доменом.

```mermaid
graph TB
    subgraph FACADES["Фасады (адаптеры внешнего мира)"]
        TG_ADAPT[TelegramAdapter<br/>bot.py → лёгкий фасад]
        WEB_ADAPT[WebAdapter<br/>FastAPI routers]
        UDS_ADAPT[UdsAdapter<br/>uds.py]
    end

    subgraph CORE["Единый контур"]
        SC[("SharedContext\n─────────────\nклиент · заказ\nфакты здоровья\nинтересы · KB-хинты\nTTL 30 мин")]
        BROKER["GiftBroker\n─────────────\nзнает SharedContext\nматчит потребности\nдоставляет дары"]

        subgraph AGENTS["Агенты-лица (ACCEPTED / DEFERRED)"]
            BEEBOT_AG[BeebotAgent\nKB+LLM]
            LOGIST_AG[LogistAgent\nFSM 7 шагов]
            CRM_AG[CrmAgent\nединственный владелец]
            WORKER_AG[WorkerAgent\ninbox + TTL]
            NOTIFIER[NotifierAgent\nTelegram push]
            ANALYST[AnalystAgent\nстатистика]
            TRACKER[TrackerAgent\nСДЭК + Почта]
        end
    end

    CRM_AG -->|единственный| INTEGRAM[(Integram CRM)]

    TG_ADAPT -->|Gift| BROKER
    WEB_ADAPT -->|Gift| BROKER
    UDS_ADAPT -->|Gift| BROKER

    BROKER <-->|read / write| SC
    BROKER -->|Gift + telos| BEEBOT_AG
    BROKER -->|Gift + telos| LOGIST_AG
    BROKER -->|Gift + telos| CRM_AG
    BROKER -->|Gift + telos| WORKER_AG
    BROKER -->|Gift + telos| NOTIFIER
    BROKER -->|Gift + telos| ANALYST
    BROKER -->|Gift + telos| TRACKER

    WORKER_AG -->|DEFERRED если занят| BROKER

    style CRM_AG fill:#bbf7d0
    style INTEGRAM fill:#86efac
    style SC fill:#ddd6fe
    style BROKER fill:#bfdbfe
```

---

### 10.3 Оркестратор: роутер → Gift Broker

```mermaid
flowchart LR
    subgraph WAS["БЫЛО"]
        direction TB
        A1[Сообщение] --> B1[_fast_classify]
        B1 --> C1{intent?}
        C1 -->|consult| D1[BeebotAgent]
        C1 -->|order| E1[passthrough\n→ bot.py FSM]
        C1 -->|edit/track| F1[passthrough\n→ bot.py]
        C1 -->|stats| G1[AnalystAgent]
        D1 -->|прямой вызов| H1[IntegramClient]
        style E1 fill:#fee2e2
        style F1 fill:#fee2e2
        style H1 fill:#fee2e2
    end

    subgraph WILL["СТАНЕТ"]
        direction TB
        A2[Gift от фасада] --> B2[GiftBroker]
        B2 <-->|обогащение| SC2[SharedContext\nфакты Марии\nпрошлые заказы\nздоровье]
        B2 -->|Gift + telos| D2[BeebotAgent\nACCEPTED]
        B2 -->|Gift + telos| E2[LogistAgent\nACCEPTED]
        B2 -->|Gift + telos| F2[CrmAgent\nACCEPTED / DEFERRED]
        D2 -->|new_fact| B2
        E2 -->|order_created| B2
        B2 -->|Gift| G2[WorkerAgent\nNotifier\nAnalyst]
        style SC2 fill:#ddd6fe
        style D2 fill:#bbf7d0
        style E2 fill:#bbf7d0
        style F2 fill:#bbf7d0
    end
```

---

### 10.4 WorkerAgent: RAM → DEFERRED + SharedContext

```mermaid
stateDiagram-v2
    state "БЫЛО (чеклист в RAM)" as WAS {
        [*] --> queue_was : /start
        queue_was --> card_was : выбрать заказ
        card_was --> checklist_was : Взять в работу
        checklist_was --> done_was : все ✅
        done_was --> [*]
        note right of checklist_was
            _checked: dict в RAM
            ⚠️ теряется при рестарте
        end note
    }

    state "СТАНЕТ (inbox + DEFERRED)" as WILL {
        [*] --> idle : старт
        idle --> busy : Gift(order) ACCEPTED
        busy --> idle : заказ собран
        idle --> deferred : Gift(order) DEFERRED\n(работник занят)
        deferred --> busy : broker доставляет\nкогда освободился
        note right of deferred
            Inbox с TTL в SharedContext
            Broker ждёт сигнала готовности
            ✅ переживает рестарт
        end note
        note right of busy
            Чеклист в SharedContext
            не в RAM
        end note
    }
```

---

### 10.5 Сквозной сценарий: Мария заказывает прополис

```mermaid
sequenceDiagram
    actor Мария
    participant TG as TelegramAdapter
    participant Broker as GiftBroker
    participant SC as SharedContext
    participant Beebot as BeebotAgent
    participant Logist as LogistAgent
    participant CRM as CrmAgent
    participant Worker as WorkerAgent
    participant Notifier as NotifierAgent

    Мария->>TG: «У меня болят суставы»
    TG->>Broker: Gift(content, telos=помочь)
    Broker->>SC: читать факты Марии → пусто
    Broker->>Broker: онтология: суставы→прополис
    Broker->>Beebot: Gift(+контекст онтологии) ACCEPTED?
    Beebot-->>Broker: ACCEPTED
    Beebot->>Beebot: FAISS → видео про прополис
    Beebot-->>Broker: Gift(ответ, new_fact=суставы, hint=прополис30мл)
    Broker->>SC: записать факт здоровья Марии
    Broker->>TG: ответ Марии

    Мария->>TG: «Хочу заказать прополис 30мл»
    TG->>Broker: Gift(content)
    Broker->>SC: читать → суставы + hint=прополис30мл
    Broker->>Logist: Gift(+продукт уже известен) ACCEPTED?
    Logist-->>Broker: ACCEPTED
    Note over Logist: пропускает шаг выбора товара
    Logist->>CRM: Gift(create_order) ACCEPTED?
    CRM-->>Logist: ACCEPTED → заказ #923
    Logist-->>Broker: Gift(order_created #923)
    Broker->>SC: обновить статус заказа Марии

    par параллельно
        Broker->>Notifier: Gift(новый заказ #923) ACCEPTED
        Notifier->>Мария: подтверждение
    and
        Broker->>Worker: Gift(заказ #923) DEFERRED
        Note over Worker: занят #921
    end

    Note over Worker: #921 завершён
    Worker-->>Broker: готов
    Broker->>Worker: доставить отложенный Gift #923 ACCEPTED
    Worker->>Worker: чеклист → собрать
    Worker-->>Broker: Gift(собран #923)
    Broker->>CRM: Gift(update_status → Отправлен)
    Broker->>Analyst: Gift(продажа: прополис ← суставы) ACCEPTED
```

---

### 10.6 Сравнение: было vs станет

| Аспект | Было | Станет |
|--------|------|--------|
| **Orchestrator** | Роутер: intent → агент → END | Gift Broker: знает SharedContext, матчит telos |
| **CRM-доступ** | 10+ прямых вызовов IntegramClient | Один CrmAgent — единственный владелец |
| **SharedContext** | Частичный: history + SQLite | Единый: клиент · заказ · здоровье · интересы |
| **WorkerAgent** | `_checked` dict в RAM, нет DEFERRED | Inbox с TTL, DEFERRED, чеклист в SharedContext |
| **LogistAgent** | FSM в bot.py, passthrough из оркестратора | Агент с ACCEPTED, знает контекст из SharedContext |
| **Отказ Integram** | Падает всё везде | CrmAgent: DEFERRED + retry очередь |
| **bot.py** | 1899 строк — монолит | Лёгкий TelegramAdapter (фасад) |
| **Читаемость потока** | Трасси́ровать по 5 файлам | Gift log = полная история события |
| **Агент знает зачем** | Нет (только intent) | Да (telos в каждом подарке) |
| **Свобода агента** | Нет (синхронный вызов) | ACCEPTED / DEFERRED (A5 Gift Ontology) |

---

### 10.7 Эволюционный план: три шага без переписывания

```mermaid
gantt
    title Переход к Gift Protocol (эволюция, не переписывание)
    dateFormat YYYY-MM-DD
    section Шаг 1 — SharedContext
    SharedContext (100 строк)        :s1, 2026-04-01, 3d
    Заменить _dialog_states + histories :s1b, after s1, 2d
    section Шаг 2 — CrmAgent
    Класс CrmAgent-обёртка            :s2, after s1b, 3d
    Перенаправить 10 мест вызова      :s2b, after s2, 3d
    section Шаг 3 — GiftBroker
    Gift TypedDict (5 полей)          :s3, after s2b, 2d
    GiftBroker вместо Orchestrator    :s3b, after s3, 5d
    WorkerAgent inbox + DEFERRED      :s3c, after s3, 3d
```

**Шаг 1 — SharedContext** (малый риск, ~100 строк)
Один `dict` + TTL на user_id. Заменяет `_dialog_states` + `_histories` + разрозненные SQLite-факты. Остальной код не меняется.

**Шаг 2 — CrmAgent** (средний риск, ~150 строк)
Класс-обёртка над существующим `IntegramClient`. Все 10 мест прямого вызова переводятся на один класс. Логика не меняется — меняется маршрут вызова.

**Шаг 3 — GiftBroker** (высокий выхлоп, ~200 строк)
`Gift` как `TypedDict` с пятью полями. `GiftBroker` заменяет `Orchestrator._build_graph()`. LangGraph остаётся внутри — просто получает обогащённый контекст из SharedContext.

---

*Связанные документы: [analysis.md](../analysis.md) · [plan.md](../plan.md)*
