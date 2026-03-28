# BEEBOT — Архитектура системы

> **Версия:** 28 марта 2026
> Подробные диаграммы отдельными файлами: [`diagrams/`](diagrams/)

---

## Содержание

1. [Общая архитектура](#1-общая-архитектура)
2. [Оркестратор — маршрутизация интентов](#2-оркестратор--маршрутизация-интентов)
3. [Консультант — гибридный поиск](#3-консультант--гибридный-поиск)
4. [Логист — FSM оформления заказа](#4-логист--fsm-оформления-заказа)
5. [Цикл жизни заказа](#5-цикл-жизни-заказа)
6. [Веб-панель](#6-веб-панель)
7. [Ассистент пчеловода](#7-ассистент-пчеловода)
8. [UDS-интеграция](#8-uds-интеграция)
9. [Авто-трекинг доставки](#9-авто-трекинг-доставки)
10. [LLM-цепочка через hive](#10-llm-цепочка-через-hive)
11. [Сравнительные таблицы](#11-сравнительные-таблицы)

---

## 1. Общая архитектура

BEEBOT — двухуровневая система: **Telegram-бот** (приём сообщений + мультиагентная обработка) и **Веб-панель** (управление заказами, клиентами, складом).

```mermaid
graph TB
    subgraph Пользователи
        TG_USER[📱 Подписчики<br/>Telegram]
        ADMIN[🐝 Пчеловод<br/>Telegram + Веб]
    end

    subgraph VPS["VPS 185.233.200.13 (Docker)"]
        BOT[🤖 Telegram-бот<br/>aiogram 3 · polling]
        WEB_API[🌐 FastAPI<br/>REST API + JWT + SSE]
        WEB_FRONT[💻 Vue 3 PWA<br/>порт 8088]
        TRACKER[⏱ Авто-трекинг<br/>каждые 2 часа]
        UDS_POLL[🔄 UDS Poller<br/>каждые 5 минут]
    end

    subgraph Агенты
        ORCH[🧠 Оркестратор<br/>LangGraph]
        CONSUL[🐝 Консультант<br/>FAISS → LLM]
        LOGIST[📦 Логист<br/>FSM 7 шагов]
        ANALYST[📊 Аналитик<br/>CRM → отчёты]
        INSPECT[🔍 Инспектор<br/>Осмотр улья]
        ADMINCHAT[🤖 Ассистент<br/>LLM + CRM-снимок]
    end

    subgraph KB["База знаний"]
        FAISS[(FAISS<br/>240 чанков)]
        DOCS[📄 PDF / TXT<br/>YouTube субтитры]
    end

    subgraph External["Внешние сервисы"]
        GROQ[⚡ Groq API<br/>llama-3.3-70b]
        CRM[(Integram CRM<br/>ai2o.ru/bibot)]
        CDEK[🚚 СДЭК API v2<br/>OAuth2]
        POCHTA[📮 Почта России<br/>tariff.pochta.ru]
        UDS_API[💳 UDS API<br/>система лояльности]
        TG_API[💬 Telegram API<br/>api.telegram.org]
    end

    subgraph HIVE["Hive (локальная машина)"]
        PROXY[groq-proxy<br/>порт 8990]
        TG_SOCKS[SOCKS5-прокси<br/>порт 9150]
    end

    TG_USER -->|сообщения| TG_API
    ADMIN -->|команды| TG_API
    ADMIN -->|браузер| WEB_FRONT

    TG_API -->|polling| BOT
    BOT --> ORCH
    ORCH --> CONSUL
    ORCH --> LOGIST
    ORCH --> ANALYST
    ORCH --> INSPECT
    BOT -->|/admin| ADMINCHAT

    CONSUL --> FAISS
    DOCS --> FAISS
    CONSUL --> GROQ
    ADMINCHAT --> GROQ

    LOGIST --> CRM
    ANALYST --> CRM
    ADMINCHAT --> CRM
    WEB_API --> CRM
    TRACKER --> CRM
    UDS_POLL --> CRM

    WEB_FRONT --> WEB_API
    TRACKER --> CDEK
    TRACKER --> POCHTA
    LOGIST --> CDEK
    LOGIST --> POCHTA
    UDS_POLL --> UDS_API

    GROQ -.->|SSH-туннель| PROXY
    PROXY -.->|reverse proxy| GROQ
    TG_API -.->|SOCKS5| TG_SOCKS
```

---

## 2. Оркестратор — маршрутизация интентов

**Файл:** `src/orchestrator.py`
**Фреймворк:** LangGraph StateGraph

Оркестратор — точка входа для каждого сообщения от пользователя. Он определяет **интент** и направляет запрос нужному агенту.

**Алгоритм:**
1. Сначала быстрая классификация по ключевым словам (без LLM)
2. Если не определено — LLM-классификация (~100 токенов)
3. Маршрутизация к агенту
4. Сохранение истории диалога (5 пар, TTL 30 мин)

```mermaid
flowchart TD
    MSG[Входящее сообщение] --> FAST{Быстрая<br/>классификация}
    FAST -->|ключевые слова| INTENT[Определён интент]
    FAST -->|не определено| LLM_CLS[LLM-классификация<br/>Groq ~100 токенов]
    LLM_CLS --> INTENT

    INTENT -->|consult| CONSUL[🐝 Консультант<br/>FAISS → LLM]
    INTENT -->|order| LOGIST[📦 Логист FSM]
    INTENT -->|stats| ANALYST[📊 Аналитик]
    INTENT -->|greeting| GREET[⚡ Быстрый ответ<br/>без LLM]
    INTENT -->|edit| EDIT[✏️ Редактирование<br/>заказа]
    INTENT -->|track| TRACK[📍 Трекинг<br/>заказа]

    CONSUL --> SAVE[Сохранить диалог<br/>TTL 30 мин]
    LOGIST --> SAVE
    ANALYST --> SAVE
```

### Быстрая классификация

| Интент | Ключевые слова |
|--------|---------------|
| greeting | привет, здравствуйте, добрый день, hi, hello |
| order | заказать, купить, хочу заказ, оформить |
| edit | изменить заказ, поменять адрес, скорректировать |
| track | где мой заказ, трек, отслеживание, статус заказа |
| stats | выручка, статистика, продажи, отчёт, ABC, сезонность, прогноз |

---

## 3. Консультант — гибридный поиск

**Файл:** `src/agents/beebot.py` + `src/knowledge_base.py`

Отвечает на вопросы подписчиков в стиле Александра Дмитрова, используя базу знаний.

**Алгоритм:**
1. Векторизация запроса через fastembed (384-dim)
2. FAISS-поиск top-10 (cosine similarity)
3. Стилометрическая оценка (5 признаков пасечного стиля)
4. Гибридный скоринг: 70% семантика + 30% стилометрия
5. Keyword-буст +40% для слов из CRM-товаров
6. Top-5 чанков → промпт → Groq LLM

```mermaid
flowchart LR
    Q[Вопрос пользователя] --> EMB[fastembed<br/>384-dim vector]
    EMB --> SEM[FAISS<br/>семантический поиск<br/>top-10]
    Q --> STYLE[Стилометрия<br/>5 признаков]
    STYLE --> STYLESCORE[Стилометрическая<br/>оценка]

    SEM --> HYBRID[Гибридная оценка<br/>70% semantic<br/>30% stylometric]
    STYLESCORE --> HYBRID

    HYBRID --> KWORD{Keyword<br/>буст?}
    KWORD -->|да +40%| BOOST[Буст результатов<br/>по ключевым словам CRM]
    KWORD -->|нет| TOP[Top-5 чанков]
    BOOST --> TOP

    TOP --> PROMPT[Системный промпт<br/>+ история + стиль<br/>+ советы пчеловода]
    PROMPT --> GROQ[Groq LLM<br/>llama-3.3-70b]
    GROQ --> RESP[Ответ пользователю]
    RESP --> INSTR{Найдена<br/>PDF-инструкция?}
    INSTR -->|да| BTN[Кнопка «Получить PDF»]
    INSTR -->|нет| END[Отправить ответ]
```

### База знаний

| Источник | Файлов | Чанков |
|----------|--------|--------|
| Тексты (data/texts/) | 21 | ~179 |
| YouTube субтитры | 26 | ~61 |
| PDFs (data/pdfs/) | 19 | — (перекрыты текстами) |
| **Итого** | **47** | **240** |

---

## 4. Логист — FSM оформления заказа

**Файл:** `src/agents/logist.py`

7-шаговый диалог: выбор товаров → ФИО → телефон → адрес → доставка → подтверждение → создание в CRM.

```mermaid
stateDiagram-v2
    [*] --> choosing_product: /order
    choosing_product --> entering_name: выбраны товары
    entering_name --> entering_phone: введено имя
    entering_phone --> entering_address: введён телефон ✓
    entering_address --> choosing_delivery: введён адрес
    choosing_delivery --> confirming_order: выбрана доставка
    confirming_order --> creating_order: подтверждено ✓
    confirming_order --> choosing_product: отказ ✗
    creating_order --> [*]: заказ создан в CRM

    choosing_product --> [*]: /cancel
    entering_name --> [*]: /cancel
    entering_phone --> [*]: /cancel
    entering_address --> [*]: /cancel
    choosing_delivery --> [*]: /cancel
    confirming_order --> [*]: /cancel
```

| Событие | Действие |
|---------|----------|
| Таймаут 15 мин | FSM автоматически сбрасывается |
| Повторный клиент | Имя и телефон предзаполняются из CRM |
| Повторный адрес | Предлагается последний адрес доставки |

---

## 5. Цикл жизни заказа

```mermaid
flowchart TD
    NEW[🆕 Новый]
    CONF[✅ Подтверждён]
    PACK[📦 В сборке]
    SENT[🚚 Отправлен]
    DONE[✅ Доставлен]
    CANCEL[❌ Отменён]

    NEW -->|подтверждение| CONF
    CONF -->|сборка| PACK
    PACK -->|отправка + трек| SENT
    SENT -->|авто-трекинг| DONE
    NEW & CONF & PACK -->|отмена| CANCEL

    subgraph Источники смены статуса
        WEB[Веб-панель<br/>PATCH /api/orders/status]
        TGCMD[Telegram /status id]
        AUTO[tracker.py<br/>авто-трекинг 2ч]
    end
```

---

## 6. Веб-панель

**Файлы:** `src/web/` + `web/`

```mermaid
graph LR
    subgraph Frontend["Frontend (Vue 3 + PrimeVue, PWA)"]
        LOGIN[LoginView]
        DASH[DashboardView<br/>6 карточек + 4 графика]
        ORDERS[OrdersView]
        CLIENTS[ClientsView]
        PRODUCTS[ProductsView]
        PACKING[PackingView<br/>offline PWA]
        STOCK[StockView<br/>offline PWA]
    end

    subgraph Backend["Backend (FastAPI, src/web/)"]
        AUTH[/api/auth/token]
        DASH_API[/api/dashboard]
        ORDERS_API[/api/orders/*]
        CLIENTS_API[/api/clients/*]
        PRODUCTS_API[/api/products/*]
        SSE[/api/events SSE]
    end

    subgraph CRM["Integram CRM"]
        DB[(bibot DB)]
    end

    DASH --> DASH_API
    ORDERS --> ORDERS_API & SSE
    CLIENTS --> CLIENTS_API
    PRODUCTS --> PRODUCTS_API
    LOGIN --> AUTH
    ORDERS_API & CLIENTS_API & PRODUCTS_API & DASH_API --> DB
```

---

## 7. Ассистент пчеловода

**Файл:** `src/agents/admin_chat.py`
Активируется командой `/admin`. Собирает полный CRM-снимок и передаёт в LLM для свободного диалога.

```mermaid
flowchart TD
    ADMIN[Пчеловод /admin] --> TOGGLE{режим включён?}
    TOGGLE -->|нет| ON[Включить режим]
    TOGGLE -->|да| OFF[Выключить режим]

    MSG[Сообщение пчеловода] --> CHECK{в режиме admin?}
    CHECK -->|нет| ORCH2[Оркестратор]
    CHECK -->|да| CRM_SNAP[Собрать CRM-снимок]

    subgraph Снимок CRM
        S1[Заказы: всего + активных + выручка]
        S2[Помесячная статистика — 6 мес.]
        S3[Последние 10 заказов с товарами]
        S4[Топ-10 товаров по количеству]
        S5[Склад: мало на складе]
        S6[Клиенты: кол-во в базе]
    end

    CRM_SNAP --> S1 --> S2 --> S3 --> S4 --> S5 --> S6
    S6 --> PROMPT[Системный промпт + CRM + история]
    PROMPT --> GROQ[Groq LLM]
    GROQ --> REPLY[Ответ пчеловоду]
    REPLY --> HIST[Сохранить в историю<br/>макс. 20 сообщений]
```

---

## 8. UDS-интеграция

**Файл:** `src/integrations/uds.py`

```mermaid
sequenceDiagram
    participant POLLER as UDS Poller (каждые 5 мин)
    participant UDS as UDS API
    participant CRM as Integram CRM
    participant BOT as Telegram-бот

    loop каждые 5 минут
        POLLER->>UDS: GET /transactions?from=cursor
        UDS-->>POLLER: список транзакций
        loop каждая транзакция
            POLLER->>POLLER: Проверить дедупликацию по UDS ID
            alt новая транзакция
                POLLER->>CRM: Найти/создать клиента (по телефону)
                POLLER->>CRM: Создать заказ (source=UDS)
                POLLER->>BOT: Уведомить пчеловода
            end
        end
        POLLER->>POLLER: Обновить cursor
    end
```

---

## 9. Авто-трекинг доставки

**Файл:** `src/delivery/tracker.py`

```mermaid
sequenceDiagram
    participant TRACKER as tracker.py (каждые 2ч)
    participant CRM as Integram CRM
    participant CDEK as СДЭК API
    participant POCHTA as Почта России
    participant TG as Telegram

    TRACKER->>CRM: Получить заказы со статусом Отправлен
    loop каждый заказ с трек-номером
        alt СДЭК
            TRACKER->>CDEK: GET tracking_number
            CDEK-->>TRACKER: статус
        else Почта России
            TRACKER->>POCHTA: GET tracking_number
            POCHTA-->>TRACKER: статус
        end
        alt доставлено
            TRACKER->>CRM: Обновить статус → Доставлен
            TRACKER->>TG: Уведомить клиента
        end
    end
```

---

## 10. LLM-цепочка через hive

Groq API блокирует IP VPS, поэтому запросы проходят через SSH-туннель на локальную машину (hive):

```
beebot (VPS) → localhost:8990
  → SSH reverse tunnel: VPS:8990 ← hive:8990
  → groq-proxy.service (hive) → api.groq.com

Telegram API (VPS) → SOCKS5 localhost:9150
  → SSH reverse tunnel: VPS:9150 ← hive:9150
  → tg-socks.service (hive) → api.telegram.org
```

| Сервис | Порт | Назначение |
|--------|------|-----------|
| `groq-proxy.service` | 8990 | Reverse proxy hive → api.groq.com |
| `groq-tunnel.service` | — | SSH-туннель VPS↔hive (8990 + 9150) |
| `tg-socks.service` | 9150 | SOCKS5-прокси для Telegram API |
| `devbot.service` | 8091 | DEVBOT автономный разработчик |

---

## 11. Сравнительные таблицы

### Агенты

| Агент | Вход | Состояние | LLM | CRM | KB |
|-------|------|-----------|-----|-----|----|
| Оркестратор | сообщение | in-memory + TTL 30мин | ✅ классификация | ❌ | ❌ |
| Консультант | запрос + история | in-memory | ✅ ответ | ❌ | ✅ FAISS |
| Логист | шаги FSM | aiogram FSMContext | ✅ подтверждение | ✅ создание | ❌ |
| Аналитик | запрос аналитики | нет | ✅ парсинг запроса | ✅ чтение | ❌ |
| Инспектор | шаги диалога | in-memory | ✅ вопросы + рекомендация | ❌ | ✅ FAISS |
| Ассистент | свободный диалог | in-memory (20 сообщ.) | ✅ диалог | ✅ снимок | ❌ |

### Доставка

| Параметр | СДЭК | Почта России | Самовывоз |
|----------|------|-------------|-----------|
| API | v2 REST (OAuth2) | tariff.pochta.ru | — |
| Трекинг | ✅ | ✅ | ❌ |
| Fallback | 350₽+50₽/кг | 250₽+30₽/кг | — |

### Уведомления

| Событие | SSE | TG клиенту | TG пчеловоду |
|---------|:---:|:---:|:---:|
| Смена статуса — веб | ✅ | ✅ | ✅ |
| Смена статуса — TG /status | ❌ | ✅ | ✅ |
| Смена статуса — авто-трекинг | ❌ | ✅ | ✅ |
| Новый заказ — бот | N/A | ✅ | ✅ |
| Новый заказ — UDS | N/A | N/A | ✅ |

---

*Связанные документы: [analysis.md](../../analysis.md) · [plan.md](../../plan.md)*
