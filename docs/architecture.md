# BEEBOT — Архитектурные диаграммы

> **Версия:** 27 марта 2026

---

## 1. Общая архитектура системы

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

## 2. Оркестратор — Маршрутизация интентов

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

### Быстрая классификация (keyword matching)

| Интент | Ключевые слова |
|--------|---------------|
| greeting | привет, здравствуйте, добрый день, hi, hello |
| order | заказать, купить, хочу заказ, оформить |
| edit | изменить заказ, поменять адрес, скорректировать |
| track | где мой заказ, трек, отслеживание, статус заказа |
| stats | выручка, статистика, продажи, отчёт, ABC, сезонность, прогноз (только ADMIN) |

---

## 3. Консультант — Гибридный поиск

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

    TOP --> PROMPT[Системный промпт<br/>+ история + стиль]
    PROMPT --> GROQ[Groq LLM<br/>llama-3.3-70b]
    GROQ --> RESP[Ответ пользователю]
    RESP --> INSTR{Найдена<br/>PDF-инструкция?}
    INSTR -->|да| BTN[Кнопка «Получить PDF»]
    INSTR -->|нет| END[Отправить ответ]
```

### Базы знаний

| Источник | Файлов | Чанков | Примечание |
|----------|--------|--------|-----------|
| Тексты (data/texts/) | 21 | ~179 | Основной источник, вручную очищенные |
| YouTube субтитры | 26 | ~61 | Расшифровки видео @a.dmitrov |
| PDFs (data/pdfs/) | 19 | — | Перекрыты текстами, не индексируются |
| **Итого** | **47** | **240** | |

---

## 4. Логист — FSM оформления заказа (7 шагов)

```mermaid
stateDiagram-v2
    [*] --> choosing_product: /order
    choosing_product --> entering_name: выбраны товары
    entering_name --> entering_phone: введено имя
    entering_phone --> entering_address: введён телефон ✓
    entering_address --> choosing_delivery: введён адрес
    choosing_delivery --> confirming_order: выбрана доставка + рассчитана стоимость
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

### Тайм-ауты и предзаполнение

| Событие | Действие |
|---------|----------|
| Таймаут 15 мин | FSM автоматически сбрасывается |
| Повторный клиент | Имя и телефон предзаполняются из CRM |
| Повторный адрес | Предлагается последний адрес доставки |
| Расчёт доставки | СДЭК / Почта России / Самовывоз |

### Создание заказа в CRM

```
LogistAgent.create_order()
  → IntegramClient.create_order(client_id, items, delivery, ...)
    → IntegramAPI.create_object(TABLE_ORDERS, number, reqs)
    → для каждого товара: IntegramAPI.create_object(TABLE_ORDER_ITEMS, ...)
  → Notifier.new_order(order) → бот пчеловоду
  → notify_client(client_tg_id) → клиенту (если есть TG)
```

---

## 5. Цикл жизни заказа — Статусы и уведомления

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
    PACK -->|отправка + трек-номер| SENT
    SENT -->|авто-трекинг| DONE
    NEW -->|отмена| CANCEL
    CONF -->|отмена| CANCEL
    PACK -->|отмена| CANCEL

    subgraph Источники смены статуса
        WEB[Веб-панель<br/>PATCH /api/orders/status]
        TGCMD[Telegram /status id статус]
        AUTO[tracker.py<br/>авто-трекинг 2ч]
    end

    WEB -->|SSE + TG клиент + TG пчеловод| CONF
    TGCMD -->|TG клиент только| CONF
    AUTO -->|TG клиент + TG пчеловод| DONE
```

> ✅ **Исправлено (план 3.2):** `notify_beekeeper_status_change` используется во всех трёх точках.

---

## 6. Веб-панель — Архитектура

```mermaid
graph LR
    subgraph Frontend["Frontend (Vue 3 + PrimeVue, PWA)"]
        LOGIN[LoginView<br/>JWT auth]
        DASH[DashboardView<br/>6 карточек + 4 графика]
        ORDERS[OrdersView<br/>список + фильтры]
        CLIENTS[ClientsView]
        PRODUCTS[ProductsView]
        PACKING[PackingView<br/>offline PWA]
        STOCK[StockView<br/>offline PWA]
        MONTHLY[MonthlyOrders]
        USERS[UsersView]
    end

    subgraph Backend["Backend (FastAPI, src/web/ — api.py 167 строк + routers/)"]
        AUTH[/api/auth/token]
        DASH_API[/api/dashboard]
        ORDERS_API[/api/orders/*]
        CLIENTS_API[/api/clients/*]
        PRODUCTS_API[/api/products/*]
        SSE[/api/events SSE]
        HEALTH[/api/health]
    end

    subgraph CRM["Integram CRM"]
        DB[(bibot DB<br/>ai2o.ru)]
    end

    DASH --> DASH_API
    ORDERS --> ORDERS_API
    CLIENTS --> CLIENTS_API
    PRODUCTS --> PRODUCTS_API
    LOGIN --> AUTH
    ORDERS --> SSE

    ORDERS_API --> DB
    CLIENTS_API --> DB
    PRODUCTS_API --> DB
    DASH_API --> DB
```

### Offline-режим (PWA)

```
Первый запуск
  → Vue загружает данные из API → сохраняет в IndexedDB
  → Service Worker кэширует статику

Offline
  → PackingView/StockView читают из IndexedDB
  → Изменения пишутся в Sync Queue (IndexedDB)

Reconnect
  → Sync Queue отправляется на сервер
  → IndexedDB обновляется свежими данными
```

---

## 7. Ассистент пчеловода (AdminChatAgent)

```mermaid
flowchart TD
    ADMIN[Пчеловод пишет /admin] --> TOGGLE{режим<br/>включён?}
    TOGGLE -->|нет| ON[Включить режим<br/>показать меню]
    TOGGLE -->|да| OFF[Выключить режим<br/>очистить историю]

    ADMIN2[Пчеловод пишет сообщение] --> CHECK{в режиме<br/>admin?}
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

    CRM_SNAP --> S1
    S1 --> S2
    S2 --> S3
    S3 --> S4
    S4 --> S5
    S5 --> S6

    S6 --> PROMPT[Системный промпт + CRM + история]
    PROMPT --> GROQ[Groq LLM]
    GROQ --> REPLY[Ответ пчеловоду]
    REPLY --> HIST[Сохранить в историю<br/>макс. 20 сообщений]
```

---

## 8. UDS-интеграция

```mermaid
sequenceDiagram
    participant UDS as UDS API
    participant POLLER as UDS Poller (каждые 5 мин)
    participant CRM as Integram CRM
    participant BOT as Telegram-бот

    loop каждые 5 минут
        POLLER->>UDS: GET /transactions?from=cursor
        UDS-->>POLLER: список транзакций

        loop каждая транзакция
            POLLER->>POLLER: Проверить дедупликацию по UDS ID
            alt уже обработана
                POLLER-->>POLLER: skip
            else новая транзакция
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

## 10. LLM-цепочка (через hive)

```
beebot (VPS) → localhost:8990
  → SSH reverse tunnel: VPS:8990 ← hive:8990
  → groq-proxy.service (hive) → api.groq.com

beebot-web (VPS) → SOCKS5 localhost:9150
  → SSH reverse tunnel: VPS:9150 ← hive:9150
  → tg-socks.service (hive) → api.telegram.org
```

### systemd-сервисы на hive

| Сервис | Порт | Назначение |
|--------|------|-----------|
| `groq-proxy.service` | 8990 | Reverse proxy hive → api.groq.com |
| `groq-tunnel.service` | — | SSH-туннель VPS↔hive (8990 + 9150) |
| `tg-socks.service` | 9150 | SOCKS5-прокси для Telegram API |

---

## 11. Сравнительная таблица агентов

| Агент | Вход | Хранение состояния | LLM | CRM | KB |
|-------|------|--------------------|-----|-----|----|
| Оркестратор | сообщение | in-memory + TTL 30мин | ✅ (классификация) | ❌ | ❌ |
| Консультант | запрос + история | in-memory | ✅ (ответ) | ❌ | ✅ FAISS |
| Логист | шаги FSM | aiogram FSMContext | ✅ (подтверждение) | ✅ создание | ❌ |
| Аналитик | запрос аналитики | нет | ✅ (парсинг запроса) | ✅ чтение | ❌ |
| Инспектор | шаги диалога | in-memory | ✅ (вопросы + рекомендация) | ❌ | ✅ FAISS |
| Ассистент | свободный диалог | in-memory (20 сообщ.) | ✅ (диалог) | ✅ снимок | ❌ |

---

## 12. Сравнительная таблица доставки

| Параметр | СДЭК | Почта России | Самовывоз |
|----------|------|-------------|-----------|
| API | v2 REST (OAuth2) | tariff.pochta.ru | — |
| Авторизация | Client ID + Secret | Токен | — |
| Стоимость | по тарифу | по тарифу | 0 ₽ |
| Трекинг | ✅ | ✅ | ❌ |
| Fallback | фикс. 350₽+50₽/кг | фикс. 250₽+30₽/кг | — |
| Кэш локаций | in-memory (города) | почтовые индексы (50+ городов) | — |

---

## 13. Сравнительная таблица уведомлений (текущее состояние)

| Событие | SSE в браузер | TG клиенту | TG пчеловоду | Код |
|---------|:---:|:---:|:---:|-----|
| Смена статуса — **веб** | ✅ | ✅ | ✅ | `web/routers/orders.py` + `web/notifications.py` |
| Смена статуса — **TG /status** | ❌ | ✅ | ✅ | `admin.py` + `web/notifications.py` |
| Смена статуса — **авто-трекинг** | ❌ | ✅ | ✅ | `tracker.py` + `web/notifications.py` |
| Новый заказ — **бот** | N/A | ✅ | ✅ | `notifications.py` (Notifier) |
| Новый заказ — **UDS** | N/A | N/A | ✅ | `integrations/uds.py` |

> ✅ **Исправлено (план 3.2):** `notify_beekeeper_status_change` из `src/web/notifications.py` — единая точка уведомлений для всех трёх источников смены статуса.

---

---

## 14. Обновление reference-полей Integram (_m_del + _m_set)

> **Открыто:** 27.03.2026 — реверс-инженерия Integram UI

`_m_save` возвращает `saved1=1`, но **молча игнорирует** изменения reference-полей (статус заказа, источник, способ доставки). Это нативная особенность Integram.

```mermaid
sequenceDiagram
    participant APP as integram_client.py
    participant API as integram_api.py
    participant CRM as Integram (ai2o.ru/bibot)

    APP->>API: set_reference_field(obj_id, req_id, new_value_id)

    API->>CRM: GET /bibot/edit_obj/{obj_id}?JSON
    CRM-->>API: reqs[req_id].multiselect.id = [link1, link2, ...]

    loop каждый старый link
        API->>CRM: POST /bibot/_m_del/{link_id}?JSON
        CRM-->>API: ok
    end

    API->>CRM: POST /bibot/_m_set/{obj_id}?JSON<br/>body: t{req_id}={new_value_id}
    CRM-->>API: ok
```

### Когда применять

| Поле заказа | Метод | Константа |
|-------------|-------|-----------|
| Статус | `set_reference_field` | `REQ_ORDER_STATUS` → `STATUS_IDS[status]` |
| Источник | `set_reference_field` | `REQ_ORDER_SOURCE` → `SOURCE_IDS[source]` |
| Способ доставки | `set_reference_field` | `REQ_ORDER_DELIVERY` → строковый ID |
| Трек-номер, ФИО, адрес | `set_requisites` | обычные текстовые поля |

---

*Связанные документы: [analysis.md](../analysis.md) · [plan.md](../plan.md)*
