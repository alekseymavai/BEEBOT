# BEEBOT

**Цифровой помощник пчеловода** — Telegram-бот + веб-панель управления заказами для «Усадьба Дмитровых».

Бот отвечает на вопросы подписчиков в стиле автора блога, принимает заказы, управляет складом, доставкой и является личным ассистентом пчеловода с доступом к CRM.

**Telegram:** [@AleksandrDmitrov_BEEBOT](https://t.me/AleksandrDmitrov_BEEBOT)
**Веб-панель:** http://185.233.200.13:8088

---

## Возможности

### Telegram-бот
- Консультации по продуктам пчеловодства — гибридный FAISS (70% семантика + 30% стилометрия + keyword-буст)
- Оформление заказов через диалог (FSM, 7 шагов) с записью в CRM и расчётом доставки
- Каталог товаров с PDF-инструкциями по каждому продукту
- Автоподстановка данных клиента по Telegram ID (имя, телефон, адрес)
- **«Голос Улья»** — 5 стилей ответов (основатель, наставник, краевед, учёный, молодой)
- **«Осмотр улья»** — диагностический диалог: 3 вопроса → персональная рекомендация
- **«Ассистент пчеловода»** (команда `/admin`) — личный LLM-помощник с полным CRM-снимком
- История диалога (последние 5 сообщений, TTL 30 мин)
- Работа в группах (по @mention или reply)

### Веб-панель (PWA)
- Дашборд с графиками (выручка, заказы, статусы, доставка)
- Управление заказами: список, детали, смена статуса, трекинг, создание
- Каталог клиентов с историей заказов
- CRUD товаров и управление складом
- Терминал сборки заказов (offline-first, PWA)
- Терминал склада: +/− остатков, алерты при низком запасе (offline-first, PWA)
- Журнал заказов по месяцам
- Экспорт в CSV (заказы, клиенты, товары)
- SSE-уведомления в реальном времени
- JWT-авторизация, rate limiting (60 req/min)

### Доставка
- СДЭК API v2: расчёт стоимости (OAuth2 + tariff), трекинг
- Почта России: расчёт стоимости (tariff.pochta.ru), трекинг
- Авто-трекинг отправлений — фоновая проверка каждые 2 часа
- Уведомление клиенту в Telegram при доставке

### CRM-интеграция (Integram bibot)
- 76 товаров · 285+ клиентов · 382+ заказов
- UDS-синхронизация (поллинг каждые 5 мин, дедупликация, уведомления)
- 6 статусов заказа: Новый → Подтверждён → В сборке → Отправлен → Доставлен / Отменён
- Keyword-буст: автоматическое расширение словаря KB из каталога CRM

---

## Архитектура

```
Telegram ──→ Оркестратор (LangGraph) ──→ Агенты
                                           ├── Консультант (FAISS → Groq LLM)
                                           ├── Логист (FSM заказов → CRM + доставка)
                                           ├── Аналитик (CRM → отчёты)
                                           ├── Инспектор (Осмотр улья)
                                           └── Ассистент (/admin → LLM + CRM-снимок)

Веб-панель (Vue 3 + PrimeVue, PWA) ──→ FastAPI ──→ Integram CRM

Groq API ←── groq-proxy (hive:8990) ←── SSH-туннель ←── VPS
Telegram API ←── SOCKS5 (hive:9150) ←── SSH-туннель ←── VPS
```

Подробные диаграммы: [docs/architecture.md](docs/architecture.md)

---

## Быстрый старт

### 1. Клонировать и настроить

```bash
git clone https://github.com/alekseymavai/BEEBOT.git
cd BEEBOT
cp .env.example .env
# Отредактировать .env: TELEGRAM_BOT_TOKEN, GROQ_API_KEY, INTEGRAM_*, WEB_PASSWORD
```

### 2. Docker (рекомендуется)

```bash
docker compose up -d
```

Бот запустится в контейнере `beebot`, веб-панель — на порту 8088.

### 3. Собрать базу знаний

```bash
docker exec beebot python -m src.build_kb
```

### 4. Локальная разработка

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.build_kb
python -m src.bot
```

### 5. Тесты

```bash
pip install pytest pytest-asyncio
python -m pytest tests/ -x -q
```

---

## Структура проекта

```
BEEBOT/
├── src/
│   ├── bot.py                  # Telegram-бот: хэндлеры, FSM, startup (1 445 строк)
│   ├── orchestrator.py         # LangGraph — 6 интентов + история диалога
│   ├── config.py               # Конфигурация из .env
│   ├── models.py               # Pydantic-модели (Order, Client, Product, OrderItem)
│   ├── crm_constants.py        # CRM ID-маппинг (статусы, источники, категории)
│   ├── crm_schema.py           # Схема таблиц и реквизитов CRM
│   ├── phone_utils.py          # Валидация и нормализация телефона
│   ├── agents/
│   │   ├── beebot.py           # Консультант: FAISS-поиск → LLM-ответ
│   │   ├── logist.py           # Логист: FSM 7 шагов → заказ в CRM
│   │   ├── analyst.py          # Аналитик: статистика продаж из CRM
│   │   ├── inspector.py        # Инспектор: диагностический диалог
│   │   └── admin_chat.py       # Ассистент пчеловода: /admin + CRM-снимок
│   ├── knowledge_base.py       # FAISS + стилометрия (70/30) + keyword-буст
│   ├── llm_client.py           # Groq API (retry + backoff + Голос Улья)
│   ├── integram_api.py         # CRM REST API низкоуровневый (auto re-auth)
│   ├── integram_client.py      # CRM обёртка высокоуровневая (Pydantic)
│   ├── admin.py                # Админ-команды Telegram (/orders, /status, /track...)
│   ├── notifications.py        # Уведомления пчеловоду (Notifier)
│   ├── delivery/
│   │   ├── calculator.py       # Калькулятор (СДЭК + Почта + самовывоз)
│   │   ├── cdek.py             # СДЭК API v2 (OAuth2 + tariff + tracking)
│   │   ├── pochta.py           # Почта России (tariff + tracking)
│   │   └── tracker.py          # Авто-трекинг каждые 2 часа
│   ├── integrations/
│   │   └── uds.py              # UDS: поллер + дедупликация → CRM
│   └── web/
│       ├── api.py              # FastAPI: main router + startup + зависимости (167 строк)
│       ├── routers/            # 8 маршрутных модулей (auth, orders, clients, products, dashboard, export, users, sse)
│       ├── notifications.py    # Уведомления при смене статуса через веб
│       ├── users.py            # Управление пользователями веб-панели
│       └── server.py           # Статика + PWA root
├── web/                        # Frontend (Vue 3 + PrimeVue 4, PWA)
│   └── src/
│       ├── views/              # 11 страниц
│       ├── components/         # UI-компоненты
│       ├── stores/             # Pinia (auth + offline/IndexedDB)
│       ├── api.js              # axios + JWT interceptor
│       └── utils.js            # formatDate, formatMoney
├── tests/                      # 284 теста (pytest + pytest-asyncio)
├── data/
│   ├── pdfs/                   # 19 PDF-инструкций (прополис, перга, ПЖВМ и др.)
│   ├── texts/                  # 20 текстовых источников (очищенные выдержки)
│   ├── subtitles/              # 26 расшифровок YouTube (@a.dmitrov)
│   └── processed/              # FAISS-индекс + chunks.json (240 чанков)
├── docs/
│   └── architecture.md         # Mermaid-диаграммы, блок-схемы, сравнительные таблицы
├── systemd/                    # systemd-сервисы для hive (groq-proxy, groq-tunnel, tg-socks)
├── .github/workflows/ci.yml   # CI/CD: lint (ruff) + тесты + деплой при мерже
├── Dockerfile                  # Бот (Python 3.12-slim + FAISS + fastembed)
├── Dockerfile.web              # Веб-панель (Node build → Python serve)
├── docker-compose.yml          # 2 сервиса: beebot + beebot-web (8088)
├── groq_proxy.py               # Reverse proxy hive:8990 → api.groq.com
├── tg_socks_proxy.py           # SOCKS5-сервер для Telegram API (hive:9150)
├── analysis.md                 # Анализ: сильные/слабые стороны, конфликты
└── plan.md                     # План развития (фазы 3–6)
```

---

## Технологии

| Компонент | Технология |
|-----------|-----------|
| Бот | Python 3.12, aiogram 3.25, asyncio |
| Оркестратор | LangGraph (StateGraph) |
| LLM | Groq API (llama-3.3-70b-versatile) |
| Эмбеддинги | fastembed (paraphrase-multilingual-MiniLM-L12-v2) |
| Векторный поиск | FAISS (IndexFlatIP, cosine similarity) |
| CRM | Integram (ai2o.ru/bibot) — REST API через httpx |
| Доставка | СДЭК API v2 (OAuth2), Почта России |
| Веб-API | FastAPI + uvicorn + SSE + slowapi |
| Frontend | Vue 3, PrimeVue 4, Vite, PWA (vite-plugin-pwa) |
| Offline | Service Worker + IndexedDB |
| Тесты | pytest + pytest-asyncio (284 теста) |
| CI/CD | GitHub Actions (lint + test + deploy) |
| Деплой | Docker, docker-compose |

---

## Документация

- [Анализ проекта](analysis.md) — сильные/слабые стороны, логические конфликты, состояние инфраструктуры
- [План развития](plan.md) — фазы 3–6: единые уведомления, долгосрочная память, AgentBus, масштабирование
- [Архитектурные диаграммы](docs/architecture.md) — Mermaid-схемы всех подсистем, сравнительные таблицы

---

## Инфраструктура

| Ресурс | Адрес |
|--------|-------|
| VPS | 185.233.200.13 (Docker, 2 GB RAM + 2 GB swap) |
| Веб-панель | http://185.233.200.13:8088 |
| CRM | ai2o.ru/bibot |
| Groq-прокси | hive:8990 (SSH-туннель, systemd) |
| GitHub upstream | [alekseymavai/BEEBOT](https://github.com/alekseymavai/BEEBOT) |

---

---

# BEEBOT (English)

**Digital beekeeper's assistant** — Telegram bot + web dashboard for order management at "Usadba Dmitrovykh".

The bot answers subscriber questions in the author's personal style, processes orders, manages inventory and shipping, and serves as a personal AI assistant for the beekeeper with direct CRM access.

**Telegram:** [@AleksandrDmitrov_BEEBOT](https://t.me/AleksandrDmitrov_BEEBOT)

## Features

- **Telegram bot:** Product consultations via hybrid FAISS search (70% semantic + 30% stylometric + keyword boost), 7-step order FSM with CRM integration, PDF product guides, "Voice of the Hive" (5 response styles), "Hive Inspection" diagnostic dialogue, **personal LLM assistant** (`/admin`) with live CRM snapshot
- **Web dashboard (PWA):** Revenue charts, order/client/product management with pagination & search, packing & stock terminals (offline-first), CSV export, real-time SSE notifications, JWT auth
- **Shipping:** CDEK API v2 (OAuth2) and Russian Post — real cost calculation + background auto-tracking every 2 hours
- **CRM:** Integram (76 products, 285+ clients, 382+ orders), UDS loyalty system sync (5-min polling)
- **Multi-agent:** LangGraph orchestrator routing to Consultant, Logist, Analyst, Inspector, and Admin Chat agents
- **Testing:** 284 tests, GitHub Actions CI/CD with auto-deploy

## Quick Start

```bash
git clone https://github.com/alekseymavai/BEEBOT.git
cd BEEBOT
cp .env.example .env
# Edit .env: TELEGRAM_BOT_TOKEN, GROQ_API_KEY, INTEGRAM_*, WEB_PASSWORD
docker compose up -d
docker exec beebot python -m src.build_kb  # build knowledge base (240 chunks)
```

## Tech Stack

Python 3.12 · aiogram 3 · LangGraph · Groq API (llama-3.3-70b) · FAISS · fastembed · FastAPI · Vue 3 · PrimeVue 4 · Docker · GitHub Actions

## Documentation

- [Project Analysis](analysis.md) — strengths, weaknesses, logic conflicts, infrastructure state
- [Development Plan](plan.md) — roadmap: unified notifications, long-term memory, AgentBus, scaling
- [Architecture Diagrams](docs/architecture.md) — Mermaid diagrams, data flows, comparison tables
