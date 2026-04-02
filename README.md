# BEEBOT

**Цифровой помощник пчеловода** — Telegram-бот + веб-панель управления заказами для «Усадьба Дмитровых».

Telegram: [@AleksandrDmitrov_BEEBOT](https://t.me/AleksandrDmitrov_BEEBOT)
Веб-панель: http://185.233.200.13:8088

---

## Возможности

### Telegram-бот
- `/start`, `/products`, `/help` — главное меню (ReplyKeyboard)
- `/order` — оформление заказа (FSM, 7 шагов)
- `/inspect` — «Осмотр улья»: диагностический диалог (3 вопроса → рекомендация)
- `/voice` — выбор «Голоса Улья» (5 стилей)
- `/admin` — личный ассистент пчеловода + CRM-контекст
- `/stats` — аналитика продаж (ABC, сезонность, прогноз)
- `/dev` — задача для DEVBOT (автономный разработчик)
- Режим работника — очередь сборки, чеклисты, push-уведомления

### Многоагентная система (LangGraph)
- **Оркестратор** — классификация интента: consult / order / edit / track / stats / greeting
- **Консультант** — FAISS-поиск → LLM (70% семантика + 30% стилометрия)
- **Логист** — FSM оформления заказа → CRM
- **Аналитик** — ABC-анализ, сезонность, прогноз спроса
- **Инспектор** — диагностический диалог на основе KB
- **Ассистент** — /admin: LLM + CrmSnapshot

### Веб-панель (14 страниц, PWA)
- Дашборд с графиками
- Заказы, клиенты, товары (CRUD)
- Склад, партии отправки, сборка (offline)
- Экспорт CSV, SSE-уведомления, JWT-авторизация

### База знаний
- 276 чанков (20 текстов + 26 YouTube-расшифровок)
- Гибридный поиск: FAISS + стилометрия + keyword-буст
- 74 симптома + 77 показаний (онтология из CRM)

### CRM-интеграция (Integram)
- **v2 (ai2o.online)** — основная CRM (85 товаров, чистые данные)
- **v1 (ai2o.ru)** — архив (1924 клиента, 1915 заказов)
- UDS-синхронизация, авто-трекинг (СДЭК + Почта), SSE-события

---

## Технологии

| Компонент | Технология |
|-----------|-----------|
| Язык | Python 3.12 (asyncio) |
| Telegram | aiogram 3.25 |
| Оркестратор | LangGraph (StateGraph) |
| LLM | Groq API (llama-3.3-70b-versatile) |
| Эмбеддинги | fastembed (paraphrase-multilingual-MiniLM-L12-v2) |
| Векторный поиск | FAISS (IndexFlatIP) |
| CRM | Integram (ai2o.online API v2 + ai2o.ru API v1) |
| Веб-API | FastAPI + uvicorn + SSE |
| Frontend | Vue 3, PrimeVue 4, Vite, PWA |
| Тесты | pytest + pytest-asyncio (32 файла, 8028 строк) |
| CI/CD | GitHub Actions (ruff + pytest + deploy) |
| Инфраструктура | Docker, docker-compose, systemd |

---

## Структура проекта

```
BEEBOT/
├── src/
│   ├── bot.py                     # Telegram-бот: инициализация, роутеры
│   ├── orchestrator.py            # LangGraph — 6 интентов
│   ├── config.py                  # Конфигурация (.env)
│   ├── models.py                  # Pydantic-модели
│   ├── agents/
│   │   ├── beebot.py              # Консультант: FAISS → LLM
│   │   ├── logist.py              # Логист: FSM → заказ в CRM
│   │   ├── analyst.py             # Аналитик: ABC, сезонность, прогноз
│   │   ├── inspector.py           # Инспектор: «Осмотр улья»
│   │   ├── admin_chat.py          # Ассистент пчеловода
│   │   └── worker.py              # Очередь сборки
│   ├── integram_client.py         # CRM v1 (ai2o.ru, read-only архив)
│   ├── integram_v2_client.py      # CRM v2 (ai2o.online, основная)
│   ├── integram_v2_constants.py   # ID таблиц и колонок v2
│   ├── crm_constants.py           # ID таблиц и колонок v1
│   ├── knowledge_base.py          # FAISS + стилометрия
│   ├── llm_client.py              # Groq API + 5 голосов
│   ├── delivery/                  # СДЭК + Почта России
│   ├── web/
│   │   ├── api.py                 # FastAPI: роутеры + SSE
│   │   └── routers/               # auth, orders, clients, products...
│   └── devbot/                    # Автономный разработчик
├── web/                           # Frontend (Vue 3 + PrimeVue)
├── tests/                         # 32 файла, 8028 строк
├── data/                          # KB: pdfs, texts, subtitles, FAISS-индекс
├── docs/                          # Архитектурные диаграммы
├── docker-compose.yml
└── .github/workflows/ci.yml
```

---

## Запуск

```bash
# Docker (production)
docker compose up -d --build

# Локальная разработка
pip install -r requirements.txt
python -m src.bot       # Telegram-бот
uvicorn src.web.api:app # Веб-панель
```

### Переменные окружения

```env
# Telegram
TELEGRAM_BOT_TOKEN=...
ADMIN_CHAT_ID=...

# LLM
GROQ_API_KEY=...

# CRM v2 (основная)
INTEGRAM_V2=true
INTEGRAM_V2_EMAIL=...
INTEGRAM_V2_PASSWORD=...
INTEGRAM_V2_WORKSPACE=alekseymavai

# CRM v1 (архив)
INTEGRAM_URL=https://ai2o.ru
INTEGRAM_LOGIN=...
INTEGRAM_PASSWORD=...
INTEGRAM_DB=bibot
```

---

## Документация

| Документ | Описание |
|----------|----------|
| [analysis.md](analysis.md) | Анализ: сильные/слабые стороны, конфликты логик |
| [plan.md](plan.md) | План развития: 5 направлений, задачи по приоритетам |
| [docs/architecture.md](docs/architecture.md) | Архитектура: диаграммы, CRM, агенты, деплой |

---

## Тесты

```bash
pytest                          # все тесты
pytest tests/test_bot.py        # только бот
pytest -k "integram"            # CRM-тесты
```

---

# BEEBOT (English)

**Digital beekeeper assistant** — Telegram bot + web panel for order management at "Usadba Dmitrovykh" (Dmitrov Estate).

## Features
- **Telegram bot** — product consultations, order placement (7-step FSM), hive inspection, sales analytics
- **Multi-agent system** (LangGraph) — 6 agents: consultant, logistician, analyst, inspector, admin assistant, warehouse worker
- **Knowledge base** — 276 chunks, hybrid search (FAISS + stylometry), 5 voice styles
- **Web panel** (14 pages, PWA) — dashboard, orders, clients, products, stock, batches, offline packing
- **CRM** — Integram v2 (ai2o.online) + v1 archive (ai2o.ru)
- **Auto-tracking** — CDEK + Russian Post every 2 hours
- **DEVBOT** — autonomous developer via /dev (Claude API)

## Tech Stack
Python 3.12 | aiogram 3 | LangGraph | Groq (llama-3.3-70b) | FAISS | FastAPI | Vue 3 | PrimeVue | Docker | GitHub Actions

## Quick Start
```bash
docker compose up -d --build
```

## Documentation
- [analysis.md](analysis.md) — Project analysis (strengths, weaknesses, logic conflicts)
- [plan.md](plan.md) — Development roadmap (5 directions, prioritized tasks)
- [docs/architecture.md](docs/architecture.md) — Architecture diagrams
- [docs/architecture.md](docs/architecture.md) — Architecture diagrams (agents, CRM, deploy)
