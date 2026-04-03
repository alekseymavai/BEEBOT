# BEEBOT — Инструкция для Claude

## Что это за проект

**BEEBOT** — Telegram-помощник + веб-панель управления заказами для «Усадьба Дмитровых». Это **цифровой двойник пчеловода** — отвечает на вопросы подписчиков, принимает заказы, управляет складом и доставкой, а также включает DEVBOT — автономного разработчика.

**Целевая аудитория:** подписчики блога Александра Дмитрова, покупатели продуктов пчеловодства.

**Бот в Telegram:** @AleksandrDmitrov_BEEBOT
**Веб-панель:** http://185.233.200.13:8088

## Текущие возможности (production, 03.04.2026)

### 1. База знаний «Пасека»
- **PDF-инструкции** — 19 документов (прополис, перга, ПЖВМ, гомогенат и др.)
- **Тексты** — 20 файлов (очищенные выдержки)
- **Субтитры YouTube** — расшифровки 26 видео с канала @a.dmitrov
- **Гибридный поиск** — 70% семантика + 30% стилометрия + keyword-буст из CRM-каталога
- В индексе **276 чанков** (20 txt + 26 YouTube; PDFs перекрыты текстами)

### 2. Telegram-бот (src/bot.py — 1 899 строк)
- `/start`, `/products`, `/ask`, `/help` — базовые команды + **ReplyKeyboard** главное меню
- `/order` — запуск FSM оформления заказа (7 шагов)
- `/inspect` — «Осмотр улья»: диагностический диалог, 3 вопроса → рекомендация
- `/voice` — выбор «Голоса Улья» (5 стилей: основатель, наставник, краевед, учёный, молодой)
- `/cancel` — прервать диалог заказа
- `/admin` — режим личного ассистента пчеловода + переключение режимов admin/worker (только ADMIN_CHAT_ID)
- `/stats [запрос]` — аналитика продаж (только ADMIN_CHAT_ID)
- `/faq` — топ частых вопросов пользователей (только ADMIN_CHAT_ID)
- `/yt_check`, `/yt_update` — YouTube KB-обновление (только ADMIN_CHAT_ID)
- `/dev <задача>` — поставить задачу DEVBOT (только ADMIN_CHAT_ID)
- **Режим работника** — `/start` для WORKER_CHAT_IDS: очередь сборки заказов, чеклист, push-уведомления
- Работа в группах (@mention или reply)
- История диалога (5 пар сообщений, TTL 30 мин)

### 3. Многоагентная система (LangGraph)
- **Оркестратор** — классифицирует интент: consult / order / edit / track / stats / greeting
- **Консультант** — FAISS-поиск → LLM-ответ в стиле автора + SQLite-память + онтология
- **Логист** — 7-шаговый FSM: товары → ФИО → телефон → адрес → доставка → подтверждение → CRM
- **Аналитик** — статистика продаж: ABC/сезонность/прогноз (только ADMIN_CHAT_ID)
- **Инспектор** — «Осмотр улья», использует KB без CRM (только через /inspect)
- **Ассистент** (AdminChatAgent) — /admin: LLM-диалог + **CrmSnapshot** (кэш, обновляется каждые 5 мин)
- **WorkerAgent** — кнопочная очередь сборки заказов для WORKER_CHAT_IDS

### 4. DEVBOT — автономный разработчик (src/devbot/, запуск на hive)
- `/dev <задача>` в BEEBOT → HTTP POST → DEVBOT API hive:8091
- FSM: IDLE → ANALYZING (Claude API) → CONFIRMING → EXECUTING (claude CLI) → FEEDBACK
- Auto-continue через `--resume <session_id>` при длинных задачах
- Память 2 уровня: файлы `memory/` + Integram таблицы DEV_TASKS, DEV_MEMORY, DEV_ADVICE
- Команды: `/devstatus`, `/devhistory`, `/devmemory`

### 5. Веб-панель (PWA, 14 страниц)
- **Дашборд** — 6 карточек + 4 графика + expandable rows с составом заказов
  - Переключатель периода: Сегодня / 7д / 30д / Квартал / Всё время
  - Блок «Требуют внимания»: новые >24ч без ответа, подтверждённые >3д без отправки, низкий склад
  - Топ-5 товаров за период (qty + выручка + доля %)
- **Заказы** — список, детали, смена статуса, трекинг, создание, история статусов, чеклист
  - Поиск по номеру / клиенту / трек-номеру (debounce)
  - Фильтр по диапазону дат (DatePicker range)
  - Цветные полосы строк по статусу (border-left)
  - Инлайн-смена статуса прямо в строке таблицы
  - Групповые действия: выбор нескольких заказов → batch-статус
  - Канбан-вид с drag & drop сменой статуса (переключатель Список/Канбан)
- **Навигация** — бейджи-счётчики: новые заказы, низкий склад (polling 2 мин)
- **Клиенты** — список с историей заказов, объединение дублей
- **Товары** — CRUD, управление остатками
- **Партии отправки** — группировка заказов для массовой отправки
- **Сборка** (PWA offline) — чеклист сборки заказов на пасеке
- **Склад** (PWA offline) — учёт остатков, +/− кнопки, алерты при низком запасе
- **Журнал** — заказы по месяцам
- Экспорт CSV, SSE-уведомления, JWT-авторизация, rate limiting

### 6. CRM-интеграция (Integram bibot)
- База `bibot` на ai2o.ru: 76 товаров · 1924 клиента · 1915 заказов
- Таблицы: Товары, Клиенты, Заказы, Позиции заказа, История статусов, Здоровье, Партии, DEVBOT (3 таблицы)
- UDS-синхронизация: поллинг каждые 5 мин, catch-up с **01.01.2024**, дедупликация
- Авто-трекинг: СДЭК + Почта России каждые 2 часа
- `crm_constants.py` — единый источник всех CRM ID (таблицы, реквизиты, lookup-справочники)
- **Онтология**: 74 симптома + 77+ показаний к применению (Integram → OntologyCache)

### 7. LLM
- **Бот**: `llama-3.3-70b-versatile` через Groq API (SSH-туннель hive:8990)
- **DEVBOT**: `claude-sonnet-4-6` через Anthropic API (analyzer) + Claude Code CLI (executor)
- Retry-логика (3 попытки с backoff) на всех LLM-вызовах
- 5 голосовых стилей с разными системными промптами

## Технологический стек

| Компонент | Технология |
|-----------|-----------|
| Язык | Python 3.12, asyncio |
| Telegram | aiogram 3.25 |
| Оркестратор | LangGraph (StateGraph) |
| LLM (бот) | Groq API (llama-3.3-70b-versatile) |
| LLM (DEVBOT) | Anthropic API (claude-sonnet-4-6) + Claude Code CLI |
| Эмбеддинги | fastembed (paraphrase-multilingual-MiniLM-L12-v2) |
| Векторный поиск | FAISS (IndexFlatIP, cosine similarity) |
| Память | SQLite (долгосрочная) + in-memory (диалоги TTL 30 мин) |
| CRM | Integram (ai2o.ru/bibot) — REST API через httpx |
| Веб-API | FastAPI + uvicorn + SSE + slowapi |
| Frontend | Vue 3, PrimeVue 4, Vite, PWA (vite-plugin-pwa) |
| Offline | Service Worker + IndexedDB |
| Валидация | Pydantic 2 |
| Тесты | pytest + pytest-asyncio (284 теста) |
| CI/CD | GitHub Actions (lint + test + deploy) |
| Инфраструктура | Docker, docker-compose, systemd |
| VPS | 185.233.200.13, 2 GB RAM + 2 GB swap |

## Структура проекта

```
BEEBOT/
├── src/
│   ├── bot.py                  # Telegram-бот: хэндлеры, FSM, startup (1 899 строк — монолит!)
│   ├── orchestrator.py         # LangGraph — 6 интентов + история + FAQ
│   ├── config.py               # Конфигурация из .env
│   ├── models.py               # Pydantic-модели (Order, Client, Product, OrderItem)
│   ├── crm_constants.py        # ← ЕДИНЫЙ источник всех CRM ID
│   ├── crm_schema.py           # Схема таблиц и реквизитов CRM (документация)
│   ├── crm_snapshot.py         # CrmSnapshot — кэш CRM с TTL (обновляется каждые 5 мин)
│   ├── memory.py               # UserMemory — SQLite-хранилище фактов пользователей
│   ├── ontology.py             # OntologyCache — симптомы → показания (Integram)
│   ├── phone_utils.py          # Валидация телефона (+7/8/9xx → +7XXXXXXXXXX)
│   ├── agents/
│   │   ├── beebot.py           # Консультант: FAISS → LLM
│   │   ├── logist.py           # Логист: FSM 7 шагов → заказ в CRM
│   │   ├── analyst.py          # Аналитик: статистика + ABC + сезонность + прогноз
│   │   ├── inspector.py        # Инспектор: «Осмотр улья» (только /inspect, не в оркестраторе!)
│   │   ├── admin_chat.py       # Ассистент пчеловода: /admin + CrmSnapshot
│   │   └── worker.py           # WorkerAgent: очередь сборки, чеклист (in-memory!)
│   ├── devbot/                 # DEVBOT — автономный разработчик (запуск только на hive)
│   │   ├── bot.py              # aiogram polling + FastAPI :8091
│   │   ├── analyzer.py         # Claude API → план изменений
│   │   ├── executor.py         # claude CLI (stream-json + --resume auto-continue)
│   │   ├── memory.py           # Integram DEV_TASKS + DEV_MEMORY + DEV_ADVICE
│   │   └── prompts.py          # build_system_prompt + build_user_prompt
│   ├── knowledge_base.py       # FAISS + стилометрия (70/30) + keyword-буст
│   ├── youtube_updater.py      # Автообновление KB из YouTube
│   ├── llm_client.py           # Groq API (retry + Голос Улья)
│   ├── integram_api.py         # CRM HTTP-клиент низкоуровневый (auto re-auth)
│   ├── integram_client.py      # CRM обёртка высокоуровневая (Pydantic-модели)
│   ├── admin.py                # Админ-команды (/orders, /status, /track, /teach...)
│   ├── notifications.py        # Notifier: пчеловод + клиенты + работники склада
│   ├── delivery/
│   │   ├── calculator.py       # Калькулятор (СДЭК + Почта + самовывоз)
│   │   ├── cdek.py             # СДЭК API v2 (OAuth2 + tariff + tracking)
│   │   ├── pochta.py           # Почта России (tariff + tracking)
│   │   └── tracker.py          # Авто-трекинг каждые 2 часа
│   ├── integrations/
│   │   └── uds.py              # UDS: поллер + catch-up с 01.01.2024 + дедупликация → CRM
│   └── web/
│       ├── api.py              # FastAPI: main router + startup (183 строки)
│       ├── routers/            # 9 маршрутных модулей
│       │   ├── auth.py / orders.py / clients.py / products.py
│       │   ├── dashboard.py / batches.py / export.py / users.py / sse.py
│       ├── notifications.py    # notify_beekeeper_status_change
│       ├── users.py            # Управление пользователями веб-панели
│       └── server.py           # Статика + PWA root files
├── web/                        # Frontend (Vue 3 + PrimeVue 4, PWA)
│   └── src/
│       ├── views/              # 14 страниц
│       ├── components/         # AppLayout, StatCard, StatusBadge, OrderItemsTable
│       ├── stores/             # Pinia (auth) + offline.js (IndexedDB + sync queue)
│       ├── api.js              # axios + JWT interceptor
│       └── utils.js            # formatDate, formatMoney
├── tests/                      # 284 теста (pytest)
├── data/
│   ├── pdfs/                   # 19 PDF-инструкций
│   ├── texts/                  # 20 текстовых источников
│   ├── subtitles/              # 26 расшифровок YouTube
│   └── processed/              # FAISS-индекс + chunks.json (276 чанков)
├── docs/
│   └── architecture.md         # 11 Mermaid-диаграмм + 6 сравнительных таблиц
├── systemd/                    # systemd-сервисы для hive
├── .github/workflows/ci.yml   # CI/CD: ruff + pytest + deploy
├── Dockerfile / Dockerfile.web / docker-compose.yml
├── groq_proxy.py               # Reverse proxy hive:8990 → api.groq.com
├── tg_socks_proxy.py           # SOCKS5-сервер для Telegram API hive:9150
├── analysis.md                 # Анализ 29.03.2026: проблемы, конфликты, инфраструктура
└── plan.md                     # План развития: фазы 8–11
```

## Архитектура

```
Пользователи / Работники / Пчеловод
  → Telegram API → SOCKS5 (hive:9150) ← groq-tunnel

Telegram-бот (VPS Docker, network_mode: host)
  → Оркестратор (LangGraph) → consult/order/edit/track/stats/greeting
  → WorkerAgent (WORKER_CHAT_IDS) → очередь сборки
  → AdminChatAgent (ADMIN_CHAT_ID) → CrmSnapshot → LLM
  → InspectorAgent (/inspect) → FAISS → LLM
  → UDSPoller (каждые 5 мин) → UDS Partner API → sync → CRM + уведомления
  → OrderTracker (каждые 2 ч) → СДЭК / Почта России → обновление статусов
  → DEVBOT (/dev → HTTP localhost:8091 → hive)

LLM-цепочка (бот):
  beebot → localhost:8990 → SSH-туннель → hive:8990 → groq-proxy → api.groq.com

DEVBOT-цепочка (hive):
  /dev → DEVBOT analyzer → Anthropic API → план
  → executor → claude --output-format stream-json → деплой на VPS

Веб-панель (Vue 3 + PrimeVue, PWA)
  → FastAPI (порт 8088) → Integram CRM (ai2o.ru/bibot)
  → JWT-авторизация, SSE real-time, CSV export
```

Подробные диаграммы: [docs/architecture.md](docs/architecture.md)

## Инфраструктура и деплой

| Ресурс | Адрес | Детали |
|--------|-------|--------|
| VPS | 185.233.200.13 | ai-agent, Docker, 2 GB RAM + 2 GB swap |
| Веб-панель | http://185.233.200.13:8088 | FastAPI + Vue PWA |
| hive | локальная машина | groq-proxy + groq-tunnel + tg-socks + DEVBOT (systemd) |
| SSH (прямой) | `ssh ai-agent@185.233.200.13` | SSH-алиас `beebot-vps` не работает (DNS) |
| CRM | ai2o.ru/bibot | Integram, база `bibot` |
| GitHub upstream | [alekseymavai/BEEBOT](https://github.com/alekseymavai/BEEBOT) | Основной репозиторий |
| GitHub fork | [unidel2035/BEEBOT](https://github.com/unidel2035/BEEBOT) | PR через fork |

### Быстрые команды

```bash
# Статус hive-сервисов
systemctl status groq-proxy groq-tunnel tg-socks

# Логи бота (с VPS)
ssh ai-agent@185.233.200.13 "docker logs --tail 30 beebot"
ssh ai-agent@185.233.200.13 "docker logs --tail 10 beebot-web"

# Редеплой только документации (без пересборки образов)
ssh ai-agent@185.233.200.13 "cd /home/ai-agent/BEEBOT && git pull"

# Редеплой с пересборкой Python-кода
ssh ai-agent@185.233.200.13 "cd /home/ai-agent/BEEBOT && git pull && docker compose up -d --build"

# Принудительная пересборка конкретного сервиса
ssh ai-agent@185.233.200.13 "cd /home/ai-agent/BEEBOT && docker compose up -d --build --force-recreate beebot"

# Пересобрать базу знаний (276 чанков)
ssh ai-agent@185.233.200.13 "docker exec beebot python -m src.build_kb"
```

### Workflow деплоя (нет push-доступа к upstream)

```bash
# 1. Коммит на ветке pr-tech-debt или новой ветке
git add ... && git commit -m "feat: ..."

# 2. Push в fork (переключиться на unidel2035 если нужно)
gh auth switch -u unidel2035 && gh auth setup-git
git push origin <branch>

# 3. PR в upstream
gh pr create --repo alekseymavai/BEEBOT --head unidel2035:<branch> --base main ...

# 4. Мерж (squash)
gh pr merge <N> --repo alekseymavai/BEEBOT --squash

# 5. VPS: сбросить на main (после squash-мержа нужен reset, не pull)
ssh ai-agent@185.233.200.13 "cd /home/ai-agent/BEEBOT && git fetch origin main && git reset --hard origin/main"

# 6. Пересборка
ssh ai-agent@185.233.200.13 "cd /home/ai-agent/BEEBOT && docker compose up -d --build --force-recreate beebot"

# 7. Вернуть gh auth
gh auth switch -u gaveron18
```

## Известные проблемы (актуально на 29.03.2026)

Полный анализ: [analysis.md](analysis.md)

### P1 — рефакторинг (фаза 8)
- **bot.py монолит** — 1 899 строк, все роли в одном файле → нужны aiogram Router-модули
- **AdminChatAgent дублирование** — `_get_crm_context` + `_build_context_from_snapshot` ~330 строк идентичного кода
- **Инспектор вне оркестратора** — нет `inspect` интента, только через /inspect
- **UDS SKU** — товары ищутся по имени, а не по SKU → product_id=0 для многих позиций

### P2 — технический долг
- Worker-чеклист в RAM — теряется при рестарте бота
- DEVBOT FastAPI без авторизации (allow_origins=["*"])

### Инфраструктурные ограничения
1. **Groq блокирует IP VPS** → решено SSH-туннелем + прокси на hive (groq-proxy.service)
2. **Telegram API блокирует IP VPS** → решено SOCKS5 через hive (tg-socks.service)
3. **YouTube блокирует IP** → субтитры скачаны заранее в `data/subtitles/`
4. **unidel2035** — нет push-доступа к upstream → только fork + PR workflow
5. **SSH-алиас `beebot-vps`** — не работает (DNS), использовать прямой IP
6. **VPS RAM 2 GB** — beebot ~762 MiB, при росте потребуется апгрейд до 4 GB
7. **hive — SPOF** — Groq-прокси + Telegram SOCKS5 + DEVBOT — всё на hive
8. **Squash-мерж** — после PR-мержа VPS нужен `git reset --hard origin/main`, не `git pull`

## Документация проекта

| Документ | Описание |
|----------|----------|
| [analysis.md](analysis.md) | Анализ 29.03.2026: сильные/слабые стороны, конфликты логик, инфраструктура |
| [plan.md](plan.md) | Фазы 8–11: рефакторинг, аналитика, инфраструктура, экосистема |
| [docs/architecture.md](docs/architecture.md) | 9 Mermaid-диаграмм + 5 сравнительных таблиц |
| [README.md](README.md) | Обзор проекта (RU + EN) |

## Для ИИ-ассистента

Ты помогаешь с:
1. **Развитием бота** — новые фичи, рефакторинг, багфиксы
2. **Системой заказов** — оркестратор, агенты, Integram CRM
3. **Веб-панелью** — страницы, графики, PWA-улучшения
4. **Базой знаний** — новые источники, пересборка KB
5. **DEVBOT** — автономный разработчик, задачи через /dev
6. **Инфраструктурой** — Docker, деплой, мониторинг

### Принципы разработки
- **Русский язык** в интерфейсе, коммитах, документации
- **Простота** — пчеловод должен понимать что происходит
- **Надёжность** — retry-логика обязательна на всех внешних вызовах
- **Стиль автора** — ответы бота должны звучать как Александр Дмитров
- **Не фантазируй** — бот отвечает только на основе базы знаний
- **Fork workflow** — PR через unidel2035/BEEBOT → alekseymavai/BEEBOT
- **Squash-мерж** — после мержа на VPS нужен `git reset --hard origin/main`
- **Не дублируй** — перед созданием кода проверь аналоги в проекте
- **Актуальность** — при изменении архитектуры обновляй CLAUDE.md, analysis.md, docs/architecture.md
- **SSH к VPS** — `ssh ai-agent@185.233.200.13`, алиас `beebot-vps` не работает
- **gh auth** — рабочий аккаунт для push: `unidel2035`; переключать через `gh auth switch -u unidel2035 && gh auth setup-git`; после деплоя вернуть `gh auth switch -u gaveron18`
