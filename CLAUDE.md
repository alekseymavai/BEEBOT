# BEEBOT — Инструкция для Claude

## Что это за проект

**BEEBOT** — Telegram-помощник + веб-панель управления заказами для «Усадьба Дмитровых». Это **цифровой двойник пчеловода** — отвечает на вопросы подписчиков, принимает заказы, управляет складом и доставкой.

**Целевая аудитория:** подписчики блога Александра Дмитрова, покупатели продуктов пчеловодства.

**Бот в Telegram:** @AleksandrDmitrov_BEEBOT
**Веб-панель:** http://185.233.200.13:8088

## Текущие возможности (production)

### 1. База знаний «Пасека»
- **PDF-инструкции** — 19 документов (прополис, перга, ПЖВМ, гомогенат и др.)
- **Тексты** — 21 файл (очищенные выдержки)
- **Субтитры YouTube** — расшифровки 26 видео с канала @a.dmitrov
- **Гибридный поиск** — 70% семантика + 30% стилометрия + keyword-буст
- ⚠️ В индексе сейчас **240 чанков** (нужна пересборка: `docker exec beebot python -m src.build_kb`)

### 2. Telegram-бот
- `/start`, `/products`, `/ask`, `/help` — базовые команды
- `/order` — запуск FSM оформления заказа (7 шагов)
- `/inspect` — «Осмотр улья»: диагностический диалог, 3 вопроса → рекомендация
- `/voice` — выбор «Голоса Улья» (5 стилей: основатель, наставник, краевед, учёный, молодой)
- `/cancel` — прервать диалог заказа
- Работа в группах (@mention или reply)
- История диалога (5 пар сообщений, TTL 30 мин)
- Контекстные кнопки после ответа (PDF-инструкция + «Все продукты»)

### 3. Многоагентная система (LangGraph)
- **Оркестратор** — классифицирует интент: consult / order / delivery / status / edit / greeting
- **Консультант** — FAISS-поиск → LLM-ответ в стиле автора
- **Логист** — 7-шаговый FSM: товары → ФИО → телефон → адрес → доставка → подтверждение → CRM
- **Аналитик** — статистика продаж, топ товаров (только ADMIN_CHAT_ID)
- **Инспектор** — «Осмотр улья», использует KB без CRM

### 4. Веб-панель (PWA)
- **Дашборд** — 6 карточек + 4 графика (выручка, заказы, статусы, доставка)
- **Заказы** — список, детали, смена статуса, трекинг, создание; пагинация + поиск + фильтры
- **Клиенты** — список с историей заказов; пагинация + поиск
- **Товары** — CRUD, управление остатками; пагинация
- **Сборка** (PWA offline) — чеклист сборки заказов на пасеке
- **Склад** (PWA offline) — учёт остатков, +/− кнопки, алерты при низком запасе
- **Журнал** — заказы по месяцам
- Экспорт CSV, SSE-уведомления, JWT-авторизация, rate limiting

### 5. CRM-интеграция (Integram bibot)
- База `bibot` на ai2o.ru: 76 товаров · 285+ клиентов · 326+ заказов
- Таблицы: Товары, Клиенты, Заказы, Позиции заказа
- UDS-синхронизация: поллинг каждые 5 мин, catch-up с 17.03.2026, 230+ транзакций
- Авто-трекинг: СДЭК + Почта России каждые 2 часа

### 6. LLM
- Модель: `llama-3.3-70b-versatile` через Groq API (SSH-туннель hive:8990)
- Retry-логика (3 попытки с backoff)
- 5 голосовых стилей с разными системными промптами

## Технологический стек

| Компонент | Технология |
|-----------|-----------|
| Язык | Python 3, asyncio |
| Telegram | aiogram 3.25 |
| Оркестратор | LangGraph (StateGraph) |
| LLM | Groq API (llama-3.3-70b-versatile) |
| Эмбеддинги | fastembed (paraphrase-multilingual-MiniLM-L12-v2) |
| Векторный поиск | FAISS (IndexFlatIP, cosine similarity) |
| Чанкинг | langchain-text-splitters |
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
│   ├── bot.py                  # Telegram-бот: хэндлеры, FSM, startup
│   ├── orchestrator.py         # LangGraph — 6 интентов + история диалога
│   ├── config.py               # Конфигурация из .env
│   ├── models.py               # Pydantic-модели (Order, Client, Product)
│   ├── crm_constants.py        # CRM ID-маппинг (статусы, источники, категории) ← единый источник
│   ├── crm_schema.py           # Схема таблиц и реквизитов CRM
│   ├── phone_utils.py          # Валидация телефона (+7/8/9xx → +7XXXXXXXXXX)
│   ├── logging_config.py       # JSON-логирование
│   ├── agents/
│   │   ├── beebot.py           # Консультант: FAISS → LLM
│   │   ├── logist.py           # Логист: FSM 7 шагов → заказ в CRM
│   │   ├── analyst.py          # Аналитик: статистика из CRM
│   │   └── inspector.py        # Инспектор: «Осмотр улья»
│   ├── knowledge_base.py       # FAISS + стилометрия + keyword-буст
│   ├── llm_client.py           # Groq API (retry + Голос Улья)
│   ├── integram_api.py         # CRM HTTP-клиент низкоуровневый (auto re-auth)
│   ├── integram_client.py      # CRM обёртка высокоуровневая (Pydantic-модели)
│   ├── admin.py                # Админ-команды (/orders, /status, /track, /clients, /stock)
│   ├── notifications.py        # Уведомления пчеловоду в Telegram
│   ├── delivery/
│   │   ├── calculator.py       # Калькулятор (СДЭК + Почта + самовывоз)
│   │   ├── cdek.py             # СДЭК API v2 (OAuth2 + tariff + tracking)
│   │   ├── pochta.py           # Почта России (tariff + tracking)
│   │   └── tracker.py          # Авто-трекинг каждые 2 часа
│   ├── integrations/
│   │   └── uds.py              # UDS: поллер + дедупликация → CRM + уведомления
│   └── web/
│       ├── api.py              # FastAPI: JWT, CRUD, пагинация, SSE, CSV, healthcheck
│       ├── notifications.py    # Уведомления клиентам при смене статуса
│       ├── users.py            # Управление пользователями веб-панели
│       └── server.py           # Статика + PWA root files
├── web/                        # Frontend (Vue 3 + PrimeVue 4, PWA)
│   └── src/
│       ├── views/              # 11 страниц
│       ├── components/         # AppLayout, StatCard, StatusBadge, OrderItemsTable
│       ├── stores/             # Pinia (auth) + offline.js (IndexedDB + sync queue)
│       ├── router/             # Vue Router (auth guard)
│       ├── api.js              # axios + JWT interceptor
│       └── utils.js            # formatDate, formatMoney
├── tests/                      # 284 теста (pytest)
├── data/
│   ├── pdfs/                   # 19 PDF-инструкций
│   ├── texts/                  # 21 текстовый источник
│   ├── subtitles/              # 26 расшифровок YouTube
│   └── processed/              # FAISS-индекс + chunks.json
├── docs/
│   └── architecture.md         # Блок-схемы, потоки данных, сравнительные таблицы
├── systemd/
│   ├── groq-proxy.service      # Reverse proxy hive:8990 → api.groq.com
│   ├── groq-tunnel.service     # SSH-туннель VPS↔hive (порты 8990 и 9150)
│   ├── tg-socks.service        # SOCKS5-прокси для Telegram API (порт 9150)
│   └── install-hive-services.sh
├── .github/workflows/ci.yml   # CI/CD: ruff + pytest + deploy
├── Dockerfile                  # Бот (Python + FAISS + fastembed)
├── Dockerfile.web              # Веб-панель (Node build → Python serve)
├── docker-compose.yml          # 2 сервиса: beebot (host) + beebot-web (8088)
├── groq_proxy.py               # Reverse proxy (hive:8990 → api.groq.com)
├── tg_socks_proxy.py           # SOCKS5-сервер для Telegram API (hive:9150)
├── analysis.md                 # Анализ: проблемы, конфликты, инфраструктура
├── plan.md                     # План развития: фазы 3–5
└── .env.example
```

## Архитектура

```
Пользователи
  → Telegram API
  → aiogram polling (VPS Docker, network_mode: host)
  → SOCKS5 прокси (hive:9150) ← tg-socks.service ← groq-tunnel -R 9150

Telegram-бот
  → Оркестратор (LangGraph) → classify intent
  → Консультант → FAISS → LLM
  → Логист     → FSM → Integram CRM
  → Аналитик   → Integram CRM → отчёты
  → Инспектор  → FAISS → LLM

LLM-цепочка:
  beebot → localhost:8990 → SSH-туннель → hive:8990 → groq-proxy → api.groq.com

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
| hive | локальная машина | groq-proxy + groq-tunnel + tg-socks (systemd) |
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

# Полный редеплой (только документация — без пересборки образов)
ssh ai-agent@185.233.200.13 "cd /home/ai-agent/BEEBOT && git pull"

# Редеплой с пересборкой (при изменении кода)
ssh ai-agent@185.233.200.13 "cd /home/ai-agent/BEEBOT && git pull && docker compose up -d --build"

# Пересобрать базу знаний (нужно: 240 чанков → 410+)
ssh ai-agent@185.233.200.13 "docker exec beebot python -m src.build_kb"

# Пересобрать только веб-панель
ssh ai-agent@185.233.200.13 "docker compose up -d --build beebot-web"
```

### Workflow деплоя (нет push-доступа к upstream)

```bash
# 1. Коммит в локальный main
git add ... && git commit -m "feat: ..."

# 2. Push в fork
git push origin main  # origin = unidel2035/BEEBOT

# 3. PR в upstream
gh pr create --repo alekseymavai/BEEBOT --head unidel2035:main --base main ...

# 4. Мерж (если есть права) или ждать
gh pr merge <N> --repo alekseymavai/BEEBOT --squash

# 5. Pull на VPS
ssh ai-agent@185.233.200.13 "cd /home/ai-agent/BEEBOT && git pull origin main"
```

## Известные проблемы (актуально на 21.03.2026)

Полный анализ: [analysis.md](analysis.md)

### P1 — требуют внимания
- **Уведомления не синхронизированы** — смена статуса через веб не уведомляет пчеловода в TG
- **KB устарела** — 240 чанков вместо 410+, нужна пересборка

### P2 — технический долг
- **Монолитный api.py** — 1242 строки, нужно разбить на модули
- **Пагинация фронтенда** — `per_page=1000` вместо серверной пагинации
- **Константы в 3 файлах** — crm_constants.py + crm_schema.py + integram_api.py

### Инфраструктурные ограничения
1. **Groq блокирует IP VPS** → решено SSH-туннелем + прокси на hive (groq-proxy.service)
2. **Telegram API блокирует IP VPS** → решено SOCKS5 через hive (tg-socks.service)
3. **YouTube блокирует IP** → субтитры скачаны заранее в `data/subtitles/`
4. **unidel2035** — нет push-доступа к upstream → только fork + PR workflow
5. **SSH-алиас `beebot-vps`** — не работает (DNS), использовать прямой IP: `ssh ai-agent@185.233.200.13`
6. **VPS RAM 2 GB** — при добавлении Claude API потребуется апгрейд до 4 GB

## Документация проекта

| Документ | Описание |
|----------|----------|
| [analysis.md](analysis.md) | Анализ 21.03.2026: сильные/слабые стороны, конфликты логик, инфраструктура |
| [plan.md](plan.md) | Фазы 3–5: Claude API, долгосрочная память, AgentBus, масштабирование |
| [docs/architecture.md](docs/architecture.md) | Блок-схемы всех подсистем, сравнительные таблицы |
| [README.md](README.md) | Обзор проекта (RU + EN) |

## Для ИИ-ассистента

Ты помогаешь с:
1. **Развитием бота** — новые фичи, рефакторинг, багфиксы
2. **Системой заказов** — оркестратор, агенты, Integram CRM
3. **Веб-панелью** — страницы, графики, PWA-улучшения
4. **Базой знаний** — новые источники, пересборка KB
5. **Инфраструктурой** — Docker, деплой, мониторинг

### Принципы разработки
- **Русский язык** в интерфейсе, коммитах, документации
- **Простота** — пчеловод должен понимать что происходит
- **Надёжность** — retry-логика обязательна на всех внешних вызовах
- **Стиль автора** — ответы бота должны звучать как Александр Дмитров
- **Не фантазируй** — бот отвечает только на основе базы знаний
- **Fork workflow** — PR через unidel2035/BEEBOT → alekseymavai/BEEBOT
- **Не дублируй** — перед созданием кода проверь аналоги в проекте
- **Актуальность** — при изменении архитектуры обновляй CLAUDE.md, analysis.md, docs/architecture.md
- **Foreground Bash сломан** — использовать `run_in_background=true` + `TaskOutput`
- **SSH к VPS** — `ssh ai-agent@185.233.200.13`, алиас `beebot-vps` не работает
