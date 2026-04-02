# BEEBOT

**Цифровой помощник пчеловода** — Telegram-бот + веб-панель управления заказами для «Усадьба Дмитровых».

Telegram: [@AleksandrDmitrov_BEEBOT](https://t.me/AleksandrDmitrov_BEEBOT)
Веб-панель: http://185.233.200.13:8088

## Что умеет

### Telegram-бот
- Отвечает на вопросы в стиле пчеловода (гибридный поиск: FAISS + стилометрия + онтология)
- Принимает заказы (FSM-диалог: товары → ФИО → телефон → адрес → доставка → подтверждение)
- «Осмотр улья» — диагностический диалог с рекомендацией
- 5 голосовых стилей («Голос Улья»)
- Режим работника склада — очередь сборки заказов
- Аналитика продаж: ABC, сезонность, прогноз
- Личный ассистент пчеловода с CRM-снимком
- DEVBOT — автономный разработчик на Claude API

### Веб-панель (PWA)
- Дашборд: 6 карточек + 4 графика + PDF-отчёты
- Заказы: CRUD, история статусов, чеклист, партии отправки
- Клиенты: список, объединение дублей
- Товары: каталог, остатки
- Сборка и склад: offline-режим
- SSE-уведомления, JWT-авторизация

### Интеграции
- **Integram CRM** — 85 товаров, ~2800 клиентов, ~2000 заказов
- **UDS** — автоимпорт заказов из магазина с составом (sku_uds маппинг)
- **СДЭК + Почта России** — расчёт доставки, автотрекинг каждые 2 часа
- **Groq API** — llama-3.3-70b для консультаций
- **Яндекс Диск** — резервное копирование

## Архитектура

```
Пользователи → Telegram → [beebot] → Redis Streams → [beebot-backend]
                                                            ↓
                                                     Service Layer
                                                     (OrderService,
                                                      ConsultService)
                                                            ↓
                                               CRM · LLM · KB · Delivery

Пчеловод → Браузер → [beebot-backend] → Integram CRM
```

Три Docker-контейнера: `redis` + `beebot` + `beebot-backend`

Подробнее: [docs/architecture.md](docs/architecture.md)

## Стек

| Компонент | Технология |
|-----------|-----------|
| Язык | Python 3.12, asyncio |
| Telegram | aiogram 3.25 |
| Оркестратор | LangGraph |
| LLM | Groq API (llama-3.3-70b) |
| Эмбеддинги | fastembed (multilingual MiniLM) |
| Векторный поиск | FAISS |
| Шина событий | Redis Streams |
| CRM | Integram (ai2o.ru) |
| Веб-API | FastAPI + SSE |
| Frontend | Vue 3, PrimeVue 4, PWA |
| Тесты | pytest (354 теста) |
| CI/CD | GitHub Actions |
| Деплой | Docker Compose |

## Быстрый старт

```bash
git clone https://github.com/alekseymavai/BEEBOT.git
cd BEEBOT
cp .env.example .env
# Заполнить: TELEGRAM_BOT_TOKEN, GROQ_API_KEY, INTEGRAM_*, WEB_PASSWORD, WEB_SECRET

docker compose up -d
docker exec beebot python -m src.build_kb
```

## Тесты

```bash
pip install -r requirements.txt
pytest tests/ -x -q
```

## Документация

| Документ | Описание |
|----------|----------|
| [docs/architecture.md](docs/architecture.md) | Архитектура: диаграммы, потоки, сравнительные таблицы |
| [analysis.md](analysis.md) | Анализ: сильные/слабые стороны, проблемы, рекомендации |
| [plan.md](plan.md) | План развития: направления A–E, задачи по приоритетам |

---

# BEEBOT (English)

**Digital Beekeeper's Assistant** — Telegram bot + order management dashboard for "Usadba Dmitrovykh" beekeeping products.

## Features

### Telegram Bot
- Knowledge base Q&A (hybrid search: FAISS + stylometry + ontology)
- 7-step order dialog with delivery calculation (CDEK, Russian Post)
- "Hive Inspection" diagnostic dialog
- 5 voice styles, sales analytics (ABC, seasonality, forecast)
- Personal beekeeper assistant with CRM access
- DEVBOT — autonomous developer agent (Claude API)

### Web Dashboard (PWA)
- Orders, clients, products management
- Status history, checklists, shipment batches
- Offline assembly & stock terminals
- PDF reports, SSE notifications, JWT auth

### Integrations
- Integram CRM (85 products, ~2800 clients, ~2000 orders)
- UDS loyalty system (auto-import with product mapping)
- CDEK + Russian Post (rate calculation, auto-tracking)
- Groq API (llama-3.3-70b)
- Yandex Disk backup

## Architecture

Three Docker containers: `redis` + `beebot` + `beebot-backend`

Communication via Redis Streams (EventBus). Service Layer: OrderService, NotificationService.

Details: [docs/architecture.md](docs/architecture.md)

## Quick Start

```bash
git clone https://github.com/alekseymavai/BEEBOT.git
cd BEEBOT
cp .env.example .env
docker compose up -d
docker exec beebot python -m src.build_kb
```

## Documentation

- [docs/architecture.md](docs/architecture.md) — Architecture diagrams & flows
- [analysis.md](analysis.md) — Project analysis & recommendations
- [plan.md](plan.md) — Development roadmap
