# BEEBOT — План развития

> **Дата:** 30 марта 2026
> **Основан на:** [analysis.md](analysis.md) · [docs/architecture.md](docs/architecture.md)
> **Ориентир архитектуры:** Gift Protocol + dronedoc2026 AgentBus + Agent Card
>
> Фазы 8–12 переработаны. Старые приоритеты (апгрейд VPS, перенос прокси) сняты —
> beebot занимает 193 MiB после PR #109, RAM стабильна.

---

## Фазы 0–7: Завершены ✅

| Фазы | Содержание | Статус |
|------|-----------|--------|
| 0–2 | Стабилизация, качество кода, функциональность | ✅ |
| 3 | LLM-ассистент, память, онтология, пагинация | ✅ 25.03.2026 |
| 4 | История статусов, партии, роутеры, DEVBOT-таблицы | ✅ 27.03.2026 |
| 5 | WorkerAgent — режим сборки заказов | ✅ 29.03.2026 |
| 6 | CrmSnapshot — кэш CRM, UDS catch-up | ✅ 29.03.2026 |
| 7 | DEVBOT — автономный разработчик (MVP) | ✅ 29.03.2026 |

---

## Синхронизация репозитория и железа

> **Правило:** после каждого squash-мержа PR — обязательная синхронизация.
> `git pull` на VPS не работает после squash — только `reset --hard`.

### Текущее состояние (30.03.2026)

| Место | Статус | Последний коммит |
|-------|--------|-----------------|
| upstream/main | ✅ актуален | PR #117 feat: phases8-9 |
| hive (локально) | ✅ актуален | 5b46517 (reset --hard) |
| VPS | ✅ актуален | PR #117 задеплоен, bot polling |

### Процедура синхронизации (выполнять после каждого PR)

```bash
# 1. hive — сбросить на main
git fetch upstream main
git reset --hard upstream/main

# 2. VPS — сбросить на main
ssh ai-agent@185.233.200.13 "cd /home/ai-agent/BEEBOT && git fetch origin main && git reset --hard origin/main"

# 3. VPS — пересборка (если изменился Python-код)
ssh ai-agent@185.233.200.13 "cd /home/ai-agent/BEEBOT && docker compose up -d --build --force-recreate beebot"

# 4. VPS — только документы (git pull без пересборки)
ssh ai-agent@185.233.200.13 "cd /home/ai-agent/BEEBOT && git fetch origin main && git reset --hard origin/main"
```

### ⚠️ Нужно сделать сейчас

- [ ] hive: `git fetch upstream main && git reset --hard upstream/main`
- [ ] VPS: синхронизировать с main (PR #111)

---

## Фаза 8: Качество и стабильность (P1)

> Актуальна. Выполняется до начала Gift Protocol — иначе рефакторить монолит.

### 8.1 Разбить bot.py на Router-модули ✅ (PR #114, 30.03.2026)

**Проблема:** `src/bot.py` 1 899 строк — все роли в одном файле.

- [x] Создать `src/routers/` с 6 файлами (user, worker, inspect, fsm_order, bot_admin, keyboards)
- [x] Перенести хэндлеры из bot.py в соответствующие роутеры
- [x] bot.py → 213 строк (startup/shutdown + `dp.include_router()`)

### 8.2 Убрать дублирование AdminChatAgent ✅ (PR #117, 31.03.2026)

- [x] Выделить `_format_context(orders, products, clients) → str`
- [x] `_get_crm_context()` и `_build_context_from_snapshot()` делегируют форматирование

### 8.3 Добавить `inspect` в оркестратор ✅ (PR #117, 31.03.2026)

- [x] Добавить `inspect` в `_INTENT_SYSTEM` (orchestrator.py)
- [x] Фразы для `_fast_classify`: «осмотр улья», «диагностика», «осмотри»
- [x] inspect → passthrough → bot.py запускает InspectFSM

### 8.4 UDS: сопоставление товаров по SKU ✅ (PR #117, 31.03.2026)

- [x] `integram_client`: добавить `get_product_by_sku(sku_uds: str) → Product | None`
- [x] `_build_order_items()`: сначала по `good["sku"]`, fallback — по имени
- [ ] Синхронизировать `sku_uds` в Integram для всех 76 товаров (ручная работа)

### ✅ Синхронизация после Фазы 8

```bash
# hive
git fetch upstream main && git reset --hard upstream/main
# VPS — сброс + пересборка (Python-код изменился)
ssh ai-agent@185.233.200.13 "cd /home/ai-agent/BEEBOT \
  && git fetch origin main && git reset --hard origin/main \
  && docker compose up -d --build --force-recreate beebot"
```

---

## Фаза 9: Gift Protocol (P1 — новая архитектура)

> Три шага эволюции: не переписывание, а наслоение поверх существующего кода.
> Подробнее: [docs/architecture.md §10](docs/architecture.md)

### 9.1 SharedContext — рабочая память (шаг 1) ✅ (PR #115, 30.03.2026)

- [x] Создать `src/shared_context.py` (UserContext + SharedContextStore, ~75 строк)
- [x] Заменить `_histories` в оркестраторе на SharedContextStore
- [x] GiftBroker обновляет SharedContext историей диалога после каждого ответа

### 9.2 CrmAgent — единственный владелец CRM (шаг 2) ✅ (PR #115, 30.03.2026)

- [x] Создать `src/crm_agent.py` (обёртка над IntegramClient, ~70 строк)
- [x] `get_client_by_telegram_id`, `get_orders_for_user`, `add_health_fact`, `get_products`
- [x] Заменяет inline `async with IntegramClient()` в оркестраторе
- [ ] Перенаправить оставшиеся 10+ прямых вызовов через CrmAgent (постепенная миграция)

### 9.3 Gift TypedDict + GiftBroker (шаг 3) ✅ (PR #115, 30.03.2026)

- [x] Создать `src/gift_protocol.py` (Gift TypedDict + GiftBroker, ~100 строк)
- [x] Gift: giver, receiver, content, context, telos, anamnesis, freedom, timestamp
- [x] GiftBroker.send(): собирает анамнез → orchestrator.route() → обновляет SharedContext
- [x] user.py использует GiftBroker (fallback на прямой вызов если не инжектирован)
- [x] Логирование каждого Gift (DEBUG уровень)

### 9.4 WorkerAgent: inbox + DEFERRED ✅ (PR #117, 31.03.2026)

- [x] `WorkerStateManager`: `state: Literal["idle", "busy"]` + `inbox: asyncio.Queue`
- [x] `receive(gift) → ACCEPTED / DEFERRED` — если занят, кладёт в inbox
- [x] Чеклист перенесён из глобального dict в WorkerStateManager._checklists
- [x] `set_idle(deliver_fn)` — доставляет отложенные Gift
- [x] `UserContext.checklist` поле в SharedContext (для будущей CrmAgent-персистентности)

### 9.5 AGENT_SPECS в Integram ✅ (PR #117, 31.03.2026)

- [x] `AgentSpecsCache`: load/get/set + update_crm; graceful fallback на defaults
- [x] `crm_constants.py`: TABLE_AGENT_SPECS + REQ_AGENT_* (placeholder, TODO создать таблицу)
- [x] BeebotAgent читает system_prompt из AGENT_SPECS (через orchestrator)
- [x] `/agent_config <agent> <field> <value>` + `/agent_config reload`
- [ ] Создать таблицу AGENT_SPECS в Integram UI, заполнить ID в crm_constants.py

### ✅ Синхронизация после Фазы 9

```bash
# hive
git fetch upstream main && git reset --hard upstream/main
# VPS — сброс + пересборка (новые модули src/)
ssh ai-agent@185.233.200.13 "cd /home/ai-agent/BEEBOT \
  && git fetch origin main && git reset --hard origin/main \
  && docker compose up -d --build --force-recreate beebot"
# Проверить что бот поднялся и Gift Broker активен
ssh ai-agent@185.233.200.13 "docker logs --tail 20 beebot"
```

---

## Фаза 10: Память и персонализация (P2)

### 10.1 AnamnesisCache — эпизодическая память (A3) ✅ (PR #115, 30.03.2026)

- [x] `src/anamnesis.py`: агрегирует SQLite-факты + историю заказов из CrmAgent
- [x] `format_for_llm()` → строка «Прошлые заказы: #42 — Доставлен, 2800 ₽»
- [x] GiftBroker включает anamnesis[] в каждый Gift автоматически
- [x] orchestrator._node_beebot добавляет anamnesis_hint в memory_facts (10.3)
- [ ] Logist: если есть история → предзаполнять адрес, пропускать шаги

### 10.2 extract_fact — устранение ложных срабатываний ✅ (PR #115, 30.03.2026)

- [x] Добавить детектор отрицаний в `memory.py:extract_fact()`
- [x] Паттерн: `нет|не|без|никогда|не было|не страдаю|...` → skip предложение

### 10.3 Персонализация: «Вы уже брали» ✅ (PR #115, 30.03.2026)

- [x] orchestrator._node_beebot: AnamnesisCache → history hint → memory_facts консультанта
- [ ] «Хотите повторить заказ от 15 марта?» при ключевых словах (отдельная задача)
- [x] Logist: предзаполнять адрес из истории заказов (PR #118)

### 10.4 YouTube-комментарии в KB ✅ (31.03.2026)

- [x] Фильтровать ответы Александра из комментариев к видео
- [x] Добавить в FAISS как отдельный источник (высокий вес — прямая речь)
- [x] src/youtube_comments.py: Q&A пары через YouTube Data API v3
- [x] knowledge_base.py: comment: буст x1.2 при поиске
- [x] /yt_comments: скачать + пересобрать KB (17 тестов)

### ✅ Синхронизация после Фазы 10

```bash
# hive
git fetch upstream main && git reset --hard upstream/main
# VPS — сброс + пересборка + пересборка KB
ssh ai-agent@185.233.200.13 "cd /home/ai-agent/BEEBOT \
  && git fetch origin main && git reset --hard origin/main \
  && docker compose up -d --build --force-recreate beebot \
  && docker exec beebot python -m src.build_kb"
# Проверить количество чанков в логах
ssh ai-agent@185.233.200.13 "docker logs --tail 10 beebot | grep -i chunk"
```

---

## Фаза 11: Интерфейсы и экосистема (P2)

### 11.1 Событийные интерфейсы (из dronedoc2026) ✅ (31.03.2026)

> Идея: UI под роль + контекст + события. Не жёсткий `if WORKER_CHAT_IDS`.
> GiftBroker решает что показать какому пользователю — через Agent Card.

- [x] Каждый агент декларирует `triggers: list[str]` в AGENT_SPECS (PR #117)
- [x] GiftBroker.suggest_interface() — выбирает интерфейс по UserContext.interface_mode
- [x] Рабочий завершил все заказы → бот предлагает переключиться в режим клиента

### 11.2 AgentBus — регистрация BEEBOT ✅ (31.03.2026)

- [x] `POST /api/agent-bus/register` с Agent Card BEEBOT (tools: kb_search/order_status/ask)
- [x] Inbox polling каждые 10 сек + dispatch инструментов
- [x] Экспортировать инструменты: kb_search, order_status, ask
- [x] Heartbeat каждые 30 сек + автоперерегистрация
- [x] AGENT_BUS_URL в .env (graceful fallback если не задан)
- [x] 22 теста в test_agent_bus.py

### 11.3 PDF-отчёты и аналитика ✅ (PR #122, 31.03.2026)

- [x] Экспорт PDF: выручка за период, топ товаров, ABC-анализ клиентов (reportlab)
- [x] `GET /api/reports/sales?period=30d|90d|365d` — только admin
- [x] Алерт при низком остатке (<5 шт.) → уведомление пчеловоду (PR #119, дебаунс 24ч)

### ✅ Синхронизация после Фазы 11

```bash
# hive
git fetch upstream main && git reset --hard upstream/main
# VPS — сброс + пересборка обоих контейнеров (AgentBus меняет web тоже)
ssh ai-agent@185.233.200.13 "cd /home/ai-agent/BEEBOT \
  && git fetch origin main && git reset --hard origin/main \
  && docker compose up -d --build"
# Проверить регистрацию в AgentBus
curl -s http://185.233.200.13:8088/health | python3 -m json.tool
```

---

## Фаза 12: Инфраструктура и надёжность (P3)

> ⚠️ Апгрейд VPS снят с приоритетов: beebot = 193 MiB, RAM стабильна.
> hive SPOF: мониторинг + fallback важнее переноса.

### 12.1 Fallback при потере hive-туннеля ✅ (PR #122, 31.03.2026)

- [x] `src/tunnel_monitor.py`: TunnelMonitor проверяет порт 8990 каждые 60 сек
- [x] При разрыве: алерт пчеловоду в Telegram (только при смене состояния)
- [x] Fallback-ответ консультанта: топ-5 FAQ без LLM при `is_healthy=False`
- [x] Dev-mode: `ConnectionRefused` → `is_healthy=True` (не сигналит на dev-машине)
- [x] 12 тестов

### 12.2 Резервное копирование ✅ (31.03.2026)

- [x] `src/backup.py`: BackupManager (yadisk >= 2.0)
- [x] Ежедневно: `data/memory.db` → Яндекс Диск `/BEEBOT/daily/`
- [x] Еженедельно (воскресенье): CRM → CSV → `/BEEBOT/weekly/`
- [x] Хранить последние 30 ежедневных бэкапов (автоудаление)
- [x] `backup_now()` для ручного запуска из /admin
- [x] Нужен токен: `YADISK_TOKEN` в .env на VPS (graceful fallback если не задан)
- [x] 12 тестов

### 12.3 DEVBOT авторизация

**Проблема:** FastAPI на 8091 без авторизации (`allow_origins=["*"]`).

- [x] Добавить Bearer-токен в DEVBOT_API_URL
- [x] Валидация в FastAPI middleware devbot/bot.py (CORS ограничен localhost)

### 12.4 SSL + домен веб-панели

- [ ] Nginx reverse proxy с Let's Encrypt SSL
- [ ] Домен → https://
- [ ] JWT в httpOnly cookie (убрать XSS-риск)

### 12.5 Платёжная интеграция

- [ ] YooKassa или СБП — онлайн-оплата в Telegram
- [ ] Авто-смена статуса после подтверждения оплаты
- [ ] Webhook для уведомлений платёжного шлюза

### ✅ Синхронизация после Фазы 12

```bash
# hive
git fetch upstream main && git reset --hard upstream/main
# VPS — полная пересборка + nginx если 12.4 выполнена
ssh ai-agent@185.233.200.13 "cd /home/ai-agent/BEEBOT \
  && git fetch origin main && git reset --hard origin/main \
  && docker compose up -d --build"
# Проверить SSL если 12.4 выполнена
curl -I https://beebot.dmitrovykh.ru/health 2>/dev/null | head -5
# Проверить бэкапы если 12.2 выполнена
ssh ai-agent@185.233.200.13 "ls -lh /home/ai-agent/BEEBOT/backups/"
```

---

## Что снято с плана (было актуально — стало неактуально)

| Было | Причина снятия |
|------|---------------|
| Апгрейд VPS до 4 GB (10.5) | beebot = 193 MiB после PR#109, RAM в норме |
| Перенос Groq-прокси на VPS (10.1 вариант A) | RAM освободилась, нет смысла усложнять |
| Стойкость чеклиста → SQLite (9.4 старый) | Заменяется: чеклист → SharedContext → CrmAgent |
| Советы DEV_ADVICE в промпте (9.3 старый) | Поглощается: AGENT_SPECS в Integram (9.5 новый) |

---

## Сводная таблица приоритетов

| П | Задача | Фаза | Сложность | Синхронизация |
|---|--------|------|-----------|---------------|
| **P0** | Синхронизация hive + VPS с main | — | Мин | hive reset + VPS reset |
| **P1** | Разбить bot.py на Router-модули | 8.1 | Средняя | PR → squash → VPS rebuild |
| **P1** | Убрать дублирование AdminChatAgent | 8.2 | Низкая | PR → squash → VPS rebuild |
| **P1** | Inspect в оркестраторе | 8.3 | Низкая | PR → squash → VPS rebuild |
| **P1** | UDS SKU-сопоставление | 8.4 | Средняя | PR → squash → VPS rebuild |
| **P1** | SharedContext (рабочая память) | 9.1 | Низкая | PR → squash → VPS rebuild |
| **P1** | CrmAgent (единый владелец) | 9.2 | Средняя | PR → squash → VPS rebuild |
| **P1** | Gift TypedDict + GiftBroker | 9.3 | Средняя | PR → squash → VPS rebuild |
| **P2** | WorkerAgent inbox + DEFERRED | 9.4 | Средняя | PR → squash → VPS rebuild |
| **P2** | AGENT_SPECS в Integram | 9.5 | Средняя | PR → squash → без rebuild |
| **P2** | AnamnesisCache (эпизодическая память) | 10.1 | Средняя | PR → squash → VPS rebuild |
| **P2** | extract_fact отрицания | 10.2 | Низкая | PR → squash → VPS rebuild |
| **P2** | Персонализация «Вы уже брали» | 10.3 | Средняя | PR → squash → VPS rebuild |
| **P2** | YouTube-комментарии в KB | 10.4 | Средняя | PR → squash → VPS rebuild |
| **P2** | Событийные интерфейсы | 11.1 | Высокая | PR → squash → VPS rebuild |
| **P2** | AgentBus регистрация | 11.2 | Средняя | PR → squash → без rebuild |
| **P2** | PDF-отчёты | 11.3 | Средняя | PR → squash → VPS rebuild |
| **P3** | Fallback при потере hive-туннеля | 12.1 | Средняя | PR → squash → VPS rebuild |
| **P3** | Резервное копирование | 12.2 | Низкая | systemd + cron на VPS |
| **P3** | DEVBOT авторизация | 12.3 | Низкая | PR → squash → hive restart |
| **P3** | SSL + домен | 12.4 | Средняя | VPS nginx config |
| **P3** | Платёжная интеграция | 12.5 | Высокая | PR → squash → VPS rebuild |

---

## Принципы (из dronedoc2026, применимые к BEEBOT)

1. **ИИ — инструмент, не судья.** Ни один агент не принимает решений без пчеловода. Все выводы — рекомендации.
2. **Минимально необходимый доступ.** CrmAgent — единственный кто видит CRM. Остальные — через Gift.
3. **Прозрачность каждого вывода.** Beebot даёт ссылку на источник знания (чанк + видео/документ).
4. **Агент не выходит за роль.** Консультант не создаёт заказы. Логист не консультирует.
5. **Свобода агента — условие дара (A5).** DEFERRED — не ошибка, а корректный ответ.

---

*Связанные документы: [analysis.md](analysis.md) · [docs/architecture.md](docs/architecture.md)*
