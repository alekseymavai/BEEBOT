# BEEBOT — План развития

> **Дата:** 4 апреля 2026 (обновлено по итогам CRM v2 серии, #183)
> **Основан на:** [analysis.md](analysis.md)

---

## Принципы

1. **Test-first** — сначала тесты, потом код.
2. **Один PR = одна задача** — атомарные коммиты, squash-мерж.
3. **Feature flags** — `INTEGRAM_V2=true/false` без деплоя образов.
4. **Не ломать production** — каждый мерж проверяется на VPS.
5. **Документация = код** — обновлять CLAUDE.md/analysis.md/plan.md при структурных изменениях.

---

## Завершённые фазы (1–6)

### Фаза 6: Микроядерная архитектура ✅ DONE (6 апреля 2026)

bot.py монолит (1899 строк) → BeeBotApp + 11 автономных плагинов.

**Ядро (`src/kernel/`):**
- `plugin.py` — абстрактный `Plugin`: `setup / register_routers / register_api / get_bg_tasks / teardown`
- `container.py` — DI-контейнер: `set / get / require / has`
- `app.py` — `BeeBotApp`: топологическая сортировка зависимостей, lifecycle

**Плагины (`src/plugins/`):**

| Плагин | Зависимости | Публикует |
|--------|-------------|-----------|
| `crm` | — | crm, auth |
| `knowledge` | crm | kb |
| `agents` | crm, knowledge | orchestrator, inspector, logist, analyst, admin_chat_agent, consult_service |
| `orders` | crm, agents | order_service, notification_service |
| `analytics` | crm, agents | analytics_service, dashboard_service |
| `delivery` | — | delivery_service |
| `workers` | crm | worker_service |
| `gift` | crm, agents | crm_agent, anamnesis, gift_broker, agent_specs |
| `monitoring` | crm | bg_manager, crm_snapshot + BgTask-и |
| `telegram` | crm, agents, orders, gift, workers | (роутеры aiogram) |
| `web` | crm, orders, analytics, delivery, workers | fastapi_app |

**bot.py** → 95 строк (vs 1899 ранее). Добавление модуля = `Plugin`-подкласс + одна строка `app.register(...)`.

---

## Завершённые фазы (1–5.0)

### Фаза 1: CRM v2 + стабилизация ✅ DONE
- A.1 Тесты IntegramV2Client (27 тестов)
- A.2 Фабрика CRM-клиента (crm_factory.py)
- A.3 Совместимость интерфейсов v1/v2
- A.4 Товары v2 (85 шт.)
- A.5 Колонка Источник
- A.6 Тесты роутеров с v2
- D.1 CORS из env
- D.2 Убрать fallback-авторизацию

### Фаза 2: Рефакторинг ✅ DONE
- C.1 Мёртвые узлы оркестратора удалены
- C.2 Унификация инъекции CRM (set_crm() везде)
- C.3 Утилиты парсинга дат (utils.py)
- C.4 Русские месяцы (RU_MONTHS)
- C.5 Централизация промптов (prompts.py)
- C.6 Убрать доступ к private crm._api
- C.7 N+1 в AnalystAgent — fixed

### Фаза 3: Service Layer ✅ DONE
- B.1 OrderService → logist.py
- B.2 OrderService → orders.py
- B.3 OrderService → uds.py
- B.4 NotificationService в bot.py
- B.5 EventBus в lifespan
- B.6 Удалить inline-уведомления

### Фаза 4: CI/CD и качество ✅ DONE
- D.3 mypy в CI
- D.4 bandit в CI
- D.5 ruff+I (import order)
- D.6 git reset --hard origin/main в CI

### Фаза 5.0: UX веб-панели ✅ DONE (#157)
- П1 Цветные полосы строк по статусу
- П2 Поиск: номер/клиент/трек (debounce)
- П3 Фильтр дат (DatePicker range)
- П4 Переключатель периода дашборда
- П5 «Требуют внимания» + /api/dashboard/alerts
- П6 Топ-5 товаров в дашборде
- П7 Инлайн-смена статуса в строке
- П8 Групповые действия (batch-status)
- П9 Бейджи-счётчики в навигации (polling 2 мин)
- П10 Канбан-вид с drag & drop

### CRM v2 серия фиксов ✅ DONE (4 апреля 2026, 13 PR)
- #183 Dashboard stats — правильные поля
- #182 Singleflight + кэш get_orders()
- #180-#166 Пагинация, позиции, схема, импорты

---

## Немедленные приоритеты (P0)

### Fix-1: Восстановить локальный dev-окружение

**Проблема:** 9 файлов сервисов удалены локально (`git status: D`), startup.py их импортирует → ImportError.

**Вариант A (рекомендуемый):** восстановить файлы
```bash
git checkout HEAD -- src/services/
```

**Вариант B (альтернатива):** обновить startup.py, убрав импорты и создание удалённых сервисов. Нужно проверить тесты.

| Задача | Критерий | Файлы |
|--------|---------|-------|
| Восстановить сервисы ИЛИ убрать импорты из startup.py | `python -m src.bot --help` без ошибок | startup.py, services/ |
| Удалить src/bot.py.rej | Файл не существует | src/ |
| Исправить тесты для удалённых сервисов | `pytest` без ImportError | tests/ |

### Fix-2: A.7 — Production switch INTEGRAM_V2=true

**Последний шаг миграции CRM v2.** Все 13 фиксов слиты в main.

```bash
# 1. Обновить .env на VPS
ssh ai-agent@185.233.200.13 "echo 'INTEGRAM_V2=true' >> /home/ai-agent/BEEBOT/.env"

# 2. Пересобрать контейнер
ssh ai-agent@185.233.200.13 "cd /home/ai-agent/BEEBOT && docker compose up -d --build --force-recreate beebot"

# 3. Проверить
ssh ai-agent@185.233.200.13 "docker logs --tail 30 beebot"
```

**Критерий готовности:** веб-панель показывает заказы из ai2o.online, бот отвечает на вопросы.

---

## Направление 1: Память агентов (Приоритет P1)

> **Цель:** персонализация — агенты помнят пользователя между сессиями.

| # | Задача | Что делать | Файлы |
|---|--------|-----------|-------|
| M.1 | LangGraph Checkpointer | Добавить SqliteSaver в StateGraph — эпизодическая память бесплатно | orchestrator.py | ✅
| M.2 | agent_id namespace | Добавить поле agent_id в user_memory. Каждый агент видит только свои факты | memory.py | ✅
| M.3 | Таблица episodes | CREATE TABLE episodes(user_id, agent_id, event_type, summary, detail, created_at) | memory.py | ✅
| M.4 | MemoryService | Единый API для всех агентов. Агент не знает о хранилище | memory_service.py | ✅
| M.5 | AgentContext | Передавать unified context в оркестратор: profile+episodes+crm_snapshot | orchestrator.py | ✅
| M.6 | Sync → Integram | Ежесуточный бэкап episodes и user_facts в Integram | memory_service.py |

**Источники:** docs/memory_architecture.md (детальный анализ best practices 2025)

---

## Направление 2: Аналитика и отчёты (Приоритет P1)

> **Цель:** восстановить AnalyticsService как полноценный модуль.

| # | Задача | Что делать | Файлы |
|---|--------|-----------|-------|
| AN.1 | Восстановить AnalyticsService | Перенести логику из orchestrator._node_stats в отдельный сервис | analytics_service.py |
| AN.2 | ABC-анализ | Классификация товаров A/B/C по выручке за период | analytics_service.py |
| AN.3 | Сезонный прогноз | Прогноз спроса на +30/60/90 дней по истории продаж | analytics_service.py |
| AN.4 | PDF-отчёты | Генерация отчётов за период (уже есть pdf_report.py) | web/routers/report.py |
| AN.5 | Экспорт аналитики | CSV/Excel с фильтрами по периоду и товарам | web/routers/export.py |
| AN.6 | График прогноза | Визуализация прогноза в веб-панели | web/views/DashboardView.vue |

---

## Направление 3: Пасечный дневник (Приоритет P2)

> **Цель:** пчеловод ведёт записи об осмотрах улья через бот.

| # | Задача | Что делать | Файлы |
|---|--------|-----------|-------|
| D.1 | Таблица «Осмотры» в CRM v2 | Дата, улей, наблюдения, рекомендации | integram_v2_constants.py |
| D.2 | /diary команда | Текстовый ввод → CRM. FSM: дата/улей/наблюдение | routers/diary.py |
| D.3 | Фото-ввод | Пользователь → фото → vision API → описание → CRM | routers/diary.py |
| D.4 | Голосовой ввод | Whisper API → текст → CRM | routers/diary.py |
| D.5 | /diary_history | Просмотр записей по дате/улью | routers/diary.py |
| D.6 | Страница в веб-панели | DiaryView.vue — список осмотров + фильтр | web/views/ |

---

## Направление 4: Offline PWA (Приоритет P2)

> **Цель:** работник на пасеке без интернета.

| # | Задача | Что делать | Файлы |
|---|--------|-----------|-------|
| O.1 | Активировать offline.js | IndexedDB кэш уже написан | web/stores/offline.js |
| O.2 | Service Worker | Перехват запросов, page caching | sw.js, vite.config.js |
| O.3 | Sync queue | При восстановлении сети — отправить накопленные изменения | web/stores/offline.js |
| O.4 | Offline-индикатор | Показывать в AppLayout что работаем offline | AppLayout.vue |
| O.5 | Тесты offline | PWA tests в Playwright | tests/e2e/ |

---

## Направление 5: Доставка в веб-панели (Приоритет P2)

> **Цель:** расчёт стоимости доставки при создании заказа через веб.

| # | Задача | Что делать | Файлы |
|---|--------|-----------|-------|
| E.1 | POST /api/delivery/calculate | Эндпоинт калькулятора (уже есть delivery/) | web/routers/delivery.py |
| E.2 | Кэш тарифов | TTL 1 час, не дёргать СДЭК/Почту каждый раз | delivery/calculator.py |
| E.3 | Интеграция в NewOrderView | При выборе адреса → автоматический расчёт | web/views/NewOrderView.vue |
| E.4 | Выбор тарифа | Пользователь видит варианты СДЭК/Почта/самовывоз | web/views/NewOrderView.vue |

---

## Направление 6: Мониторинг (Приоритет P3)

> **Цель:** видеть здоровье системы в реальном времени.

| # | Задача | Что делать | Файлы |
|---|--------|-----------|-------|
| Mon.1 | Prometheus метрики | request_count, latency_p95, error_rate, crm_requests | web/api.py |
| Mon.2 | /metrics эндпоинт | Стандартный prometheus scrape endpoint | web/routers/metrics.py |
| Mon.3 | RAM алерт | Если RSS > 1.5 GB → Telegram алерт пчеловоду | tunnel_monitor.py |
| Mon.4 | CRM health check | /api/health — статус CRM, EventBus, BackgroundTasks | web/routers/health.py |
| Mon.5 | Grafana дашборд | Визуализация здоровья на отдельном порту | docker-compose.yml |

---

## Направление 7: База знаний (Приоритет P3)

> **Цель:** расширять и улучшать KB без пересборки FAISS вручную.

| # | Задача | Что делать | Файлы |
|---|--------|-----------|-------|
| KB.1 | Страница KB в веб-панели | Список источников, поиск, добавление документов | web/views/KBView.vue |
| KB.2 | GET /api/kb/search | Поиск по базе знаний через API | web/routers/kb.py |
| KB.3 | Авто-пересборка | При добавлении нового документа → rebuild FAISS | build_kb.py |
| KB.4 | Метаданные источников | Автор, дата, тип (pdf/text/youtube) в chunks.json | knowledge_base.py |
| KB.5 | YouTube auto-update | Ежедневное обновление субтитров при разблокировке | youtube_updater.py |

---

## Направление 8: AgentBus (Приоритет P3)

> **Цель:** BEEBOT — участник экосистемы агентов.
> **Основа:** docs/memory_architecture.md §5, reference_dronedoc_agentbus.md

| # | Задача | Что делать | Файлы |
|---|--------|-----------|-------|
| AB.1 | Внутренний AgentBus | in-process: GiftBroker → агенты через SharedContext | gift_protocol.py |
| AB.2 | Экспорт инструментов | kb_search(q), order_status(id), product_info(name) | agent_specs.py |
| AB.3 | AgentContext унификация | Единый контекст: profile + episodes + crm_snapshot | shared_context.py |
| AB.4 | Тесты AgentBus | Покрытие gift_protocol, shared_context, agent_specs | tests/ |

---

## Направление 9: Инфраструктура (Приоритет P3)

| # | Задача | Что делать |
|---|--------|-----------|
| I.1 | Апгрейд VPS до 4 GB | При росте beebot > 1.5 GB RSS |
| I.2 | Резервный Groq-прокси | Fallback на облачный сервер при падении hive |
| I.3 | Redis persistence | AOF для сохранения данных при рестарте Redis |
| I.4 | Бекап .env | Зашифрованный бекап на Яндекс.Диск |
| I.5 | DEVBOT авторизация | JWT для DEVBOT FastAPI (сейчас allow_origins=["*"]) |

---

## Сводная таблица приоритетов

```
P0 (немедленно):
  Fix-1: Восстановить dev-окружение (startup.py/services)
  Fix-2: A.7 Production switch INTEGRAM_V2=true

P1 (эта неделя):
  M.1-M.3: LangGraph Checkpointer + agent_id namespace + episodes
  AN.1-AN.2: Восстановить AnalyticsService + ABC

P2 (следующий спринт):
  D.1-D.5: Пасечный дневник
  O.1-O.3: Offline PWA
  E.1-E.4: Доставка в веб-панели

P3 (бэклог):
  Mon.1-Mon.5: Мониторинг
  KB.1-KB.5: База знаний в веб-панели
  AB.1-AB.4: AgentBus
  I.1-I.5: Инфраструктура
```

---

## Метрики успеха

| Метрика | Сейчас | Цель |
|---------|--------|------|
| CRM в production | v1 (ai2o.ru) | v2 (ai2o.online) — A.7 |
| Сервисы работают локально | ❌ ImportError | ✅ Fix-1 |
| Тесты проходят | ~60% (остальные → ImportError) | 100% зелёные |
| Агенты с памятью | 1/6 (Консультант) | 6/6 через MemoryService |
| KB поиск в веб | ❌ | ✅ E.4 |
| Offline PWA активна | ❌ (написана, не включена) | ✅ O.1-O.3 |

---

*Связанные документы: [analysis.md](analysis.md) | [docs/architecture.md](docs/architecture.md) | [README.md](README.md)*
