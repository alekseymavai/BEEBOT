# BEEBOT — Анализ текущего состояния

> **Дата:** 2 апреля 2026 (обновлено по итогам фаз 1-4)
> **Версия:** v2.1
> **Коммит:** 5c63419 (#146)

---

## 1. Обзор проекта

BEEBOT — Telegram-помощник + веб-панель управления заказами для «Усадьба Дмитровых».
Многоагентная система на LangGraph с 6 агентами, FAISS-поиском по базе знаний,
CRM-интеграцией (Integram) и PWA веб-панелью.

### Масштаб кодовой базы

| Компонент | Файлов | Строк |
|-----------|--------|-------|
| Бот + агенты | 8 | 2 853 |
| CRM-клиенты (v1 + v2) | 4 | 2 281 |
| Веб-панель (backend) | 10 | 1 480 |
| Service Layer (не подключён) | 4 | 870 |
| Доставка | 5 | 875 |
| KB + LLM + память | 5 | 850 |
| Frontend (Vue 3) | 14 views | ~3 000 |
| Тесты | 32 | 8 028 |
| **Итого** | ~80 | **~20 000** |

---

## 2. Сильные стороны

### Архитектура
- **Многоагентная система** — 6 специализированных агентов с чётким разделением ответственности
- **LangGraph-оркестратор** — классификация intent + маршрутизация (consult / order / stats / greeting / edit / track)
- **Graceful degradation** — fallback на статичные данные при недоступности CRM, LLM, туннелей
- **Гибридный KB-поиск** — 70% семантика (FAISS) + 30% стилометрия + keyword-буст из каталога
- **5 голосовых стилей** — «Голос Улья» (наставник, практик, селекционер, зимовщик, эколог)

### Реализация
- **Типизация** — Pydantic-модели (Product, Client, Order, OrderItem)
- **Async/await** — полностью асинхронный бот и веб-сервер
- **Кеширование** — CrmSnapshot (TTL 5 мин), кеш заказов в веб-панели (90 сек)
- **Docker** — multi-stage build, оптимизация для 2 GB VPS
- **Тесты** — 32 файла, 8 028 строк
- **CI/CD** — GitHub Actions (ruff + pytest + deploy)

### Функциональность
- **14 страниц веб-панели** (PWA, offline-сборка, SSE-уведомления)
- **DEVBOT** — автономный разработчик через /dev (Claude API + CLI)
- **Авто-трекинг** — СДЭК + Почта России каждые 2 часа
- **UDS-синхронизация** — поллинг каждые 5 мин

---

## 3. Критические проблемы (P0)

### 3.1 ~~IntegramV2Client написан, но не подключён~~ ЧАСТИЧНО РЕШЕНО

`src/crm_factory.py` создан — фабрика `get_crm_client()` возвращает v1 или v2 по флагу `INTEGRAM_V2`. 13 недостающих методов добавлены в v2 клиент (совместимость с v1). 27 тестов написаны. **Осталось:** подключить фабрику в `web/deps.py` и `bot.py`, переключить production на `INTEGRAM_V2=true`.

### 3.2 Service Layer не активирован

`OrderService` (301 строка), `NotificationService` (140), `EventBus` (244), `BotServiceClient` (181) написаны в рамках Hexagonal Architecture (steps 0-5), но **не вызываются** из production-кода. Создание заказа дублируется в 3 местах с разной логикой уведомлений.

### 3.3 ~~Нет тестов для v2 клиента~~ РЕШЕНО

27 unit-тестов написаны в `tests/test_integram_v2_client.py` (auth, CRUD, helpers). Все зелёные.

---

## 4. Серьёзные проблемы (P1)

### 4.1 Несогласованная инъекция зависимостей

```python
logist._crm = client          # прямое присвоение
analyst._crm = client         # прямое присвоение
admin_chat.set_crm(client)    # через метод
orchestrator._groq             # свой экземпляр LLM
```

### 4.2 ~~Мёртвый код в оркестраторе~~ РЕШЕНО

Удалены: `_logist`, `_node_logist()`, `_node_passthrough()`. Интенты order/edit/track/inspect → END (обрабатываются в bot.py роутерами).

### 4.3 N+1 запросы в аналитике

`analyst.py` строка 586: fallback `get_order_items(order.id)` для каждого заказа. На 1915 заказах — 1915+ HTTP-запросов.

### 4.4 AdminChatAgent — двойная загрузка CRM

Два пути получения данных: live-запросы (23 HTTP на сообщение) и CrmSnapshot (кеш 5 мин). Оба активны одновременно.

---

## 5. Конфликты логик

### 5.1 Три пути создания заказа — без единого центра

| Путь | Файл | Уведомления | OrderService? |
|------|------|-------------|---------------|
| Telegram FSM | logist.py | Пчеловод + работники | Нет |
| Веб-панель | orders.py | Только пчеловод | Нет |
| UDS-поллер | uds.py | Никого | Нет |

### 5.2 Orchestrator vs Router — рассыпанная логика

| Intent | Orchestrator обрабатывает? | Где реально? |
|--------|---------------------------|-------------|
| consult | Да (BeebotAgent) | Orchestrator |
| stats | Да (AnalystAgent) | Orchestrator |
| greeting | Да | Orchestrator |
| order | Нет → END | fsm_order_router |
| edit | Нет → END | bot.py |
| track | Нет → END | bot.py |
| inspect | Нет → END | inspect_router |

Из 7 интентов 3 обрабатываются оркестратором, 4 → END (роутеры bot.py). Мёртвые узлы удалены в #146.

### 5.3 CRM v1 vs v2 — два мира

| | v1 (ai2o.ru) | v2 (ai2o.online) |
|---|---|---|
| Статус | **Используется везде** | **Написан, не подключён** |
| Auth | Cookie-based | JWT |
| Поля | По REQ_ID | По имени колонки |
| Данные | 1924 клиента, 1915 заказов (грязные) | 85 товаров (чистые) |

### 5.4 BeebotAgent vs InspectorAgent — перекрытие

Оба используют одну KB и LLM. Разница только в формате (одноразовый ответ vs 3-шаговый диалог). Пользователь может задать диагностический вопрос в обычном чате и получить ответ от BeebotAgent.

---

## 6. Технический долг

### Мёртвый код (~1 000 строк, ~13% базы)

| Компонент | Строки | Причина | Статус |
|-----------|--------|---------|--------|
| services/order_service.py | 301 | Написан, не подключён | Ждёт фазу 3 |
| bus.py + bot_client.py | 425 | EventBus не используется | Ждёт фазу 3 |
| web/bus_handlers.py | ~200 | Все методы → ошибка | Ждёт фазу 3 |
| ~~orchestrator (logist, passthrough)~~ | ~~65~~ | ~~Пустые узлы~~ | ✅ Удалено в #146 |
| delivery/base.py | 41 | Заглушка | |

### Дублирование

| Что | Где | Сколько мест | Статус |
|-----|-----|-------------|--------|
| Парсинг DD.MM.YYYY | analyst, admin_chat, integram_api | 6 | `src/utils.py` создан (#146) |
| Русские месяцы | analyst, admin_chat | 2 | `RU_MONTHS` в utils.py (#146) |
| Создание заказа | logist, orders.py, uds.py | 3 | Ждёт фазу 3 |
| LLM-промпты | orchestrator, analyst, inspector, admin_chat | 4 | Ждёт C.5 |

---

## 7. Безопасность

| Проблема | Файл | Критичность |
|----------|------|-------------|
| ~~Hardcoded CORS origins~~ | ~~web/api.py~~ | ✅ Из env (WEB_CORS_ORIGINS) |
| ~~Fallback-авторизация через env~~ | ~~auth.py~~ | ✅ Убрана в #146 |
| Доступ к private `crm._api` | batches.py:40,110 | Средняя |
| SSE-токен без проверки срока | api.py:149 | Низкая |
| Слабый rate limiting | 60 req/min на всё | Средняя |

---

## 8. Инфраструктурные ограничения

| Ограничение | Решение | Статус |
|-------------|---------|--------|
| VPS 2 GB RAM (beebot ~762 MiB) | Апгрейд до 4 GB | Запланировано |
| Groq/TG блокируют IP VPS | SSH-туннели через hive | Работает |
| Hive — SPOF | Облачный fallback | Не начато |
| API v2 rate limit (~8 write/таблицу) | Temp-table workaround | Обходится |

---

## 9. Рейтинг компонентов

| Компонент | Качество | Тесты | Проблемы |
|-----------|---------|-------|----------|
| Knowledge Base | A | B | — |
| LLM Client | A | B | — |
| BeebotAgent | A | B | — |
| LogistAgent | B | A | Fallback каталог |
| AnalystAgent | B | A | N+1 запросы |
| InspectorAgent | B | B | Overlap с Beebot |
| AdminChatAgent | C | B | Двойная загрузка CRM |
| WorkerAgent | B | B | State в RAM |
| IntegramClient (v1) | B | A | Устаревает |
| IntegramV2Client | B | **B** | 27 тестов (#146) |
| OrderService | — | — | Не подключён |
| Веб-панель | B | A | Нет offline mode |
| Docker + CI/CD | B | — | Нет mypy/bandit |

---

*Связанные документы: [plan.md](plan.md) | [docs/architecture.md](docs/architecture.md) | [README.md](README.md)*
