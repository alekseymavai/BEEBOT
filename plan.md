# BEEBOT — План развития

> **Дата:** 2 апреля 2026
> **Основан на:** [analysis.md](analysis.md)

---

## Направление A: Активация CRM v2

**Цель:** Переключить весь проект на ai2o.online (чистые данные, новый API).

| # | Задача | Приоритет | Файлы |
|---|--------|-----------|-------|
| A.1 | Написать тесты для IntegramV2Client | P0 | tests/test_integram_v2_client.py |
| A.2 | Фабрика CRM-клиента: if INTEGRAM_V2 → v2 else v1 | P0 | src/web/deps.py, src/bot.py |
| A.3 | Проверить совместимость интерфейсов v1/v2 | P0 | src/integram_v2_client.py |
| A.4 | Догрузить 78 товаров в основную таблицу (581) | P1 | scripts/migrate_to_v2.md |
| A.5 | Добавить колонку Источник к Клиентам и Заказам | P1 | Integram UI |
| A.6 | Протестировать все роутеры веб-панели с v2 | P1 | tests/ |
| A.7 | Переключить production: INTEGRAM_V2=true | P1 | .env на VPS |

---

## Направление B: Подключение Service Layer

**Цель:** Единая точка создания заказов, устранение дублирования.

| # | Задача | Приоритет | Файлы |
|---|--------|-----------|-------|
| B.1 | Подключить OrderService к logist.py | P1 | src/agents/logist.py |
| B.2 | Подключить OrderService к web/routers/orders.py | P1 | src/web/routers/orders.py |
| B.3 | Подключить OrderService к uds.py | P1 | src/integrations/uds.py |
| B.4 | Активировать NotificationService | P1 | src/services/notification_service.py |
| B.5 | Подключить EventBus (Redis Streams) | P2 | src/bus.py, src/web/bus_handlers.py |
| B.6 | Удалить дублирование логики уведомлений | P2 | logist, orders.py, uds.py |

---

## Направление C: Очистка и рефакторинг

**Цель:** Удалить мёртвый код, унифицировать паттерны.

| # | Задача | Приоритет | Файлы |
|---|--------|-----------|-------|
| C.1 | Удалить мёртвые узлы из оркестратора | P1 | src/orchestrator.py |
| C.2 | Унифицировать инъекцию CRM (set_crm() везде) | P1 | src/agents/*.py, src/bot.py |
| C.3 | Создать утилиту парсинга дат | P2 | src/utils.py |
| C.4 | Централизовать русские названия месяцев | P2 | src/utils.py |
| C.5 | Централизовать LLM-промпты | P2 | src/prompts.py |
| C.6 | Убрать доступ к private crm._api | P2 | src/web/routers/batches.py |
| C.7 | Исправить N+1 в AnalystAgent | P1 | src/agents/analyst.py |

---

## Направление D: Безопасность и CI/CD

**Цель:** Закрыть уязвимости, усилить проверки.

| # | Задача | Приоритет | Файлы |
|---|--------|-----------|-------|
| D.1 | CORS origins из .env | P1 | src/web/api.py |
| D.2 | Убрать fallback-авторизацию | P1 | src/web/routers/auth.py |
| D.3 | Добавить mypy в CI | P2 | .github/workflows/ci.yml |
| D.4 | Добавить bandit в CI | P2 | .github/workflows/ci.yml |
| D.5 | Расширить ruff-правила | P2 | pyproject.toml |
| D.6 | Deploy: git reset вместо git merge | P2 | .github/workflows/ci.yml |

---

## Направление E: Новые возможности

**Цель:** Развитие функциональности после стабилизации.

| # | Задача | Приоритет | Описание |
|---|--------|-----------|----------|
| E.1 | Пасечный дневник (HiveJournal) | P2 | 3 режима: текст/фото/голос → хранение в CRM v2 |
| E.2 | Offline mode во фронтенде | P2 | Активировать offline.js, IndexedDB, sync queue |
| E.3 | Расчёт доставки в веб-панели | P2 | Эндпоинты для СДЭК/Почта через delivery/ |
| E.4 | KB-поиск в веб-панели | P3 | Интеграция knowledge_base в FastAPI |
| E.5 | Бекап на Яндекс.Диск | P3 | Автоматический бекап CRM + KB |
| E.6 | Мониторинг (Grafana) | P3 | Дашборд: API latency, ошибки, память |
| E.7 | Мульти-агентная экосистема (AgentBus) | P3 | BEEBOT как узел в шине агентов |

---

## Порядок выполнения

```
Неделя 1:  A.1 → A.2 → A.3 → A.6 → A.7  (CRM v2 в production)
Неделя 2:  C.1 → C.2 → C.7 → D.1 → D.2  (очистка + безопасность)
Неделя 3:  B.1 → B.2 → B.3 → B.4         (Service Layer)
Неделя 4:  C.3–C.6 → D.3–D.6             (рефакторинг + CI)
Далее:     E.1–E.7                         (новые возможности)
```

---

*Связанные документы: [analysis.md](analysis.md) | [docs/architecture.md](docs/architecture.md) | [README.md](README.md)*
