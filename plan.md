# BEEBOT — План развития

> **Дата:** 2 апреля 2026
> **Основан на:** [analysis.md](analysis.md) · [docs/architecture.md](docs/architecture.md)

---

## Завершённые фазы ✅

| Фаза | Содержание | Дата |
|------|-----------|------|
| 0–2 | Стабилизация, качество, функциональность | ✅ |
| 3 | LLM-ассистент, память, онтология, пагинация | ✅ 25.03 |
| 4 | История статусов, партии, DEVBOT | ✅ 27.03 |
| 5 | WorkerAgent, CrmSnapshot | ✅ 29.03 |
| 6 | UDS catch-up, кэш заказов | ✅ 29.03 |
| 7 | DEVBOT MVP | ✅ 29.03 |
| 8 | Разбить bot.py, UDS SKU | ✅ 31.03 |
| 9 | Gift Protocol, SharedContext, CrmAgent | ✅ 31.03 |
| 10 | AnamnesisCache, YouTube Q&A | ✅ 31.03 |
| 11 | AgentBus, PDF-отчёты, Worker→client | ✅ 31.03 |
| 12 | TunnelMonitor, BackupManager, SSL-конфиг | ✅ 31.03 |
| Steps 0–5 | Redis + EventBus + Service Layer + Hexagonal | ✅ 02.04 |
| Data | UDS-SHOP: 1358 заказов обновлены (статус+дата+сумма+состав) | ✅ 01.04 |
| Data | SKU маппинг: 75 товаров + 9 новых | ✅ 01.04 |

---

## Направление A: Чистота кода (P0)

> Убрать мёртвый код, закоммитить незакоммиченные изменения, привести в порядок.

### A.1 Удалить мёртвые копии
- [ ] Удалить src/infrastructure/ (18 файлов-копий)
- [ ] Удалить src/transport/ (6 файлов-копий)
- [ ] Удалить src/domain/ (4 файла-копии)
- [ ] Оставить src/services/ (содержит новый код: OrderService, NotificationService)
- [ ] Оставить src/bus.py (EventBus — новый код)
- [ ] Оставить src/bot_client.py (BotServiceClient — новый код)
- [ ] Оставить src/web/bus_handlers.py, bg_tasks.py (новый код)

### A.2 Закоммитить рабочие изменения
- [ ] src/integram_api.py — пропуск битых страниц
- [ ] src/integram_client.py — парсинг даты MM/DD/YYYY + время
- [ ] src/backup.py — app_folder пути (PR #127)

### A.3 Обновить документацию
- [ ] analysis.md — текущее состояние
- [ ] plan.md — актуальный план
- [ ] README.md — описание проекта
- [ ] CLAUDE.md — инструкция для AI
- [ ] Удалить устаревшие: docs/architecture/BEEBOT_ARCHITECTURE.md, devbot_plan.md

---

## Направление B: Единый Service Layer (P1)

> Подключить OrderService и NotificationService к реальному коду.

### B.1 OrderService — единый создатель заказов
- [ ] logist.py: заменить `self._crm.create_order()` на `order_service.create_order_with_client()`
- [ ] web/routers/orders.py: заменить прямой CRM на `order_service.create_order()`
- [ ] uds.py: заменить `integram_client.create_order()` на `order_service.create_order()`
- [ ] Тесты: убедиться что все 3 пути проходят через один сервис

### B.2 NotificationService — единые уведомления
- [ ] Объединить src/notifications.py + src/web/notifications.py в services/notification_service.py
- [ ] Один вход для отправки: клиент / пчеловод / работники / группы
- [ ] Удалить дублирование

### B.3 Статусы через OrderService
- [ ] admin.py: заменить `crm.update_order_status()` на `order_service.update_status()`
- [ ] tracker.py: аналогично
- [ ] worker.py: аналогично

---

## Направление C: Данные (P1)

> Привести CRM в порядок — импорт из тетрадки, привязка клиентов.

### C.1 Импорт заказов из тетрадки пчеловода
- [ ] Парсинг xlsx (9 листов, ~370 заказов)
- [ ] Маппинг товаров по названию (состав: «Прополис 50мл 4шт × 600р»)
- [ ] Создание клиентов по телефону
- [ ] Проставление статусов и дат

### C.2 Привязка клиентов UDS
- [ ] Дообогатить телефоны из UDS Admin API
- [ ] Дедупликация: один человек через UDS + Telegram = один клиент

### C.3 Починка Integram
- [ ] Дождаться восстановления 6 битых страниц
- [ ] Или: определить и пересоздать повреждённые записи

---

## Направление D: Бот → тонкий клиент (P2)

> Реализовать архитектуру из steps 0–5: бот общается с Backend через Redis.

### D.1 Подключить EventBus в bot.py
- [ ] При старте: подключиться к Redis
- [ ] Консультации: bot → Redis → ConsultService → Redis → bot
- [ ] Заказы: bot → Redis → OrderService → Redis → bot
- [ ] Feature flag для постепенного переключения

### D.2 ConsultService
- [ ] Создать src/services/consult_service.py
- [ ] answer(query, user_id, history, style) — KB + LLM
- [ ] inspect(questions, answers) — осмотр улья

### D.3 Worker чеклист → SQLite
- [ ] Перенести _checklists из RAM в SQLite (или SharedContext)
- [ ] Переживает рестарт бота

### D.4 Лёгкий Docker-образ для бота
- [ ] Dockerfile.bot: только aiogram + redis-py (~50 MB)
- [ ] Без FAISS, Groq, CRM — всё в Backend

---

## Направление E: Монетизация и масштаб (P3)

### E.1 SSL + домен
- [ ] Получить домен (напр. beebot.dmitrovykh.ru)
- [ ] Let's Encrypt через nginx/beebot.conf (конфиг готов)
- [ ] JWT в httpOnly cookie

### E.2 ЮKassa — онлайн-оплата
- [ ] Интеграция YooKassa в Telegram-бот
- [ ] QR-код для оплаты
- [ ] Авто-смена статуса после оплаты

### E.3 AGENT_SPECS в Integram
- [ ] Создать таблицу в Integram
- [ ] Промпты агентов из CRM — пчеловод меняет поведение без программиста

---

## Сводная таблица

| # | Приоритет | Задача | Направление |
|---|---|---|---|
| 1 | **P0** | Удалить мёртвый код (28 файлов) | A.1 |
| 2 | **P0** | Закоммитить фиксы | A.2 |
| 3 | **P0** | Обновить документацию | A.3 |
| 4 | **P1** | OrderService → 3 пути создания заказов | B.1 |
| 5 | **P1** | Единый NotificationService | B.2 |
| 6 | **P1** | Импорт 370 заказов из тетрадки | C.1 |
| 7 | **P1** | Привязка клиентов UDS | C.2 |
| 8 | **P2** | Бот → Redis Streams | D.1 |
| 9 | **P2** | ConsultService | D.2 |
| 10 | **P2** | Worker чеклист → SQLite | D.3 |
| 11 | **P3** | SSL + домен | E.1 |
| 12 | **P3** | ЮKassa | E.2 |

---

*Связанные документы: [analysis.md](analysis.md) · [docs/architecture.md](docs/architecture.md)*
