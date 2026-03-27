# DEVBOT — Детальный план автономного разработчика

> **Дата:** 27 марта 2026
> **Статус:** Планирование
> **Связан с:** [plan.md](plan.md) Фаза 7

---

## Концепция

**DEVBOT** — отдельный Telegram-бот-разработчик. Александр описывает задачу в чате, DEVBOT:
1. Анализирует и показывает план
2. Ждёт подтверждения / уточнений
3. Пишет код, тестирует, деплоит
4. Записывает результат в память разработчика
5. Докладывает в Telegram

---

## Архитектура

### Где живёт

```
hive (локальная машина)
├── DEVBOT процесс (aiogram, polling)
│   └── Токен: отдельный от BEEBOT
├── Claude Code CLI (уже установлен — это я)
│   └── /home/new/BEEBOT/ — рабочий репозиторий
└── SSH-доступ к VPS (уже настроен)

VPS 185.233.200.13
└── Docker (beebot, beebot-web) — деплой как сейчас
```

**Почему hive, а не VPS:**
- Claude Code CLI работает на hive
- Репозиторий на hive
- SSH к VPS уже есть
- Не нагружает VPS (2GB RAM)

---

### Схема взаимодействия

```
Александр
  │
  ▼
BEEBOT (VPS)
  │  /dev <задача>
  │  → сохраняет задачу в Integram (таблица «Задачи разработки»)
  │  → HTTP POST → DEVBOT API (hive:8091)
  │
  ▼
DEVBOT (hive)
  │
  ├─ 1. Анализ задачи через Claude API
  │     → «Вот что нужно изменить:
  │        1. src/models.py — поле delivery_comment
  │        2. src/web/routers/orders.py — endpoint
  │        3. web/src/views/OrderDetail.vue — форма
  │        Подтверждаешь?»
  │
  ├─ 2. Диалог с Александром
  │     → /approve — выполнить
  │     → /edit    — уточнить план
  │     → /cancel  — отменить
  │
  ├─ 3. Исполнение (claude --print)
  │     → Claude Code читает репо, вносит изменения
  │     → pytest — тесты должны пройти
  │     → git commit → push → gh pr create → gh pr merge
  │     → ssh VPS "docker compose up -d --build"
  │
  ├─ 4. Запись в память разработчика
  │     → Integram: таблица «Память разработчика»
  │     → Файлы памяти: /home/new/.claude/projects/.../memory/
  │
  └─ 5. Отчёт в Telegram
        → «Готово! PR #92 смержен. Проверь: OrderDetail.vue»
```

---

## Integram: новые таблицы

### Таблица 1: «Советы пчеловода»

Хранит операционные знания Александра — загружаются в контекст BEEBOT и DEVBOT.

| Поле | Тип | Описание |
|------|-----|----------|
| Текст | textarea | Сам совет |
| Категория | select | клиент / crm / продукт / процесс |
| Приоритет | select | высокий / средний / справочный |
| Статус | select | активен / архив |

**Как используется BEEBOT:**
- категория `клиент` → добавляется в системный промпт консультанта
- категория `продукт` → добавляется в KB через `/teach`

**Как используется DEVBOT:**
- категория `crm` / `процесс` → передаётся в контекст Claude Code при старте задачи

---

### Таблица 2: «Задачи разработки»

Очередь и журнал задач от Александра.

| Поле | Тип | Описание |
|------|-----|----------|
| Описание | textarea | Текст задачи |
| Статус | select | новая / анализ / подтверждение / выполняется / готово / отменена / ошибка |
| Приоритет | select | срочно / обычный / когда-нибудь |
| Файлы затронуты | text | Автозаполняется DEVBOT |
| PR-ссылка | text | Автозаполняется DEVBOT |
| Коммит | text | SHA коммита |
| Уроки | textarea | Что узнали из диалога (для памяти) |
| Дата создания | datetime | Авто |
| Дата выполнения | datetime | Авто |

---

### Таблица 3: «Память разработчика»

Долгосрочная база решений — что делали, как решили, почему.

| Поле | Тип | Описание |
|------|-----|----------|
| Тема | text | Краткое название (напр. «delivery_comment поле») |
| Контекст | textarea | Почему Александр попросил |
| Решение | textarea | Что именно сделано |
| Файлы | text | Список изменённых файлов |
| PR | text | Ссылка на GitHub PR |
| Антипаттерн | textarea | Что НЕ делать (из отказов Александра) |
| Категория | select | модель / api / frontend / kb / infra / crm |
| Дата | datetime | Авто |

---

## Структура DEVBOT

```
BEEBOT/
└── src/
    └── devbot/
        ├── bot.py          # Точка входа, aiogram polling, /start /approve /edit /cancel
        ├── fsm.py          # FSM: receive → analyze → confirm → execute → done
        ├── analyzer.py     # Claude API: анализ задачи → план изменений
        ├── executor.py     # Запуск claude --print, сбор вывода, отчёт
        ├── memory.py       # Чтение/запись Integram (таблицы 2 и 3) + локальные файлы
        └── config.py       # DEVBOT_TOKEN, DEVBOT_ADMIN_CHAT_ID, DEVBOT_API_PORT
```

---

## FSM (состояния диалога)

```
[IDLE]
   │ /dev <задача> из BEEBOT или напрямую
   ▼
[ANALYZING]  — Claude API анализирует задачу (~10 сек)
   │ отправляет план + кнопки
   ▼
[CONFIRMING]  — ждёт ответа Александра
   ├─ /approve → [EXECUTING]
   ├─ /edit <уточнение> → [ANALYZING] (перезапуск)
   └─ /cancel → [IDLE]
   ▼
[EXECUTING]  — claude --print запущен
   │ прогресс-апдейты каждые 30 сек
   ▼
[REPORTING]  — тесты ✅ → merge → deploy → запись памяти
   └─ → [IDLE]
```

---

## Как DEVBOT анализирует задачу (Analyzer)

```python
# analyzer.py — использует Claude API напрямую (не claude --print)
prompt = f"""
Ты архитектор проекта BEEBOT (Python/FastAPI/Vue3/aiogram).

Задача от пчеловода: {task_description}

Память разработчика (последние решения):
{dev_memory_context}

Советы пчеловода (категория crm/процесс):
{advice_context}

Проанализируй и верни план:
1. Список файлов для изменения (с кратким описанием)
2. Оценка сложности (простая / средняя / сложная)
3. Риски (если есть)
4. Нужна ли пересборка KB / Docker
"""
# → Anthropic API (claude-sonnet-4-6)
```

---

## Как DEVBOT исполняет задачу (Executor)

```python
# executor.py
import subprocess, asyncio

async def execute(task: str, plan: str, dev_memory: str) -> str:
    prompt = f"""
Repository: /home/new/BEEBOT
Task: {task}
Agreed plan: {plan}
Developer memory context: {dev_memory}

Rules:
- Run pytest after changes, abort if tests fail
- Commit, create PR via gh, merge with squash
- Deploy: ssh ai-agent@185.233.200.13 "cd /home/ai-agent/BEEBOT && git pull && docker compose up -d --build"
- Report: list of changed files + PR link
"""
    proc = await asyncio.create_subprocess_exec(
        "claude", "--print", prompt,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd="/home/new/BEEBOT",
    )
    stdout, _ = await proc.communicate()
    return stdout.decode()
```

---

## Запись в память после выполнения

```python
# memory.py — после успешного deploy
async def record_completion(task, plan, pr_url, files_changed, lessons):
    # 1. Integram: обновить задачу (статус → готово, PR-ссылка, файлы)
    await integram.update_task(task_id, status="готово", pr=pr_url, files=files_changed)

    # 2. Integram: новая запись в Память разработчика
    await integram.add_dev_memory(
        topic=task[:60],
        context=task,
        solution=plan,
        files=files_changed,
        pr=pr_url,
        lessons=lessons,
    )

    # 3. Локальные файлы памяти Claude Code
    # /home/new/.claude/projects/-home-new-BEEBOT/memory/
    # → обновить tasks_roadmap.md или создать project_devhistory.md
```

---

## Интеграция с BEEBOT

### Команда `/dev` в BEEBOT

```python
# src/bot.py — только ADMIN_CHAT_ID
@dp.message(Command("dev"))
async def cmd_dev(message: types.Message):
    task = message.text.removeprefix("/dev").strip()
    if not task:
        await message.answer("Использование: /dev <описание задачи>")
        return
    # Сохранить в Integram
    task_id = await integram.create_dev_task(task)
    # Уведомить DEVBOT
    async with httpx.AsyncClient() as c:
        await c.post("http://localhost:8091/task", json={"id": task_id, "text": task})
    await message.answer(f"✅ Задача #{task_id} передана разработчику. Ожидай анализ.")
```

### HTTP API DEVBOT (hive:8091)

```
POST /task  — принять новую задачу от BEEBOT
GET  /status/<task_id>  — статус задачи
GET  /health
```

---

## Безопасность и ограничения

| Правило | Реализация |
|---------|-----------|
| Только ADMIN_CHAT_ID | Проверка в каждом хэндлере |
| Тесты обязательны | `pytest` до merge, при fail → стоп |
| Подтверждение всегда | FSM не пропускает EXECUTING без /approve |
| Нет force push | gh pr merge --squash только |
| Нет DROP TABLE / rm -rf | Фильтр в executor.py на опасные паттерны |
| Лог всех операций | Integram «Задачи разработки» — полная история |

---

## Запуск DEVBOT

### systemd-сервис (hive)

```ini
# /etc/systemd/system/devbot.service
[Unit]
Description=BEEBOT Developer Agent
After=network.target

[Service]
User=new
WorkingDirectory=/home/new/BEEBOT
ExecStart=/home/new/.local/bin/python -m src.devbot.bot
Restart=on-failure
EnvironmentFile=/home/new/BEEBOT/.env

[Install]
WantedBy=multi-user.target
```

### .env добавить

```
DEVBOT_TOKEN=<токен от @BotFather>
DEVBOT_ADMIN_CHAT_ID=<telegram_id Александра>
DEVBOT_API_PORT=8091
ANTHROPIC_API_KEY=<ключ для анализатора>
```

---

## Фазы реализации

### Фаза 7.1 — Integram + Советы (3-4 дня)

- [ ] Создать таблицу «Советы пчеловода» в Integram
- [ ] Создать таблицу «Задачи разработки» в Integram
- [ ] Создать таблицу «Память разработчика» в Integram
- [ ] Расширить `OntologyCache` — загружать советы при старте BEEBOT
- [ ] Инжектировать советы (категория `клиент`) в системный промпт консультанта
- [ ] Команда `/advice <текст> [категория]` — добавить совет через бот

### Фаза 7.2 — DEVBOT MVP (5-7 дней)

- [ ] `src/devbot/config.py` — конфиг
- [ ] `src/devbot/bot.py` — хэндлеры `/dev`, `/approve`, `/cancel`
- [ ] `src/devbot/fsm.py` — FSM состояний
- [ ] `src/devbot/analyzer.py` — Claude API анализ задачи
- [ ] `src/devbot/executor.py` — запуск `claude --print`
- [ ] FastAPI endpoint hive:8091 — приём задач от BEEBOT
- [ ] Интеграция `/dev` в `src/bot.py`
- [ ] systemd-сервис `devbot.service` на hive

### Фаза 7.3 — Память разработчика (3-4 дня)

- [ ] `src/devbot/memory.py` — запись в Integram после deploy
- [ ] Чтение памяти при анализе задачи (контекст)
- [ ] Обновление локальных файлов памяти Claude Code
- [ ] Автообновление `CLAUDE.md` при архитектурных изменениях

### Фаза 7.4 — Полировка (2-3 дня)

- [ ] Прогресс-апдейты во время выполнения (каждые 30 сек)
- [ ] Команда `/devstatus` — текущие задачи
- [ ] Команда `/devhistory` — последние 10 задач
- [ ] Команда `/devmemory` — что помним о репо
- [ ] Фильтр сложности: если Claude оценивает «сложная» → доп. подтверждение

---

## Что нужно для старта

1. **Токен DEVBOT** — создать через @BotFather
2. **ANTHROPIC_API_KEY** — для анализатора (Analyzer использует API напрямую, не через Groq)
3. **Integram**: создать 3 таблицы (делается через MySQL или Integram-интерфейс)

---

## Итог: что изменится в системе

```
До:
  Александр → BEEBOT → LLM → ответ

После:
  Александр → BEEBOT → консультация (+ советы пчеловода в промпте)
  Александр → BEEBOT /dev → DEVBOT → план → подтверждение → Claude Code
                                           → тесты → PR → deploy → память
                                           → отчёт Александру
```

---

*Документ создан: 27.03.2026*
*Следующий шаг: получить токен DEVBOT + создать таблицы в Integram*
