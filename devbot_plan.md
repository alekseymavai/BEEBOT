# DEVBOT — Детальный план автономного разработчика

> **Дата:** 27 марта 2026 (обновлено после анализа hive-mind)
> **Статус:** Планирование
> **Связан с:** [plan.md](plan.md) Фаза 7
> **Референс:** [konard/hive-mind](https://github.com/konard/hive-mind) — аналогичная система для GitHub Issues (JS)

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

## Паттерны из hive-mind (референс)

Проект [konard/hive-mind](https://github.com/konard/hive-mind) решает аналогичную задачу (автономное решение GitHub-issues через Claude CLI + Telegram). Заимствуем:

| Паттерн | hive-mind | Наш DEVBOT |
|---------|-----------|------------|
| Разделение промптов | `buildSystemPrompt` + `buildUserPrompt` | `prompts.py` — две функции |
| Флаги Claude CLI | `--output-format stream-json --append-system-prompt` | те же флаги |
| Продолжение сессии | `--resume <session_id>` | при обрыве длинной задачи |
| Feedback loop | PR-комментарии → переподача в Claude | ответ Александра → перезапуск |
| Auto-continue | автовозобновление при лимите токенов | аналогично |
| Стриминг вывода | NDJSON парсинг → прогресс в Telegram | прогресс-апдейты каждые 30 сек |

### Ключевые флаги Claude CLI (из hive-mind)
```bash
claude \
  --output-format stream-json \   # стриминг для прогресс-апдейтов
  --verbose \                      # диагностика
  --model claude-sonnet-4-6 \
  -p "{user_prompt}" \
  --append-system-prompt "{rules}" # добавить правила поверх CLAUDE.md
```

---

## Структура DEVBOT

```
BEEBOT/
└── src/
    └── devbot/
        ├── bot.py          # Точка входа, aiogram polling, /start /approve /edit /cancel /feedback
        ├── fsm.py          # FSM: receive → analyze → confirm → execute → feedback → done
        ├── prompts.py      # build_system_prompt() + build_user_prompt() (паттерн hive-mind)
        ├── analyzer.py     # Claude API: анализ задачи → план изменений
        ├── executor.py     # Запуск claude CLI со стримингом, auto-continue, feedback loop
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
   ├─ /edit <уточнение> → [ANALYZING] (перезапуск с фидбеком)
   └─ /cancel → [IDLE]
   ▼
[EXECUTING]  — claude CLI запущен со стримингом (паттерн hive-mind)
   │ прогресс-апдейты каждые 30 сек
   │ при обрыве → auto-continue через --resume <session_id>
   ▼
[FEEDBACK]   — Александр может скорректировать результат
   ├─ ответ в течение 10 мин → [EXECUTING] (feedback loop, паттерн hive-mind)
   └─ нет ответа / /ok → [REPORTING]
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
# executor.py — паттерн из hive-mind: stream-json + auto-continue
import asyncio, json

async def execute(
    task: str, plan: str, dev_memory: str,
    feedback: str | None = None,
    session_id: str | None = None,        # для --resume (auto-continue)
    progress_cb=None,                      # callback для Telegram-апдейтов
) -> dict:
    user_prompt = build_user_prompt(task, plan, dev_memory, feedback)
    system_rules = build_system_prompt()   # правила поверх CLAUDE.md

    args = [
        "claude",
        "--output-format", "stream-json",  # стриминг → прогресс
        "--verbose",
        "--model", "claude-sonnet-4-6",
        "-p", user_prompt,
        "--append-system-prompt", system_rules,
    ]
    if session_id:
        args += ["--resume", session_id]   # продолжить прерванную сессию

    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd="/home/new/BEEBOT",
    )

    result_lines, new_session_id = [], None
    async for line in proc.stdout:
        try:
            event = json.loads(line)
            if event.get("type") == "session_id":
                new_session_id = event["session_id"]
            if event.get("type") == "text" and progress_cb:
                await progress_cb(event["text"])   # → Telegram
            result_lines.append(event)
        except json.JSONDecodeError:
            pass

    await proc.wait()
    return {"events": result_lines, "session_id": new_session_id, "exit_code": proc.returncode}
```

### build_system_prompt (prompts.py)
```python
MEMORY_DIR = "/home/new/.claude/projects/-home-new-BEEBOT/memory"

def build_system_prompt() -> str:
    return f"""
Ты опытный Python-разработчик. Репозиторий: /home/new/BEEBOT.
Стек: Python 3.12, aiogram 3, FastAPI, Vue 3, FAISS, Integram CRM.

Правила выполнения задачи:
1. Прочитай затронутые файлы перед правками
2. Запусти pytest после изменений — при ошибках СТОП, доложи причину
3. Коммит: git add <файлы> && git commit (не git add -A)
4. PR: gh pr create → gh pr merge --squash
5. Deploy: ssh ai-agent@185.233.200.13 "cd /home/ai-agent/BEEBOT && git pull && docker compose up -d --build"
6. Итог: список изменённых файлов + ссылка на PR + SHA коммита

Обновление памяти (ОБЯЗАТЕЛЬНО после успешного выполнения):
7. Обнови {MEMORY_DIR}/tasks_roadmap.md — отметь задачу выполненной
8. Если изменилась архитектура/стек — обнови {MEMORY_DIR}/project_beebot.md
9. Если есть нетривиальный урок (решение, антипаттерн, предпочтение Александра) —
   создай новый файл {MEMORY_DIR}/lesson_<тема>.md с frontmatter type: feedback
   и добавь строку в {MEMORY_DIR}/MEMORY.md
"""
```

> **Почему это критично:** каждый запуск `claude --print` — это новая сессия.
> Без явной инструкции обновить файлы памяти я прочитаю контекст, но не запишу уроки.
> С этим правилом память накапливается автоматически даже через Telegram.

---

## Запись в память после выполнения

Память обновляется на **двух уровнях** — оба обязательны:

```
┌─────────────────────────────────────────────────────────┐
│  Уровень 1: файлы Claude Code (hive)                    │
│  /home/new/.claude/projects/.../memory/                  │
│  Кто пишет: я сам (Claude Code) по инструкции           │
│             в build_system_prompt()                      │
│  Что пишу:  tasks_roadmap.md, новые lesson_*.md,         │
│             обновления project_beebot.md                 │
│  Читается:  при каждом запуске claude (и здесь, и через │
│             Telegram)                                    │
├─────────────────────────────────────────────────────────┤
│  Уровень 2: Integram bibot (облако)                     │
│  Кто пишет: DEVBOT Python-код (memory.py)               │
│  Что пишет: Задачи разработки, Память разработчика      │
│  Читается:  analyzer.py при анализе новой задачи        │
└─────────────────────────────────────────────────────────┘
```

```python
# memory.py — после успешного deploy
async def record_completion(task, plan, pr_url, files_changed, lessons):
    # Уровень 2: Integram
    await integram.update_task(task_id, status="готово", pr=pr_url, files=files_changed)
    await integram.add_dev_memory(
        topic=task[:60],
        context=task,
        solution=plan,
        files=files_changed,
        pr=pr_url,
        lessons=lessons,
    )
    # Уровень 1 (файлы памяти) обновляет сам Claude Code
    # по инструкции в build_system_prompt() — пп. 7-9
```

### Что это означает на практике

- Задача пришла через Telegram → я выполнил → **оба уровня памяти обновлены**
- Следующая сессия (Telegram или прямая) → я вижу что делали, почему, какие уроки
- Александр может спросить `/devmemory` → получит сводку из Integram
- Я при старте читаю `MEMORY.md` → вижу то же самое из файлов

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

- [ ] `src/devbot/memory.py` — запись в Integram после deploy (уровень 2)
- [ ] Чтение памяти Integram при анализе задачи (контекст для analyzer.py)
- [ ] `build_system_prompt()` — пп. 7-9: инструкция обновлять файлы памяти (уровень 1)
- [ ] Проверить что файлы памяти действительно обновляются после тестовой задачи
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
