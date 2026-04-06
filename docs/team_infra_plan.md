# Team Infrastructure — План реализации

> **Дата:** 6 апреля 2026
> **Статус:** Утверждён Андреем Гавриловым
> **Цель:** универсальная команда агентов-разработчиков с общей памятью,
>            переиспользуемая между проектами (BEEBOT, AnalizShum, DahuaAudio и др.)

---

## Концепция

```
[хороший инструмент] → [продукт]

Сейчас:    один Claude меняет шляпы → продукт (амнезия между сессиями)
Цель:      команда агентов с памятью → продукт (накопление опыта)
```

Каждый проект подключает `team-infra` как зависимость.
Команда помнит паттерны, решения, антипаттерны — между сессиями и между проектами.
Andрей видит всё через Integram — память команды прозрачна.

---

## Архитектура

```
┌─────────────────────────────────────────────────────────────┐
│                  TEAM LAYER (переиспользуемый)              │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │  Team Memory  │  │  AgentBus    │  │  Role Registry   │  │
│  │  (Integram    │  │  (file-based │  │  (YAML prompts)  │  │
│  │   devteam)    │  │  + log)      │  │                  │  │
│  └──────┬───────┘  └──────┬───────┘  └────────┬─────────┘  │
│         └─────────────────┼────────────────────┘            │
│                    ┌──────▼──────┐                          │
│                    │  AgentCore  │  ← единая точка входа    │
│                    └──────┬──────┘                          │
└───────────────────────────┼─────────────────────────────────┘
                            │ инициализируется с
┌───────────────────────────▼─────────────────────────────────┐
│                  PROJECT LAYER (специфичный)                 │
│                                                             │
│  ┌─────────────────┐  ┌──────────────────────────────────┐  │
│  │  context.yaml   │  │  Project Memory                  │  │
│  │  (стек, ADR,    │  │  (ADR, антипаттерны, state)      │  │
│  │   fragile zones)│  │  → Integram devteam              │  │
│  └─────────────────┘  └──────────────────────────────────┘  │
│                                                             │
│  BEEBOT / AnalizShum / DahuaAudio / следующий проект...    │
└─────────────────────────────────────────────────────────────┘
```

### Пайплайн выполнения задачи

```
Андрей
  │ "разработайте X"
  ▼
Координатор (главный Claude)
  │ читает context.yaml
  │ инициализирует AgentBus
  │
  ├──[Agent: Scout]─────────────────────────────┐
  │    читает TeamMemory(devteam)                │ параллельно
  │    исследует кодовую базу                   │
  │    → ScoutReport → bus/inbox/architect      │
  │                                             │
  ├──[Agent: Architect A]───────────────────────┤
  │    вариант 1                                │ параллельно
  ├──[Agent: Architect B]───────────────────────┤
  │    вариант 2                                │
  │                                             │
  ├──[Agent: Security]──────────────────────────┘
  │    оценивает оба варианта
  │
  Координатор синтезирует консенсус
  │ пишет DECISIONS в Integram(devteam)
  │ обновляет context.yaml
  ▼
Андрей: два варианта + рекомендация + риски
  │ одобряет
  ▼
[Backend Dev] + [Frontend Dev] (параллельно)
  │ каждый читает TeamMemory перед работой
  │ пишет в AgentBus по завершении
  ▼
QA → Security → DevOps → Tech Writer
  │ каждый логирует в Integram
  ▼
LESSON записан в TeamMemory(devteam)
  → доступен в следующей сессии и следующем проекте
```

---

## Подсистемы

### 1. Team Memory (Integram, воркспейс `devteam`)

**Таблицы:**

| Таблица | Назначение | Ключевые поля |
|---------|-----------|---------------|
| `PATTERNS` | Паттерны которые работают | name, context, solution, when_to_use, project_examples |
| `ANTIPATTERNS` | Что нельзя делать и почему | name, what_not_to_do, why, incident, projects_affected |
| `DECISIONS` | Архитектурные решения (ADR) | project, decision, rationale, alternatives, status, date |
| `LESSONS` | Уроки из инцидентов | project, what_happened, root_cause, fix, prevention |
| `AGENT_BUS_LOG` | История передачи эстафеты | session_id, from_role, to_role, task_id, payload, ts |

**Почему Integram:**
- REST API из любого агента
- Семантический поиск (`semantic_search`)
- Ref-колонки — связи между записями
- Уже используется в проекте (`integram_api.py`)
- Ты видишь всё через веб-интерфейс — память прозрачна
- Разные воркспейсы: `bibot` (CRM бота) vs `devteam` (память команды)

### 2. AgentBus (файловая шина + лог в Integram)

```
.agent_bus/              ← в .gitignore
  inbox/
    scout.json           ← задание для Scout
    architect.json       ← ScoutReport для Architect
  outbox/
    scout_to_architect.json
  consensus/
    task_M6.json         ← результат совещания по задаче M.6
```

Формат сообщения:
```json
{
  "task_id": "M.6",
  "from": "scout",
  "to": "architect",
  "timestamp": "2026-04-06T10:00:00",
  "payload": {
    "files_analyzed": ["src/memory_service.py"],
    "patterns_found": ["service layer"],
    "risks": ["N+1 при sync"],
    "open_questions": ["частота синхронизации?"]
  }
}
```

### 3. Project Context (context.yaml)

```yaml
# docs/memory/context.yaml
project: BEEBOT
version: "2026-04-06"

stack:
  backend: [python3.12, aiogram3, langgraph, fastapi, sqlite]
  frontend: [vue3, primevue4, vite, pwa]
  infra: [docker, systemd, integram]

fragile_zones:
  - file: src/bot.py
    reason: "1899 строк монолит — Scout обязателен перед касанием"
  - file: src/crm_constants.py
    reason: "единый источник всех CRM ID — изменение ломает всё"

current_state:
  broken: ["ImportError в startup.py (Fix-1)"]
  in_progress: ["M.5 AgentContext (feat/m5-agent-context)"]
  blocked: ["Fix-2 A.7 ждёт Fix-1"]

decisions:
  - id: ADR-001
    summary: "Squash merge всегда, git reset --hard на VPS (не pull)"
  - id: ADR-002
    summary: "INTEGRAM_V2 за feature flag — переключение без деплоя"
  - id: ADR-003
    summary: "network_mode: host в Docker — tunnels работают"

team_memory_workspace: "devteam"
agent_bus_dir: ".agent_bus/"
```

### 4. Role Registry (машинно-читаемые промты)

```yaml
# docs/agents/scout.yaml
role: scout
description: "Исследует кодовую базу перед любым изменением"
reads_from: ["context.yaml", "team_memory.PATTERNS", "team_memory.ANTIPATTERNS"]
writes_to: ["agent_bus.outbox/scout_to_architect.json"]
tools: [Glob, Grep, Read]  # только чтение, никаких изменений
output_schema:
  files_analyzed: list[str]
  patterns_found: list[str]
  risks: list[str]
  fragile_zones_touched: list[str]
  recommendation: str
```

---

## Репозиторий

**Отдельный репозиторий:** `gaveron18/team-infra`

```
team-infra/
├── team_infra/
│   ├── __init__.py
│   ├── team_memory.py      # Integram-клиент для командной памяти
│   ├── agent_bus.py        # файловая шина + лог
│   ├── project_context.py  # загрузка context.yaml
│   ├── agent_core.py       # базовый класс агента
│   └── roles/
│       ├── scout.py
│       ├── architect.py
│       ├── backend_dev.py
│       ├── frontend_dev.py
│       ├── qa.py
│       ├── security.py
│       ├── devops.py
│       └── tech_writer.py
├── tests/
│   ├── test_team_memory.py
│   ├── test_agent_bus.py
│   ├── test_project_context.py
│   └── test_full_pipeline.py
├── docs/
│   ├── quickstart.md
│   ├── integram_schema.md
│   └── bus_protocol.md
├── pyproject.toml
└── README.md
```

Подключение в проекте:
```bash
pip install git+https://github.com/gaveron18/team-infra.git
```

---

## Фазы реализации

### Фаза 0 — Основа (1 день)
- [ ] Создать воркспейс `devteam` в Integram
- [ ] Создать таблицы: PATTERNS, ANTIPATTERNS, DECISIONS, LESSONS, AGENT_BUS_LOG
- [ ] Создать `docs/memory/context.yaml` для BEEBOT
- [ ] Создать репозиторий `gaveron18/team-infra`
- [ ] Добавить `.agent_bus/` в `.gitignore`

### Фаза 1 — Team Memory (3 дня)
- [ ] `team_infra/team_memory.py` — Integram-клиент
- [ ] Заполнить начальные записи: 5 паттернов из BEEBOT, 5 ADR, 3 антипаттерна
- [ ] `tests/test_team_memory.py` — запись / чтение / поиск
- [ ] Все тесты зелёные

### Фаза 2 — AgentBus (2 дня)
- [ ] `team_infra/agent_bus.py` — файловая шина
- [ ] Логирование в Integram `AGENT_BUS_LOG`
- [ ] `tests/test_agent_bus.py` — Scout → Architect передача
- [ ] Все тесты зелёные

### Фаза 3 — Роли как код (3 дня)
- [ ] `team_infra/agent_core.py` — базовый класс
- [ ] `roles/scout.py` + `roles/architect.py` — машинные промты
- [ ] Первый реальный прогон на задаче из BEEBOT
- [ ] Документация: quickstart.md

### Фаза 4 — Интеграция в BEEBOT (1 день)
- [ ] `pip install team-infra` в BEEBOT
- [ ] Scout + Architect запускаются как субагенты
- [ ] Проверка на реальной задаче из plan.md

### Фаза 5 — Переосмысление архитектуры BEEBOT
- [ ] Полная команда с памятью и AgentBus
- [ ] Scout исследует весь проект с нуля
- [ ] Architect предлагает новую архитектуру
- [ ] Консенсус → одобрение Андрея → реализация

---

## Требования безопасности

- `context.yaml` — в git, без секретов (только `${ENV_VAR}` ссылки)
- `.agent_bus/` — в `.gitignore`, временные данные
- `devteam` Integram — отдельный токен, не смешивать с `bibot`
- Локальный SQLite-fallback при недоступности Integram (Фаза 1+)

---

## Метрики готовности

| Метрика | Критерий |
|---------|---------|
| Scout как субагент | возвращает ScoutReport за < 2 мин |
| Team Memory | паттерн записан → найден поиском в новой сессии |
| AgentBus | Scout → Architect без потери данных |
| Новый проект | context.yaml + `pip install` → команда работает за 15 мин |
| Переиспользование | 1 паттерн из BEEBOT применён в AnalizShum |

---

*Файл: docs/team_infra_plan.md*
*Связанные файлы: docs/team_infra_prompt.md, docs/memory/context.yaml, docs/team.md*
