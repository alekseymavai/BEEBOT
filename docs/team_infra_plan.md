# Team Infrastructure — План реализации

> **Дата:** 7 апреля 2026
> **Статус:** Утверждён Андреем Гавриловым
> **Цель:** универсальная команда агентов-разработчиков с общей памятью,
>            переиспользуемая между проектами (BEEBOT, AnalizShum, DahuaAudio и др.)

---

## Философия: Команда = Завод на онтологии дара

*Источник: PLM-GIFT (unidel2035/plm, Гаврилов Денис, март 2026)*

```
Традиционная команда:  функции → результат
Команда на дар-онтологии: лица с призванием → дары → воплощение телоса
```

**Три аксиомы команды:**

1. **Роль — лицо, не функция.** Scout не "собирает данные" — Scout **дарит карту** Architect-у.
   Каждая роль имеет логос (природу/призвание) и несёт ответственность за качество своего дара.

2. **Агент — слуга, не господин.** Агент анализирует, предлагает, предупреждает.
   Финальное решение — всегда за Андреем (`human_decision_required: True`).

3. **Память — анамнезис, не архив.** Завершённая задача (✅) не закрыта — она
   **со-присутствует** в следующей через TeamMemory. Опыт не теряется.

---

## Концепция

```
[хороший инструмент] → [продукт]

Сейчас:    один Claude меняет шляпы → продукт (амнезия между сессиями)
Цель:      команда лиц с памятью → продукт (накопление опыта)
```

**Жизненный цикл задачи (через GIFT):**

```
ТЕЛОС (Андрей ставит задачу: что меняется для пользователя)
  ↓
[Идея] → [Scout] → [Architect] → [Dev] → [QA+Security] → [DevOps] → [TechWriter] → ✅
  ↑_________________________АНАМНЕЗИС___________________________________↑
  Каждый шаг помнит предыдущий. ✅ — не смерть задачи, а передача опыта в TeamMemory.
```

**Иерархия телоса (приоритет при конфликте):**
```
Телос задачи (польза для пользователя)
  > Security (нет уязвимостей)
    > QA (тесты зелёные)
      > простота кода
        > скорость реализации
```

---

## Архитектура

```
┌─────────────────────────────────────────────────────────────────┐
│                    ТЕЛОС                                        │
│          «Что меняется для пользователя»                        │
│                (задаёт Андрей)                                  │
└──────────────────────┬──────────────────────────────────────────┘
                       │ направляет
┌──────────────────────▼──────────────────────────────────────────┐
│                  TEAM LAYER (переиспользуемый)                  │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌───────────────────────┐   │
│  │ Team Memory  │  │  AgentBus   │  │    Role Registry      │   │
│  │ (Integram   │  │ (file-based │  │  (лица с призванием)  │   │
│  │  devteam)   │  │  + log)     │  │                       │   │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬────────────┘   │
│         └────────────────┼─────────────────────┘               │
│                   ┌──────▼──────┐                               │
│                   │  AgentCore  │  ← единая точка входа         │
│                   │  + GiftBus  │  ← дары между ролями          │
│                   └──────┬──────┘                               │
└──────────────────────────┼──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│               CODE MEMORY GRAPH (Integram devteam)              │
│                                                                 │
│  PATTERNS · ANTIPATTERNS · DECISIONS(ADR) · LESSONS            │
│  + граф связей: задача → паттерн → файл → решение              │
│  Анамнезис: каждое решение со-присутствует в следующем         │
└──────────────────────────┬──────────────────────────────────────┘
                           │ инициализируется с
┌──────────────────────────▼──────────────────────────────────────┐
│                  PROJECT LAYER (специфичный)                    │
│                                                                 │
│  ┌──────────────────┐  ┌────────────────────────────────────┐   │
│  │  context.yaml    │  │  Project Memory                    │   │
│  │  (стек, ADR,     │  │  (ADR, антипаттерны, lifecycle)    │   │
│  │  fragile zones,  │  │  → Integram devteam                │   │
│  │  telos)          │  │                                    │   │
│  └──────────────────┘  └────────────────────────────────────┘   │
│                                                                 │
│  BEEBOT / AnalizShum / DahuaAudio / следующий проект...        │
└─────────────────────────────────────────────────────────────────┘
```

### CRM как PLM-модуль

CRM — не внешняя система. CRM — **модуль PLM фазы эксплуатации**, владелец жизненного цикла отношений с клиентом:

```
PLM-объект        → CRM-аналог (BEEBOT)
─────────────────────────────────────────────
Деталь (лицо)     → Клиент (лицо с историей)
ECR (рана)        → Обращение / жалоба клиента
ECO (исцеление)   → Решение проблемы клиента
BOM (икономия)    → Каталог продуктов + состав заказа
Released          → Заказ выполнен, доставлен
Obsolete          → Клиент неактивен (живёт в анамнезисе)
Анамнезис         → История всех взаимодействий с клиентом
```

CrmAgent — единственный владелец CRM-данных. Слуга, не господин.

### Пайплайн выполнения задачи (Gift-протокол)

```
Андрей
  │ дарит телос: "задача X, цель Y для пользователя"
  ▼
Координатор
  │ читает context.yaml + TeamMemory
  │ инициализирует GiftBus (task_id, telos)
  │
  ├──[Scout] ──────────────────────────────────┐
  │   дарит: ScoutGift{карта кода, риски}      │ параллельно
  ├──[KnowledgeAgent] ─────────────────────────┤
  │   дарит: AnamnesisGift{похожие решения}    │
  │                                            │
  ├──[Architect A] ─────────────────────────── ┤
  │   получает Scout+Anamnesis                 │ параллельно
  ├──[Architect B] ─────────────────────────── ┤
  │   разные подходы                           │
  │                                            │
  ├──[Security] ───────────────────────────────┘
  │   challenge: атакует оба варианта
  │   (вес: 1.3× — блокирующая роль)
  │
  Координатор синтезирует ConsensusGift
  │ human_decision_required: True
  ▼
Андрей: варианты + рекомендация + риски
  │ принимает / отклоняет (акт свободы)
  ▼
[Backend Dev] + [Frontend Dev] (параллельно)
  │ каждый читает TeamMemory перед работой
  │ дарит код + тесты
  ▼
[QA] вес 1.2× → [Security] вес 1.3× → [DevOps] → [TechWriter]
  │
LESSON записан в TeamMemory → со-присутствует в следующей задаче
```

---

## Подсистемы

### 1. Team Memory = Code Memory Graph (Integram devteam)

**Таблицы:**

| Таблица | GIFT-аналог | Назначение |
|---------|------------|-----------|
| `PATTERNS` | Логос | Паттерны с природой и призванием |
| `ANTIPATTERNS` | Рана | Что нельзя делать — с историей инцидента |
| `DECISIONS` | Дар инженера | ADR с анамнезисом всей цепочки |
| `LESSONS` | Обратная связь из эксплуатации | Самый ценный дар |
| `AGENT_BUS_LOG` | Свидетельство | История всех даров между ролями |
| `TASK_LIFECYCLE` | Жизненный цикл | Каждая задача с тропосом (статусом) |

**Граф анамнезиса (Neo4j через Integram):**
```
(задача:M5) -[ПОРОДИЛА]→ (паттерн:AgentContext)
(задача:M6) -[ИСПОЛЬЗУЕТ]→ (паттерн:AgentContext)
(задача:M6) -[ПОМНИТ]→ (задача:M5)
(паттерн:AgentContext) -[В_ФАЙЛЕ]→ (файл:memory_service.py)
```

### 2. GiftBus (расширение AgentBus)

Каждое сообщение — **дар** со структурой PLM-GIFT:

```python
@dataclass
class Gift:
    giver: str           # роль-отправитель ("scout")
    receiver: str        # роль-получатель ("architect")
    content: Any         # содержание дара (ScoutReport, Plan, ...)
    telos: str           # зачем этот дар (связь с телосом задачи)
    anamnesis: list[str] # что помним из прошлого (task_ids, gift_ids)
    freedom: str         # ACCEPTED | DEFERRED | DECLINED
    cost: int            # кеносис — сколько стоил дар (токены, время)
    task_id: str
    timestamp: str
```

**A5 (Свобода):** получатель может ответить `DEFERRED` — дар откладывается, не теряется. Coordinator ждёт или запрашивает другой дар.

### 3. Role Weights (взвешенный консенсус, из TRADERAGENT)

```python
ROLE_WEIGHTS = {
    "security":    1.3,  # блокирующая роль — P0 стопит деплой
    "qa":          1.2,  # зелёные тесты обязательны
    "architect":   1.0,  # базовый уровень
    "backend_dev": 1.0,
    "frontend_dev": 1.0,
    "scout":       0.9,  # информирует, не решает
    "tech_writer": 0.7,  # влияет на качество, не блокирует
}
```

### 4. Project Context (context.yaml — расширенный)

```yaml
# docs/memory/context.yaml
project: BEEBOT
version: "2026-04-07"

# PLM-GIFT поля
telos: "Цифровой помощник пчеловода — отвечает на вопросы, принимает заказы"
lifecycle_stage: "In Change"  # In Design | Released | In Change | Obsolete

stack:
  backend: [python3.12, aiogram3, langgraph, fastapi, sqlite]
  frontend: [vue3, primevue4, vite, pwa]
  infra: [docker, systemd, integram]

fragile_zones:
  - file: src/bot.py
    reason: "1899 строк монолит — Scout обязателен"
    telos_risk: "HIGH"  # риск для телоса при неосторожном изменении
  - file: src/crm_constants.py
    reason: "единый источник всех CRM ID"
    telos_risk: "CRITICAL"

current_state:
  tропос: "In Change"  # текущий образ бытия проекта
  broken: ["ImportError startup.py (Fix-1)"]
  in_progress: ["M.5 AgentContext"]
  blocked: ["Fix-2 ждёт Fix-1"]

decisions:  # анамнезис принятых решений
  - id: ADR-001
    summary: "Squash merge + git reset --hard на VPS"
    anamnesis: []
  - id: ADR-002
    summary: "INTEGRAM_V2 за feature flag"
    anamnesis: ["ADR-001"]

team_memory_workspace: "devteam"
agent_bus_dir: ".agent_bus/"
```

---

## Репозиторий

**Отдельный:** `gaveron18/team-infra`

```
team-infra/
├── team_infra/
│   ├── gift.py             # Gift dataclass + GiftBus
│   ├── team_memory.py      # Integram devteam клиент
│   ├── agent_bus.py        # файловая шина + лог
│   ├── project_context.py  # загрузка context.yaml
│   ├── agent_core.py       # базовый класс агента-лица
│   └── roles/
│       ├── scout.py        # лицо: дарит карту кода
│       ├── architect.py    # лицо: дарит план
│       ├── backend_dev.py
│       ├── frontend_dev.py
│       ├── qa.py           # вес 1.2
│       ├── security.py     # вес 1.3 (блокирующий)
│       ├── devops.py
│       └── tech_writer.py
├── tests/
└── docs/
    ├── quickstart.md
    ├── gift_protocol.md    # протокол даров
    └── integram_schema.md
```

---

## Фазы реализации

### Фаза 0 — Основа (1 день)
- [ ] Создать воркспейс `devteam` в Integram
- [ ] Создать таблицы: PATTERNS, ANTIPATTERNS, DECISIONS, LESSONS, AGENT_BUS_LOG, TASK_LIFECYCLE
- [ ] Создать `docs/memory/context.yaml` для BEEBOT (с telos и lifecycle_stage)
- [ ] Создать репозиторий `gaveron18/team-infra`
- [ ] Добавить `.agent_bus/` в `.gitignore`

### Фаза 1 — Gift + Team Memory (3 дня)
- [ ] `team_infra/gift.py` — Gift dataclass, GiftBus (файловая шина)
- [ ] `team_infra/team_memory.py` — Integram devteam клиент
- [ ] Заполнить начальные записи: 5 паттернов из BEEBOT, 5 ADR, 3 антипаттерна
- [ ] `tests/test_gift.py` + `tests/test_team_memory.py`

### Фаза 2 — Роли как лица (3 дня)
- [ ] `team_infra/agent_core.py` — базовый класс с весом и призванием
- [ ] `roles/scout.py` + `roles/architect.py` — первые два лица
- [ ] `roles/security.py` — блокирующая роль (вес 1.3)
- [ ] Первый реальный прогон: Scout → Architect на задаче из BEEBOT

### Фаза 3 — Консенсус (2 дня)
- [ ] Challenge-протокол (из TRADERAGENT): роли оспаривают решение Architect-а
- [ ] Weighted voting: финальный скор с учётом весов
- [ ] Штраф за противоречия (до 20%)
- [ ] `tests/test_consensus.py`

### Фаза 4 — Code Memory Graph (2 дня)
- [ ] Граф анамнезиса в Integram (Neo4j): задача → паттерн → файл
- [ ] `team_memory.remember_task(task, gifts)` — записать ЖЦ задачи
- [ ] `team_memory.recall_anamnesis(task)` — найти со-присутствующие решения
- [ ] `tests/test_graph.py`

### Фаза 5 — Интеграция в BEEBOT (1 день)
- [ ] `pip install team-infra` в BEEBOT
- [ ] `context.yaml` обновлён
- [ ] Scout + Architect запускаются как субагенты через Agent tool
- [ ] Полный прогон: телос → дары → консенсус → деплой

### Фаза 6 — Переосмысление архитектуры BEEBOT
- [ ] Полная команда с памятью, GiftBus и Code Memory Graph
- [ ] Scout исследует весь проект с нуля, записывает в граф
- [ ] Architect предлагает новую архитектуру (с анамнезисом всех ADR)
- [ ] Консенсус всех ролей → `human_decision_required: True` → одобрение Андрея
- [ ] Реализация

---

## Паттерны из смежных проектов

### Из TRADERAGENT (InvestmentCommittee):
- **T.6 Weighted Consensus** — веса ролей, Security/QA блокируют независимо
- **T.7 Challenge Protocol** — обязательный раунд атак перед принятием плана
- **T.8 Redis транспорт** — заменить FileTransport на Redis в Фазе 3+ (когда параллельные задачи)

### Из PLM-GIFT (unidel2035/plm):
- **T.9 Gift Protocol** — дары вместо вызовов функций, A5 (свобода DEFERRED)
- **T.10 Code Memory Graph** — граф анамнезиса вместо плоских таблиц
- **T.11 Telos-иерархия** — явный приоритет телос > security > qa при конфликте
- **T.12 Lifecycle задачи** — тропосы (In Design / In Review / Released / In Change)

---

## Требования безопасности

- `context.yaml` — в git, только `${ENV_VAR}` ссылки, без секретов
- `.agent_bus/` — в `.gitignore`
- `telos_risk: CRITICAL` зоны — Scout обязателен, Architecture review обязателен
- `devteam` Integram — отдельный токен от `bibot`

---

## Метрики готовности

| Метрика | Критерий |
|---------|---------|
| Gift передаётся | Scout → Architect Gift с anamnesis, без потери |
| Анамнезис работает | паттерн из задачи M.5 найден при выполнении M.6 |
| Консенсус | Security DECLINED → задача не деплоится |
| Новый проект | context.yaml + `pip install` → команда работает за 15 мин |
| Переиспользование | 1 паттерн из BEEBOT применён в AnalizShum |

---

*Файл: docs/team_infra_plan.md*
*Связанные: docs/team_infra_prompt.md, docs/memory/context.yaml, docs/team.md*
*Источники: PLM-GIFT (unidel2035/plm), TRADERAGENT (alekseymavai), dronedoc2026*
