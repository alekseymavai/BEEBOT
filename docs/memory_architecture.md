# Архитектура памяти BEEBOT

*Анализ проведён 03.04.2026 на основе best practices индустрии (mem0, LangMem, Letta, Redis Agent Memory Server, arxiv 2025)*

---

## 1. Все виды памяти в BEEBOT (текущее состояние)

В проекте 5 разных механизмов памяти, каждый для своей задачи:

### 1.1 Диалоговая история (RAM, TTL 30 мин)

Файл: `src/shared_context.py` — `SharedContextStore` → `UserContext`

| | |
|---|---|
| **Что хранит** | Последние 5 пар «вопрос — ответ» |
| **Где живёт** | RAM (dict по user_id) |
| **TTL** | 30 минут без активности → сброс |
| **Кто использует** | Консультант (beebot.py) через оркестратор |
| **Зачем** | LLM видит контекст беседы, может отвечать на «а ещё?» |

Ассистент (`admin_chat.py`) хранит свою историю отдельно — `self._history: dict[int, list]`, последние 10 пар. Тоже RAM, тоже теряется при рестарте.

### 1.2 Долгосрочная память пользователей (SQLite)

Файл: `src/memory.py` — `UserMemory`

| | |
|---|---|
| **Что хранит** | Факты о здоровье/интересах: «у меня язва», «принимаю пергу» |
| **Где живёт** | `data/memory.db` (SQLite на VPS) |
| **TTL** | Бессрочно |
| **Кто использует** | Консультант — загружает факты → передаёт в LLM как контекст |
| **Как попадает** | Автодетект regex в тексте пользователя (`extract_fact()`) или ручное `/remember` |

### 1.3 Эпизодическая память / Анамнез (агрегатор)

Файл: `src/anamnesis.py` — `AnamnesisCache`

| | |
|---|---|
| **Что хранит** | Объединяет SQLite-факты + историю заказов из CRM |
| **Где живёт** | Агрегируется на лету из #2 + CRM |
| **Кто использует** | Консультант — подсказка «Вы уже брали: заказ #42 — Доставлен, 2800₽» |
| **Зачем** | Персонализация: LLM знает что клиент уже покупал |

### 1.4 CRM-снапшот (RAM-кэш CRM)

Файл: `src/crm_snapshot.py` — `CrmSnapshot`

| | |
|---|---|
| **Что хранит** | Заказы + позиции + клиенты + товары из Integram |
| **Где живёт** | RAM, обновляется каждые 5 мин из CRM API |
| **Кто использует** | Ассистент (admin_chat.py) — пчеловод спрашивает «сколько заказов?» |
| **Зачем** | Не дёргать CRM API на каждый вопрос |

### 1.5 Память DEVBOT (Integram + файлы)

Файл: `src/devbot/memory.py` — `DevMemory`

| | |
|---|---|
| **Что хранит** | Задачи разработки, уроки, антипаттерны |
| **Где живёт** | Integram — 3 таблицы: DEV_TASKS, DEV_MEMORY, DEV_ADVICE |
| **Кто использует** | DEVBOT — запоминает что делал, какие ошибки были |
| **Зачем** | Контекст между сессиями разработки |

---

## 2. Какой агент что видит

| Агент | Диалог (RAM) | Факты (SQLite) | Анамнез (CRM) | CRM-снапшот | DEVBOT-память |
|-------|-------------|----------------|---------------|-------------|---------------|
| Консультант | ✅ 5 пар | ✅ | ✅ | — | — |
| Логист | — | — | — | — | — |
| Аналитик | — | — | — | — | — |
| Инспектор | — | — | — | — | — |
| Ассистент | ✅ 10 пар (своя) | — | — | ✅ | — |
| Работник | — | — | — | — | — |
| DEVBOT | — | — | — | — | ✅ |

**Главная проблема:** только Консультант имеет полноценную память. Остальные агенты — без памяти, каждый раз с чистого листа.

---

## 3. Что говорит мировая практика (best practices 2025)

### 3.1 Таксономия уровней памяти (индустриальный стандарт)

| Уровень | Срок жизни | Пример | BEEBOT сейчас | Разрыв |
|---------|-----------|--------|---------------|--------|
| **Рабочая** | секунды–минуты | «о чём мы сейчас говорим» | ✅ RAM dict | Нет Redis-изоляции по agent_id |
| **Эпизодическая** | часы–дни | «вчера клиент спрашивал пергу» | ❌ Отсутствует | **Ключевая дыра** |
| **Семантическая** | недели–месяцы | «у клиента язва», «уже брал» | ⚠️ SQLite без agent_id | Нет изоляции агентов |
| **Институциональная** | постоянно | KB, промпты, онтология | ✅ FAISS + Integram | Ок |

### 3.2 Ключевые библиотеки и их архитектуры

**mem0** (30k+ звёзд, [github.com/mem0ai/mem0](https://github.com/mem0ai/mem0))

Три параллельных слоя хранения. LLM сам извлекает факты из диалога и записывает. Namespace по трём осям: `user_id / agent_id / run_id`.

```python
memory.add("Предпочитает мёд с акации", {
    "user_id": "user_123",
    "agent_id": "consultant_agent",
    "run_id": "session_456"
})
# Консультант видит факты → Логист тоже видит при оформлении заказа
memories = memory.search("мёд", user_id="user_123")  # все агенты
```

Метрики vs OpenAI full-context: +26% accuracy, −91% p95 latency, −90% токенов.

**LangMem** ([github.com/langchain-ai/langmem](https://github.com/langchain-ai/langmem))

Три типа памяти: Episodic (прошлые события) + Semantic (факты) + **Procedural** (обновляет системный промпт). Агент буквально «обучается» без fine-tuning через накопленный опыт.

**LangGraph checkpointer** — встроенный shared state

```python
checkpointer = SqliteSaver.from_conn_string("data/checkpoints.db")
graph = builder.compile(checkpointer=checkpointer)
# Весь граф-стейт персистируется. Эпизодическая память — бесплатно.
```

У нас уже LangGraph. Checkpointer **не используется** — стейт живёт только в RAM.

**Memory-as-a-Service (MaaS)** — паттерн 2025

Память как независимый микросервис с REST API. Агенты не знают о реализации хранилища:
```
Agent A ──┐
Agent B ──┼──→ Memory Service API ──→ Storage backends
Agent C ──┘        (REST/MCP)
```

### 3.3 Выбор backend в production-системах

| Сценарий | Backend | Почему |
|---------|---------|--------|
| Разработка / малый масштаб | SQLite | Простота, нет зависимостей |
| Продакшн, монолит | PostgreSQL + pgvector | ACID, SQL + векторный поиск в одном месте |
| Продакшн, высокая нагрузка | Redis | Скорость (sub-ms), Pub/Sub, vector search |
| Граф-зависимости между сущностями | Neo4j / Kuzu | Связи между сущностями |
| Гибридный (большинство production) | Vector DB + SQLite/Postgres | Семантика + структурированные данные |

---

## 4. Integram как Memory Backend — уникальное преимущество

Пока все проекты выбирают между Qdrant / Chroma / Redis / Postgres, у нас уже есть **собственный облачный стор** с таблицами, REST API, UI и без write-лимита.

| Возможность | mem0 | Integram (как Memory) |
|-------------|------|----------------------|
| Запись фактов | ✅ | ✅ (любая таблица) |
| Namespace по user/agent | ✅ | ✅ (поля-реквизиты) |
| Семантический поиск | ✅ (vector) | ⚠️ Только полнотекстовый пока |
| История изменений | ✅ | ✅ (история статусов) |
| UI для просмотра | ❌ | ✅ (интерфейс Integram) |
| Облачный бэкап | ❌ (self-hosted) | ✅ |
| Граф связей | ✅ (с Neo4j) | ⚠️ Через связанные таблицы |

Integram проигрывает только по **семантическому поиску** — нет эмбеддингов. Компенсируется FAISS на VPS.

---

## 5. Целевая архитектура

### 5.1 Двухуровневая гибридная схема

```
┌─────────────────────────────────────────────────────────────┐
│                    MemoryService (новый класс)               │
│                                                              │
│  HOT tier — VPS-local, быстро, без latency                   │
│  ┌──────────────────────┐  ┌──────────────────────────────┐ │
│  │ LangGraph            │  │ SQLite                       │ │
│  │ Checkpointer         │  │ ├── user_memory (факты)      │ │
│  │ (граф-стейт +        │  │ │   + agent_id namespace     │ │
│  │  вся история runs)   │  │ ├── episodes (новая таблица) │ │
│  └──────────────────────┘  │ └── prefill (адрес/телефон)  │ │
│                             └──────────────────────────────┘ │
│                                                              │
│  COLD tier — облако, бэкап + BI-аналитика                    │
│  ┌──────────────────────────────────────────────────────┐    │
│  │ Integram                                             │    │
│  │ ├── DEV_TASKS, DEV_MEMORY, DEV_ADVICE (DEVBOT)       │    │
│  │ ├── USER_EPISODES (бэкап эпизодов, раз в сутки)      │    │
│  │ └── USER_FACTS (бэкап семантических фактов)          │    │
│  └──────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 AgentContext — что AgentBus передаёт каждому агенту

```python
@dataclass
class AgentContext:
    user_id: int
    # Shared (readonly snapshot, агент не меняет напрямую)
    user_profile: list[str]       # факты из SQLite
    recent_episodes: list[dict]   # последние 5 эпизодов
    crm_snapshot: CrmSnapshot     # текущее состояние CRM
    # Working
    dialog_history: list[dict]    # RAM/Redis с TTL 30 мин
    # Private
    private_data: dict            # namespace по agent_id

@dataclass
class AgentResult:
    response: str
    new_facts: list[str] = field(default_factory=list)  # → SQLite user_memory
    new_episode: dict | None = None                      # → SQLite episodes
    private_updates: dict = field(default_factory=dict)  # → private namespace
```

Агент не знает о технологии хранения. Завтра заменим SQLite на Postgres — агенты не меняются.

### 5.3 Схема таблицы episodes (новая)

```sql
CREATE TABLE episodes (
    id          INTEGER PRIMARY KEY,
    user_id     INTEGER NOT NULL,
    agent_id    TEXT NOT NULL,        -- 'consultant', 'logist', 'inspector' ...
    event_type  TEXT NOT NULL,        -- 'query', 'order', 'inspect', 'status_change'
    summary     TEXT NOT NULL,        -- краткое описание что произошло
    detail      TEXT,                 -- полный контекст (JSON)
    created_at  TEXT DEFAULT (datetime('now'))
);
CREATE INDEX idx_episodes_user ON episodes(user_id, created_at DESC);
```

---

## 6. Приоритетный план реализации

| # | Шаг | Файл | Эффект | Сложность |
|---|-----|------|--------|-----------|
| 1 | Включить LangGraph checkpointer (SQLite) | `src/orchestrator.py` | Эпизодическая память бесплатно | Малая |
| 2 | Добавить `agent_id` в `user_memory` | `src/memory.py` | Namespace-изоляция агентов | Малая |
| 3 | Создать таблицу `episodes` + методы | `src/memory.py` | Агенты помнят прошлые сессии | Средняя |
| 4 | Создать `MemoryService` — единый API | `src/memory_service.py` | Агенты не знают о хранилище | Средняя |
| 5 | Передавать `AgentContext` через оркестратор | `src/orchestrator.py` | Все агенты видят память | Средняя |
| 6 | Async batch-sync → Integram (раз в сутки) | `src/memory_service.py` | Облачный бэкап памяти | Средняя |

---

## 7. Источники

- [mem0ai/mem0](https://github.com/mem0ai/mem0) — Universal memory layer for AI Agents
- [langchain-ai/langmem](https://github.com/langchain-ai/langmem) — LangMem SDK
- [redis/agent-memory-server](https://github.com/redis/agent-memory-server) — Redis Agent Memory Server
- [Letta — Agent Memory blog](https://www.letta.com/blog/agent-memory)
- [Multi-Agent Memory from a Computer Architecture Perspective](https://arxiv.org/html/2603.10062v1) — arxiv 2025
- [Memory-as-a-Service (MaaS)](https://arxiv.org/html/2506.22815v1) — arxiv 2025
- [Intrinsic Memory Agents](https://arxiv.org/abs/2508.08997) — arxiv 2025
- [State of AI Agent Memory 2026](https://mem0.ai/blog/state-of-ai-agent-memory-2026) — Mem0 blog
- [LangMem SDK Launch](https://blog.langchain.com/langmem-sdk-launch/) — LangChain blog
