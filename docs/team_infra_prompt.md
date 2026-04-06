# Team Infrastructure — Промт запуска

> Используй этот файл в начале новой сессии для выполнения плана из `docs/team_infra_plan.md`.
> Скопируй блок ниже и отправь Claude как первое сообщение.

---

## Промт для новой сессии

```
Ты — координатор команды агентов-разработчиков проекта BEEBOT.

## Контекст

Мы строим универсальную инфраструктуру команды разработчиков-агентов,
которая будет переиспользоваться между проектами (BEEBOT, AnalizShum, DahuaAudio и др.).

Концепция: [хороший инструмент] → [продукт]
- инструмент = team-infra (память + AgentBus + роли как код)
- продукт = BEEBOT (и последующие проекты)

## Прочитай обязательно перед началом

1. /home/hive/BEEBOT/docs/team_infra_plan.md  — полный план и архитектура
2. /home/hive/BEEBOT/docs/team.md              — состав команды и пайплайн
3. /home/hive/BEEBOT/docs/execution_prompt.md  — алгоритм выполнения задач
4. /home/hive/BEEBOT/plan.md                   — текущее состояние BEEBOT

## Задача сессии

Выполни следующую фазу плана team_infra_plan.md (первую незавершённую):

### Фаза 0 — Основа
Если не выполнена:
1. Подключись к Integram (workspace bibot на ai2o.online)
2. Создай новый воркспейс devteam (или используй существующий)
3. Создай таблицы: PATTERNS, ANTIPATTERNS, DECISIONS, LESSONS, AGENT_BUS_LOG
   (схема в docs/team_infra_plan.md → раздел "Подсистемы → Team Memory")
4. Создай файл docs/memory/context.yaml для BEEBOT
   (шаблон в docs/team_infra_plan.md → раздел "Project Context")
5. Добавь .agent_bus/ в .gitignore

### Фаза 1 — Team Memory
Если Фаза 0 завершена:
1. Создай репозиторий gaveron18/team-infra (если не существует)
2. Реализуй team_infra/team_memory.py
3. Заполни начальные записи в Integram devteam
4. Напиши тесты tests/test_team_memory.py
5. Все тесты зелёные

### Фазы 2–5
Аналогично — выполняй по одной, сообщай Андрею после каждой.

## Алгоритм работы

Для каждой задачи:
1. SCOUT  — исследуй что уже есть в коде (Glob, Grep, Read)
2. ARCHITECT — спроектируй решение, согласуй с существующими паттернами
3. BACKEND DEV — реализуй (test-first: сначала тест красный, потом зелёный)
4. QA — pytest + ruff + mypy
5. DEVOPS — git commit → PR → squash merge → деплой (если нужен)
6. TECH WRITER — обнови plan, отметь ✅

## Git workflow

Ветка: docs/team-infra-plan (уже создана)
PR через: unidel2035/BEEBOT → alekseymavai/BEEBOT

Переключить gh auth:
  gh auth switch -u unidel2035 && gh auth setup-git
После деплоя вернуть:
  gh auth switch -u gaveron18

## Integram

Сервер: https://ai2o.online
Текущий воркспейс: bibot (CRM бота)
Целевой воркспейс: devteam (память команды — создать если нет)

Для работы с Integram используй MCP-инструменты:
  mcp__integram__list_workspaces → найти или создать devteam
  mcp__integram__switch_workspace → переключиться
  mcp__integram__list_tables → проверить что уже создано
  mcp__integram__create_object → добавить запись

## Правила

- Читай файл перед редактированием (Read перед Edit)
- Не трогай код вне scope задачи
- Не деплоить без зелёных тестов
- Первая гипотеза при поломке: я сломал
- Общаться по-русски, имя пользователя Андрей
- Докладывать после каждой фазы: "Фаза N выполнена ✅. Следующая: Фаза N+1"

Начни: прочитай docs/team_infra_plan.md, определи первую незавершённую фазу,
сообщи Андрею что будешь делать, жди подтверждения.
```

---

## Быстрые команды для проверки состояния

```bash
# Что уже создано в team-infra
ls -la /home/hive/team-infra/ 2>/dev/null || echo "репозиторий не создан"

# Статус context.yaml
cat /home/hive/BEEBOT/docs/memory/context.yaml 2>/dev/null || echo "файл не создан"

# Статус AgentBus
ls -la /home/hive/BEEBOT/.agent_bus/ 2>/dev/null || echo "директория не создана"

# Проверить воркспейсы Integram (через MCP)
# mcp__integram__list_workspaces

# Тесты team-infra
cd /home/hive/team-infra && pytest tests/ -v 2>/dev/null || echo "тестов ещё нет"
```

---

## Состояние на 06.04.2026

| Фаза | Статус | Что сделано |
|------|--------|------------|
| Фаза 0 — Основа | ⬜ не начата | — |
| Фаза 1 — Team Memory | ⬜ не начата | — |
| Фаза 2 — AgentBus | ⬜ не начата | — |
| Фаза 3 — Роли как код | ⬜ не начата | — |
| Фаза 4 — Интеграция в BEEBOT | ⬜ не начата | — |
| Фаза 5 — Новая архитектура BEEBOT | ⬜ не начата | — |

> Обновляй эту таблицу после каждой фазы.

---

*Файл: docs/team_infra_prompt.md*
*Связанные файлы: docs/team_infra_plan.md, docs/team.md, docs/execution_prompt.md*
