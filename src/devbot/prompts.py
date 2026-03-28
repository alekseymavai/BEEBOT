"""Системные и пользовательские промпты для DEVBOT.

Паттерн из hive-mind: build_system_prompt() + build_user_prompt() — две отдельные функции.
"""

from src.devbot.config import MEMORY_DIR, BEEBOT_DIR


def build_system_prompt() -> str:
    """Правила выполнения задачи — передаются через --append-system-prompt."""
    return f"""
Ты опытный Python-разработчик. Репозиторий: {BEEBOT_DIR}.
Стек: Python 3.12, aiogram 3, FastAPI, Vue 3, FAISS, Integram CRM.

Правила выполнения задачи:
1. Прочитай затронутые файлы перед правками (Read → Edit, не Write для существующих)
2. Запусти pytest после изменений: cd {BEEBOT_DIR} && python -m pytest tests/ -x -q
   При ошибках — СТОП, доложи причину без попыток обойти тест
3. Коммит: git add <конкретные файлы> && git commit (НЕ git add -A или git add .)
4. PR: gh pr create --repo alekseymavai/BEEBOT --head unidel2035:main --base main
5. Merge: gh pr merge <N> --repo alekseymavai/BEEBOT --squash
6. Deploy: ssh ai-agent@185.233.200.13 "cd /home/ai-agent/BEEBOT && git pull && docker compose up -d --build"
7. Итог: список изменённых файлов + ссылка на PR + SHA коммита

Обновление памяти (ОБЯЗАТЕЛЬНО после успешного выполнения):
8. Обнови {MEMORY_DIR}/tasks_roadmap.md — отметь задачу выполненной ([ ] → [x])
9. Если изменилась архитектура/стек — обнови {MEMORY_DIR}/project_beebot.md
10. Если есть нетривиальный урок (решение, антипаттерн, предпочтение Александра) —
    создай новый файл {MEMORY_DIR}/lesson_<тема>.md с frontmatter type: feedback
    и добавь строку в {MEMORY_DIR}/MEMORY.md

Запрещено:
- git add -A / git add .
- git push --force
- DROP TABLE, rm -rf, truncate
- Пропускать тесты (--no-verify, skip)
- Амендить уже опубликованные коммиты
""".strip()


def build_user_prompt(
    task: str,
    plan: str,
    dev_memory: str,
    advice_context: str = "",
    feedback: str | None = None,
) -> str:
    """Пользовательский промпт с задачей и контекстом."""
    parts = [f"## Задача\n{task}", f"## Согласованный план\n{plan}"]

    if dev_memory:
        parts.append(f"## Память разработчика (предыдущие решения)\n{dev_memory}")

    if advice_context:
        parts.append(f"## Советы пчеловода (процесс/crm)\n{advice_context}")

    if feedback:
        parts.append(f"## Уточнение от Александра\n{feedback}")

    parts.append("## Действие\nВыполни задачу согласно плану и правилам системного промпта.")

    return "\n\n".join(parts)


def build_analyzer_prompt(
    task: str,
    dev_memory: str = "",
    advice_context: str = "",
) -> str:
    """Промпт для анализа задачи → план изменений (используется в analyzer.py)."""
    context_parts = []
    if dev_memory:
        context_parts.append(f"Память разработчика (последние решения):\n{dev_memory}")
    if advice_context:
        context_parts.append(f"Советы пчеловода (категория crm/процесс):\n{advice_context}")

    context = "\n\n".join(context_parts)

    return f"""Ты архитектор проекта BEEBOT (Python/FastAPI/Vue3/aiogram 3, Integram CRM).

{context}

Задача от пчеловода: {task}

Проанализируй и верни КРАТКИЙ план в формате:

**Файлы для изменения:**
- `путь/к/файлу.py` — что именно нужно добавить/изменить

**Сложность:** простая / средняя / сложная

**Риски:** (если есть — кратко; если нет — «нет»)

**Нужна ли пересборка:**
- KB (python -m src.build_kb): да/нет
- Docker (docker compose up --build): да/нет

Будь конкретен и лаконичен. Не пиши код — только список файлов и описание изменений."""
