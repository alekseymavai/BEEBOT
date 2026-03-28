"""Двухуровневая память DEVBOT.

Уровень 1: файлы Claude Code (/home/new/.claude/projects/.../memory/)
  — обновляет сам Claude Code по инструкции в build_system_prompt() (пп. 8-10)

Уровень 2: Integram bibot (облако)
  — обновляет этот модуль после успешного deploy
"""

import logging
from typing import Any

import httpx

from src.devbot.config import (
    INTEGRAM_URL, INTEGRAM_DB, INTEGRAM_LOGIN, INTEGRAM_PASSWORD,
    TABLE_DEV_TASKS, TABLE_DEV_MEMORY, TABLE_DEV_ADVICE,
)
from src.crm_constants import (
    REQ_ADVICE_TEXT, REQ_ADVICE_CATEGORY, REQ_ADVICE_PRIORITY, REQ_ADVICE_STATUS,
    REQ_TASK_DESC, REQ_TASK_STATUS, REQ_TASK_PRIORITY, REQ_TASK_FILES,
    REQ_TASK_PR, REQ_TASK_COMMIT, REQ_TASK_LESSONS,
    REQ_MEM_CONTEXT, REQ_MEM_SOLUTION, REQ_MEM_FILES, REQ_MEM_PR,
    REQ_MEM_ANTIPATTERN, REQ_MEM_CATEGORY,
)

logger = logging.getLogger(__name__)

_BASE = (INTEGRAM_URL or "https://ai2o.ru").rstrip("/")
_DB = INTEGRAM_DB or "bibot"


class DevMemory:
    """Работа с Integram: задачи разработки + память разработчика."""

    def __init__(self) -> None:
        self._token: str | None = None
        self._xsrf: str | None = None
        self._http: httpx.AsyncClient | None = None

    async def _client(self) -> httpx.AsyncClient:
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(base_url=_BASE, timeout=30.0)
        return self._http

    async def close(self) -> None:
        if self._http and not self._http.is_closed:
            await self._http.aclose()

    async def _auth(self) -> None:
        http = await self._client()
        resp = await http.post(
            f"/{_DB}/auth?JSON",
            data={"login": INTEGRAM_LOGIN, "pwd": INTEGRAM_PASSWORD},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        resp.raise_for_status()
        data = resp.json()
        self._token = data.get("token")
        xsrf_resp = await http.get(f"/{_DB}/xsrf?JSON", cookies={_DB: self._token or ""})
        if xsrf_resp.status_code == 200:
            xd = xsrf_resp.json()
            self._xsrf = xd.get("_xsrf", self._xsrf)
            self._token = xd.get("token", self._token)

    async def _post(self, url: str, data: dict) -> dict:
        if not self._token:
            await self._auth()
        http = await self._client()
        data["_xsrf"] = self._xsrf or ""
        resp = await http.post(
            url,
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            cookies={_DB: self._token or ""},
        )
        if resp.status_code in (401, 403):
            await self._auth()
            data["_xsrf"] = self._xsrf or ""
            resp = await http.post(
                url,
                data=data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                cookies={_DB: self._token or ""},
            )
        resp.raise_for_status()
        return resp.json()

    async def _get_all(self, table_id: int) -> list[dict[str, Any]]:
        """Получить все записи таблицы (все страницы)."""
        if not self._token:
            await self._auth()
        http = await self._client()
        items = []
        pg = 1
        while True:
            resp = await http.get(
                f"/{_DB}/object/{table_id}/?JSON&F_U=1&pg={pg}",
                cookies={_DB: self._token or ""},
            )
            resp.raise_for_status()
            data = resp.json()
            objects = data.get("object", [])
            if not objects:
                break
            reqs_map = data.get("&object_reqs", {})
            head = next(
                (v for k, v in data.items() if "&uni_obj_head" in k and "filter" not in k),
                {},
            )
            req_ids = head.get("typ", [])
            for obj in objects:
                obj_id = str(obj.get("id", ""))
                raw = reqs_map.get(obj_id, [])
                reqs = {req_ids[i]: raw[i] for i in range(min(len(req_ids), len(raw)))}
                items.append({"id": int(obj_id) if obj_id.isdigit() else 0, "val": obj.get("val", ""), "reqs": reqs})
            if len(objects) < 20:
                break
            pg += 1
        return items

    # ------------------------------------------------------------------
    # Таблица «Задачи разработки»
    # ------------------------------------------------------------------

    async def create_task(self, description: str) -> int:
        """Создать задачу разработки. Возвращает ID."""
        if not TABLE_DEV_TASKS:
            logger.warning("TABLE_DEV_TASKS не настроен, пропускаем запись в Integram")
            return 0
        # Название = первые 100 символов описания
        name = description[:100]
        data = await self._post(
            f"/{_DB}/_m_new/{TABLE_DEV_TASKS}?JSON",
            {f"t{TABLE_DEV_TASKS}": name, "up": "1"},
        )
        obj_id = int(data.get("obj") or data.get("id") or 0)
        if obj_id:
            # Записать полное описание и статус в реквизиты
            await self._post(
                f"/{_DB}/_m_save/{obj_id}?JSON",
                {
                    f"t{TABLE_DEV_TASKS}": name,
                    f"t{REQ_TASK_DESC}": description[:2000],
                    f"t{REQ_TASK_STATUS}": "новая",
                    f"t{REQ_TASK_PRIORITY}": "обычный",
                },
            )
        logger.info("DevMemory: создана задача id=%d", obj_id)
        return obj_id

    async def update_task(
        self,
        task_id: int,
        status: str,
        pr_url: str = "",
        files: str = "",
        sha: str = "",
        lessons: str = "",
    ) -> None:
        """Обновить задачу после выполнения."""
        if not TABLE_DEV_TASKS or not task_id:
            return
        reqs: dict[str, str] = {f"t{REQ_TASK_STATUS}": status}
        if pr_url:
            reqs[f"t{REQ_TASK_PR}"] = pr_url[:500]
        if files:
            reqs[f"t{REQ_TASK_FILES}"] = files[:500]
        if sha:
            reqs[f"t{REQ_TASK_COMMIT}"] = sha[:50]
        if lessons:
            reqs[f"t{REQ_TASK_LESSONS}"] = lessons[:2000]
        await self._post(f"/{_DB}/_m_save/{task_id}?JSON", reqs)
        logger.info("DevMemory: обновлена задача id=%d → %s", task_id, status)

    # ------------------------------------------------------------------
    # Таблица «Память разработчика»
    # ------------------------------------------------------------------

    async def add_dev_memory(
        self,
        topic: str,
        context: str,
        solution: str,
        files: str = "",
        pr_url: str = "",
        lessons: str = "",
        category: str = "api",
    ) -> int:
        """Добавить запись в память разработчика."""
        if not TABLE_DEV_MEMORY:
            logger.warning("TABLE_DEV_MEMORY не настроен, пропускаем")
            return 0
        name = f"[{category}] {topic[:80]}"
        data = await self._post(
            f"/{_DB}/_m_new/{TABLE_DEV_MEMORY}?JSON",
            {f"t{TABLE_DEV_MEMORY}": name, "up": "1"},
        )
        obj_id = int(data.get("obj") or data.get("id") or 0)
        if obj_id:
            reqs: dict[str, str] = {f"t{TABLE_DEV_MEMORY}": name}
            if context:
                reqs[f"t{REQ_MEM_CONTEXT}"] = context[:2000]
            if solution:
                reqs[f"t{REQ_MEM_SOLUTION}"] = solution[:2000]
            if files:
                reqs[f"t{REQ_MEM_FILES}"] = files[:500]
            if pr_url:
                reqs[f"t{REQ_MEM_PR}"] = pr_url[:200]
            if lessons:
                reqs[f"t{REQ_MEM_ANTIPATTERN}"] = lessons[:2000]
            reqs[f"t{REQ_MEM_CATEGORY}"] = category
            await self._post(f"/{_DB}/_m_save/{obj_id}?JSON", reqs)
        logger.info("DevMemory: записана память id=%d topic=%s", obj_id, topic[:40])
        return obj_id

    # ------------------------------------------------------------------
    # Таблица «Советы пчеловода»
    # ------------------------------------------------------------------

    async def get_advice(self, categories: list[str] | None = None) -> list[dict]:
        """Получить советы пчеловода. categories=['crm','процесс'] etc."""
        if not TABLE_DEV_ADVICE:
            return []
        try:
            items = await self._get_all(TABLE_DEV_ADVICE)
            if categories:
                items = [
                    i for i in items
                    if any(cat in (i.get("val") or "") for cat in categories)
                ]
            return items
        except Exception as e:
            logger.warning("DevMemory.get_advice error: %s", e)
            return []

    async def get_recent_dev_memory(self, limit: int = 10) -> str:
        """Получить последние N записей памяти разработчика в виде текста."""
        if not TABLE_DEV_MEMORY:
            return ""
        try:
            items = await self._get_all(TABLE_DEV_MEMORY)
            recent = items[-limit:] if len(items) > limit else items
            lines = [f"- {i['val']}" for i in recent if i.get("val")]
            return "\n".join(lines)
        except Exception as e:
            logger.warning("DevMemory.get_recent_dev_memory error: %s", e)
            return ""

    # ------------------------------------------------------------------
    # Полный цикл: record_completion
    # ------------------------------------------------------------------

    async def record_completion(
        self,
        task_id: int,
        task_text: str,
        plan: str,
        pr_url: str = "",
        sha: str = "",
        files_changed: str = "",
        lessons: str = "",
    ) -> None:
        """Записать успешное выполнение задачи (Уровень 2 памяти).

        Уровень 1 (файлы памяти) обновляет сам Claude Code по build_system_prompt() пп. 8-10.
        """
        # Обновить статус задачи
        await self.update_task(task_id, "готово", pr_url, files_changed, sha, lessons)

        # Добавить в память разработчика
        topic = task_text[:60]
        category = _guess_category(task_text + plan)
        await self.add_dev_memory(
            topic=topic,
            context=task_text,
            solution=plan,
            files=files_changed,
            pr_url=pr_url,
            lessons=lessons,
            category=category,
        )


def _guess_category(text: str) -> str:
    """Угадать категорию памяти по тексту задачи."""
    text_l = text.lower()
    if any(w in text_l for w in ("vue", "frontend", "страниц", "ui", "компонент")):
        return "frontend"
    if any(w in text_l for w in ("integram", "crm", "заказ", "клиент", "товар")):
        return "crm"
    if any(w in text_l for w in ("docker", "vps", "деплой", "nginx", "deploy", "systemd")):
        return "infra"
    if any(w in text_l for w in ("kb", "база знаний", "faiss", "embedding", "чанк")):
        return "kb"
    if any(w in text_l for w in ("api", "fastapi", "router", "endpoint")):
        return "api"
    return "model"


# Singleton
dev_memory = DevMemory()
