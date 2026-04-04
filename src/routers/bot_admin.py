"""Admin-команды из bot.py: stats, yt_*, faq, advice, dev, admin-mode, view-switcher."""
import logging
from typing import Optional

import httpx
from aiogram import Router, Bot, types, F
from aiogram.filters import Command

from src.config import DEVBOT_API_URL, DEVBOT_API_KEY
from src.agents.analyst import AnalystAgent
from src.services.auth_service import AuthService
from src.agents.admin_chat import AdminChatAgent
from src.orchestrator import Orchestrator
from src.agents.inspector import InspectorAgent
from src.routers._state import _admin_mode_users, _admin_view_mode
from src.routers.keyboards import (
    BTN_STATS, BTN_ADMIN, BTN_QUEUE, BTN_VIEW_USER, BTN_VIEW_WORK,
    BTN_BACK_ADMIN, BTN_REFRESH,
    build_main_keyboard,
)
from src.routers.worker import _worker_show_queue

logger = logging.getLogger(__name__)
router = Router()

_analyst: Optional[AnalystAgent] = None
_orchestrator: Optional[Orchestrator] = None
_admin_chat_agent: Optional[AdminChatAgent] = None
_inspector: Optional[InspectorAgent] = None
_bot: Optional[Bot] = None
_auth: Optional[AuthService] = None


def setup_bot_admin(
    analyst: AnalystAgent,
    orchestrator: Orchestrator,
    admin_chat_agent: AdminChatAgent,
    inspector: InspectorAgent,
    bot: Bot,
    auth: Optional[AuthService] = None,
) -> None:
    global _analyst, _orchestrator, _admin_chat_agent, _inspector, _bot, _auth
    _analyst = analyst
    _orchestrator = orchestrator
    _admin_chat_agent = admin_chat_agent
    _inspector = inspector
    _bot = bot
    _auth = auth


def _is_admin(user_id: int) -> bool:
    if _auth:
        return _auth.is_admin(user_id)
    from src.config import ADMIN_IDS
    return bool(ADMIN_IDS and user_id in ADMIN_IDS)


# ---------------------------------------------------------------------------
# Кнопки admin-меню
# ---------------------------------------------------------------------------

@router.message(F.text == BTN_STATS)
async def btn_stats(message: types.Message):
    if not _is_admin(message.from_user.id):
        return
    await cmd_stats(message)


@router.message(F.text == BTN_ADMIN)
async def btn_admin_mode(message: types.Message):
    if not _is_admin(message.from_user.id):
        return
    await cmd_admin_mode(message)


@router.message(F.text == BTN_QUEUE)
async def btn_queue(message: types.Message):
    if not _is_admin(message.from_user.id):
        return
    await _worker_show_queue(message.chat.id, message)


@router.message(F.text == BTN_VIEW_USER)
async def btn_view_as_user(message: types.Message):
    """Администратор переключается в вид «Глазами клиента»."""
    uid = message.from_user.id
    if not _is_admin(uid):
        return
    _admin_view_mode[uid] = "user"
    await message.answer(
        "👤 *Вид «Глазами клиента»*\n\nТеперь ты видишь то, что видят покупатели.\nКнопка «🔙 Режим Админа» вернёт тебя обратно.",
        parse_mode="Markdown",
        reply_markup=build_main_keyboard(is_admin=True, view="user"),
    )


@router.message(F.text == BTN_VIEW_WORK)
async def btn_view_as_worker(message: types.Message):
    """Администратор переключается в вид «Глазами работника»."""
    uid = message.from_user.id
    if not _is_admin(uid):
        return
    _admin_view_mode[uid] = "worker"
    await message.answer(
        "👷 *Вид «Глазами работника»*\n\nПоказываю очередь сборки — как её видит работник склада.",
        parse_mode="Markdown",
        reply_markup=build_main_keyboard(is_admin=True, view="worker"),
    )
    await _worker_show_queue(message.chat.id, message)


@router.message(F.text == BTN_BACK_ADMIN)
async def btn_back_to_admin(message: types.Message):
    """Вернуться в полный режим администратора."""
    uid = message.from_user.id
    if not _is_admin(uid):
        return
    _admin_view_mode[uid] = "admin"
    await message.answer(
        "🔙 *Режим Админа*",
        parse_mode="Markdown",
        reply_markup=build_main_keyboard(is_admin=True, view="admin"),
    )


@router.message(F.text == BTN_REFRESH)
@router.message(Command("refresh_crm"))
async def btn_refresh_crm(message: types.Message):
    """Принудительное обновление CRM snapshot (только админ)."""
    import src.routers._state as _state_mod
    if not _is_admin(message.from_user.id):
        return
    snapshot = _state_mod._crm_snapshot
    if not snapshot:
        await message.answer("⚠️ CRM snapshot не инициализирован.")
        return
    msg = await message.answer("🔄 Обновляю снимок CRM...")
    try:
        await snapshot.refresh()
        await msg.edit_text(
            f"✅ Снимок CRM обновлён\n"
            f"📦 Заказов: {len(snapshot.orders)}\n"
            f"👥 Клиентов: {len(snapshot.clients)}\n"
            f"🍯 Товаров: {len(snapshot.products)}\n"
            f"⏱ {snapshot.age_str}"
        )
    except Exception as e:
        await msg.edit_text(f"❌ Ошибка обновления: {e}")


# ---------------------------------------------------------------------------
# /admin — режим «Ассистент пчеловода»
# ---------------------------------------------------------------------------

@router.message(Command("admin"))
async def cmd_admin_mode(message: types.Message):
    """Переключить режим «Ассистент пчеловода» (только ADMIN_IDS)."""
    if not _is_admin(message.from_user.id):
        await message.answer("⛔ Команда доступна только администратору.")
        return

    user_id = message.from_user.id
    if user_id in _admin_mode_users:
        _admin_mode_users.discard(user_id)
        _admin_chat_agent.clear_history(user_id)
        await message.answer(
            "🐝 Режим *Ассистент* выключен.\n"
            "Снова работаю как бот для подписчиков.",
            parse_mode="Markdown",
        )
    else:
        _admin_mode_users.add(user_id)
        await message.answer(
            "🤖 *Режим: Ассистент пчеловода*\n\n"
            "Теперь я твой личный помощник с доступом к CRM.\n"
            "Спрашивай что угодно:\n\n"
            "• _Сколько заказов за этот месяц?_\n"
            "• _Какие товары нужно срочно пополнить?_\n"
            "• _Напиши ответ клиенту на жалобу о задержке_\n"
            "• _Топ продуктов за всё время_\n\n"
            "/admin — выключить режим и очистить историю",
            parse_mode="Markdown",
        )


# ---------------------------------------------------------------------------
# /stats — аналитика продаж
# ---------------------------------------------------------------------------

@router.message(Command("stats"))
async def cmd_stats(message: types.Message):
    """Аналитика продаж — только для ADMIN_IDS."""
    if not _is_admin(message.from_user.id):
        await message.answer("⛔ Команда доступна только администратору.")
        return

    query = (message.text or "").removeprefix("/stats").strip()
    if not query:
        query = "общая статистика"

    await _bot.send_chat_action(message.chat.id, "typing")
    try:
        report = await _analyst.handle_query(query)
        await message.answer(report, parse_mode="Markdown")
    except Exception as e:
        logger.error("Ошибка аналитики: %s", e)
        await message.answer("Не удалось получить статистику. Попробуйте позже.")


# ---------------------------------------------------------------------------
# /yt_check, /yt_update, /yt_comments — YouTube KB
# ---------------------------------------------------------------------------

@router.message(Command("yt_check"))
async def cmd_yt_check(message: types.Message):
    """Проверить новые видео на YouTube-канале."""
    if not _is_admin(message.from_user.id):
        await message.answer("⛔ Команда доступна только администратору.")
        return

    from src import config as _cfg
    if not _cfg.YOUTUBE_API_KEY:
        await message.answer(
            "⚠️ YouTube API недоступен.\n\n"
            "Добавь в `.env`:\n"
            "```\nYOUTUBE_API_KEY=ваш_ключ\n```\n"
            "Получить ключ: console.cloud.google.com → YouTube Data API v3",
            parse_mode="Markdown",
        )
        return

    await _bot.send_chat_action(message.chat.id, "typing")
    from src.youtube_updater import check_new_videos, _get_known_ids
    try:
        all_ids, new_ids = await check_new_videos(_cfg.YOUTUBE_API_KEY, _cfg.YOUTUBE_CHANNEL_HANDLE)
        if not all_ids:
            await message.answer("❌ Не удалось получить список видео. Проверь YOUTUBE_API_KEY.")
            return
        text = (
            f"📺 *Канал {_cfg.YOUTUBE_CHANNEL_HANDLE}*\n\n"
            f"Всего видео: {len(all_ids)}\n"
            f"Субтитров в KB: {len(_get_known_ids())}\n"
            f"*Новых: {len(new_ids)}*"
        )
        if new_ids:
            text += f"\n\nID новых видео:\n" + "\n".join(f"• `{v}`" for v in new_ids[:10])
            text += f"\n\nДля загрузки используй /yt\\_update"
        else:
            text += "\n\n✅ База знаний актуальна — новых видео нет."
        await message.answer(text, parse_mode="Markdown")
    except Exception as e:
        logger.error("yt_check error: %s", e)
        await message.answer(f"Ошибка при проверке: {e}")


@router.message(Command("yt_update"))
async def cmd_yt_update(message: types.Message):
    """Скачать субтитры новых видео и пересобрать базу знаний."""
    if not _is_admin(message.from_user.id):
        await message.answer("⛔ Команда доступна только администратору.")
        return

    from src import config as _cfg
    if not _cfg.YOUTUBE_API_KEY:
        await message.answer("⚠️ YOUTUBE_API_KEY не задан в .env.")
        return

    await message.answer("⏳ Проверяю новые видео и обновляю базу знаний...")
    await _bot.send_chat_action(message.chat.id, "typing")
    from src.youtube_updater import run_update
    try:
        report = await run_update(_cfg.YOUTUBE_API_KEY, _cfg.YOUTUBE_CHANNEL_HANDLE)
        if "пересобрана" in report:
            try:
                _orchestrator.load_kb()
                _inspector.kb = _orchestrator._beebot.kb
                report += f"\n\n🔄 KB перезагружена в боте: {len(_orchestrator._beebot.kb.chunks)} чанков"
            except Exception as e:
                report += f"\n\n⚠️ KB пересобрана, но перезагрузка не удалась: {e}"
        await message.answer(report, parse_mode="Markdown")
    except Exception as e:
        logger.error("yt_update error: %s", e)
        await message.answer(f"Ошибка при обновлении: {e}")


@router.message(Command("yt_comments"))
async def cmd_yt_comments(message: types.Message):
    """Скачать комментарии с ответами автора и добавить в базу знаний."""
    if not _is_admin(message.from_user.id):
        await message.answer("⛔ Команда доступна только администратору.")
        return

    from src import config as _cfg
    if not _cfg.YOUTUBE_API_KEY:
        await message.answer("⚠️ YOUTUBE_API_KEY не задан в .env.")
        return

    await message.answer("⏳ Скачиваю комментарии с ответами автора (~2 мин)...")
    await _bot.send_chat_action(message.chat.id, "typing")

    import asyncio
    from src.youtube_comments import download_all_comments
    from src.youtube_loader import CHANNEL_VIDEO_IDS
    from src.youtube_updater import rebuild_knowledge_base

    try:
        results = await asyncio.to_thread(
            download_all_comments,
            CHANNEL_VIDEO_IDS,
            _cfg.YOUTUBE_API_KEY,
            _cfg.YOUTUBE_CHANNEL_HANDLE,
        )
        videos_with_qa = sum(1 for n in results.values() if n > 0)
        total_pairs = sum(results.values())

        report = (
            f"💬 *Комментарии с ответами автора*\n\n"
            f"Видео обработано: {len(results)}\n"
            f"Видео с Q&A: {videos_with_qa}\n"
            f"Всего Q&A пар: *{total_pairs}*\n\n"
            f"Пересобираю базу знаний..."
        )
        await message.answer(report, parse_mode="Markdown")

        n_chunks = await asyncio.to_thread(rebuild_knowledge_base)

        try:
            _orchestrator.load_kb()
            _inspector.kb = _orchestrator._beebot.kb
            reload_msg = f"🔄 KB перезагружена в боте: {len(_orchestrator._beebot.kb.chunks)} чанков"
        except Exception as e:
            reload_msg = f"⚠️ KB пересобрана, но перезагрузка не удалась: {e}"

        await message.answer(
            f"✅ База знаний пересобрана: *{n_chunks} чанков*\n{reload_msg}",
            parse_mode="Markdown",
        )
    except Exception as e:
        logger.error("yt_comments error: %s", e)
        await message.answer(f"Ошибка: {e}")


# ---------------------------------------------------------------------------
# /faq — топ частых вопросов
# ---------------------------------------------------------------------------

@router.message(Command("faq"))
async def cmd_faq(message: types.Message):
    """Показать топ частых вопросов пользователей."""
    if not _is_admin(message.from_user.id):
        await message.answer("⛔ Команда доступна только администратору.")
        return

    _orchestrator.flush_faq()

    args = (message.text or "").removeprefix("/faq").strip()
    try:
        n = int(args) if args else 20
        n = max(5, min(50, n))
    except ValueError:
        n = 20

    top = _orchestrator.get_top_queries(n)
    if not top:
        await message.answer("📝 FAQ пока пуст — пользователи ещё не задавали вопросов.")
        return

    total = sum(c for _, c in top)
    lines = [f"📝 *Топ-{n} частых вопросов* (всего запросов: {total}):\n"]
    for i, (query, count) in enumerate(top, 1):
        bar = "▪" * min(count, 10)
        lines.append(f"{i}. [{count}] {bar} {query}")

    await message.answer("\n".join(lines), parse_mode="Markdown")


# ---------------------------------------------------------------------------
# /advice — советы пчеловода
# ---------------------------------------------------------------------------

@router.message(Command("advice"))
async def cmd_advice(message: types.Message):
    """Показать загруженные советы пчеловода."""
    if not _is_admin(message.from_user.id):
        await message.answer("⛔ Команда доступна только администратору.")
        return

    items = _orchestrator._ontology.advice_items
    if not items:
        await message.answer(
            "📭 Советы пчеловода не загружены.\n\n"
            "Добавь записи в таблицу 7195 (Советы пчеловода) в Integram — "
            "они подхватятся при следующем перезапуске бота."
        )
        return

    _pri_emoji = {"высокий": "🔴", "средний": "🟡", "справочный": "⚪"}
    lines = [f"🐝 *Советы пчеловода* ({len(items)} активных):\n"]
    for item in items:
        pri = _pri_emoji.get(item["priority"], "⚪")
        cat = f" [{item['category']}]" if item["category"] else ""
        text = item["text"] or item["name"]
        lines.append(f"{pri}{cat} {text[:200]}")

    await message.answer("\n".join(lines), parse_mode="Markdown")


# ---------------------------------------------------------------------------
# /agent_config — управление спецификациями агентов (Фаза 9.5)
# ---------------------------------------------------------------------------

@router.message(Command("agent_config"))
async def cmd_agent_config(message: types.Message):
    """Управление AGENT_SPECS: /agent_config <agent> <field> <value>
    или /agent_config reload — перезагрузить из Integram.

    Пример: /agent_config beebot system_prompt Ты — Александр, пчеловод...
    """
    if not _is_admin(message.from_user.id):
        await message.answer("⛔ Команда доступна только администратору.")
        return

    agent_specs = getattr(_orchestrator, "_agent_specs", None)
    if not agent_specs:
        await message.answer("⚠️ AgentSpecsCache не инициализирован.")
        return

    args = (message.text or "").split(None, 3)[1:]  # убираем /agent_config
    if not args:
        # Показать текущие спецификации
        lines = ["🤖 *Текущие спецификации агентов:*\n"]
        for aid in agent_specs.list_agents():
            spec = agent_specs.get(aid) or {}
            sp = spec.get("system_prompt")
            sp_str = f"`{sp[:60]}...`" if sp and len(sp) > 60 else f"`{sp}`" if sp else "_defaults_"
            lines.append(f"*{aid}*: system_prompt={sp_str}")
        lines.append("\nИсточник: " + ("Integram" if agent_specs.loaded_from_crm else "in-code defaults"))
        await message.answer("\n".join(lines), parse_mode="Markdown")
        return

    if args[0] == "reload":
        await agent_specs.load()
        src = "Integram" if agent_specs.loaded_from_crm else "defaults (таблица не создана)"
        await message.answer(f"✅ AGENT_SPECS перезагружены из {src}.")
        return

    if len(args) < 3:
        await message.answer(
            "Использование: `/agent_config <agent_id> <field> <value>`\n"
            "Поля: system\\_prompt, skills (через запятую), triggers (через запятую), voice\\_style\n"
            "Или: `/agent_config reload`",
            parse_mode="Markdown",
        )
        return

    agent_id, field, value = args[0], args[1], args[2]
    valid_fields = {"system_prompt", "skills", "triggers", "voice_style"}
    if field not in valid_fields:
        await message.answer(f"⚠️ Неизвестное поле «{field}». Допустимые: {', '.join(sorted(valid_fields))}")
        return

    agent_specs.set(agent_id, field, value)
    saved = await agent_specs.update_crm(agent_id)
    crm_note = " и сохранено в Integram" if saved else " (в память; Integram таблица ещё не создана)"
    await message.answer(f"✅ [{agent_id}] {field} обновлён{crm_note}.")


# ---------------------------------------------------------------------------
# /dev — отправить задачу в DEVBOT
# ---------------------------------------------------------------------------

@router.message(Command("dev"))
async def cmd_dev(message: types.Message):
    """Отправить задачу автономному разработчику DEVBOT (hive:8091)."""
    if not _is_admin(message.from_user.id):
        await message.answer("⛔ Команда доступна только администратору.")
        return

    task = (message.text or "").removeprefix("/dev").strip()
    if not task:
        await message.answer(
            "Использование: /dev <описание задачи>\n\n"
            "Пример: /dev добавь кнопку «Повторить заказ» в бот"
        )
        return

    status_msg = await message.answer("📤 Отправляю задачу в DEVBOT...")
    try:
        headers = {"Authorization": f"Bearer {DEVBOT_API_KEY}"} if DEVBOT_API_KEY else {}
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(f"{DEVBOT_API_URL}/task", json={"text": task}, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        if data.get("status") == "ok":
            await status_msg.edit_text(
                f"✅ Задача отправлена в DEVBOT.\n\n"
                f"📋 *{task[:200]}*\n\n"
                f"DEVBOT уведомит тебя в Telegram когда будет готов план.",
                parse_mode="Markdown",
            )
        else:
            err = data.get("error", "неизвестная ошибка")
            await status_msg.edit_text(f"⚠️ DEVBOT ответил: {err}")
    except httpx.ConnectError:
        await status_msg.edit_text(
            "❌ DEVBOT недоступен — проверь SSH-туннель:\n"
            "`ssh -R 8091:localhost:8091 ai-agent@185.233.200.13`",
            parse_mode="Markdown",
        )
    except Exception as e:
        logger.error("cmd_dev error: %s", e)
        await status_msg.edit_text(f"❌ Ошибка: {e}")
