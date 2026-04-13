"""Роутер управления PDF-инструкциями через Telegram.

Команды (только ADMIN):
  /docs            — показать список инструкций с кнопками «Скачать DOCX»
  [кнопка]         — отправить DOCX конкретной инструкции
  [приём .docx]    — FSM замены: выбрать PDF → подтвердить → сохранить
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Optional

from aiogram import Router, Bot, F, types
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    FSInputFile,
)

from src.services.docs_service import DocsService

logger = logging.getLogger(__name__)
router = Router()

_docs_svc: Optional[DocsService] = None
_bot: Optional[Bot] = None
_admin_ids: list[int] = []


def setup_docs(docs_svc: DocsService, bot: Bot, admin_ids: list[int]) -> None:
    global _docs_svc, _bot, _admin_ids
    _docs_svc = docs_svc
    _bot = bot
    _admin_ids = admin_ids


class DocsUploadFSM(StatesGroup):
    waiting_for_target = State()
    waiting_for_confirm = State()


def _is_admin(user_id: int) -> bool:
    return user_id in _admin_ids


def _build_docs_keyboard() -> InlineKeyboardMarkup:
    buttons = [
        [InlineKeyboardButton(
            text=f"📄 {name}",
            callback_data=f"docs:get:{name}",
        )]
        for name in (_docs_svc.list_pdfs() if _docs_svc else [])
    ]
    return InlineKeyboardMarkup(inline_keyboard=buttons)


@router.message(Command("docs"))
async def cmd_docs(message: types.Message) -> None:
    if not _is_admin(message.from_user.id):
        return
    names = _docs_svc.list_pdfs()
    if not names:
        await message.answer("Инструкции не найдены в data/pdfs/")
        return
    await message.answer(
        f"📚 <b>Инструкции</b> ({len(names)} шт.)\n\nНажмите чтобы скачать как DOCX:",
        reply_markup=_build_docs_keyboard(),
        parse_mode="HTML",
    )


@router.callback_query(F.data.startswith("docs:get:"))
async def cb_get_docx(callback: types.CallbackQuery) -> None:
    if not _is_admin(callback.from_user.id):
        await callback.answer("Нет доступа")
        return

    name = callback.data.removeprefix("docs:get:")
    await callback.answer("Конвертирую...")
    await callback.message.answer(f"⏳ Конвертирую «{name}» в DOCX...")

    try:
        docx_path = _docs_svc.pdf_to_docx(name)
    except FileNotFoundError:
        await callback.message.answer(f"❌ Файл «{name}.pdf» не найден")
        return
    except Exception as e:
        logger.exception("Ошибка конвертации PDF→DOCX: %s", e)
        await callback.message.answer(f"❌ Ошибка конвертации: {e}")
        return

    await callback.message.answer_document(
        FSInputFile(str(docx_path), filename=f"{name}.docx"),
        caption=(
            f"📝 <b>{name}</b>\n\n"
            "Отредактируйте и отправьте DOCX обратно — "
            "сохраню в PDF и обновлю базу знаний."
        ),
        parse_mode="HTML",
    )


DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


@router.message(F.document.mime_type == DOCX_MIME)
async def receive_docx(message: types.Message, state: FSMContext) -> None:
    if not _is_admin(message.from_user.id):
        return

    names = _docs_svc.list_pdfs()
    if not names:
        await message.answer("❌ Нет PDF-инструкций для замены")
        return

    await state.update_data(
        docx_file_id=message.document.file_id,
        docx_filename=message.document.file_name or "document.docx",
    )
    await state.set_state(DocsUploadFSM.waiting_for_target)

    stem = Path(message.document.file_name or "").stem
    exact_match = stem if stem in names else None

    if exact_match:
        await state.update_data(target_name=exact_match)
        await state.set_state(DocsUploadFSM.waiting_for_confirm)
        kb = InlineKeyboardMarkup(inline_keyboard=[[
            InlineKeyboardButton(text="✅ Да, заменить", callback_data="docs:confirm:yes"),
            InlineKeyboardButton(text="❌ Отмена", callback_data="docs:confirm:no"),
        ]])
        await message.answer(
            f"Заменить инструкцию <b>«{exact_match}»</b>?",
            reply_markup=kb,
            parse_mode="HTML",
        )
    else:
        buttons = [
            [InlineKeyboardButton(text=n, callback_data=f"docs:target:{n}")]
            for n in names
        ]
        buttons.append([InlineKeyboardButton(text="❌ Отмена", callback_data="docs:confirm:no")])
        await message.answer(
            "Какую инструкцию заменить?",
            reply_markup=InlineKeyboardMarkup(inline_keyboard=buttons),
        )


@router.callback_query(DocsUploadFSM.waiting_for_target, F.data.startswith("docs:target:"))
async def cb_select_target(callback: types.CallbackQuery, state: FSMContext) -> None:
    target = callback.data.removeprefix("docs:target:")
    await state.update_data(target_name=target)
    await state.set_state(DocsUploadFSM.waiting_for_confirm)
    kb = InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="✅ Да, заменить", callback_data="docs:confirm:yes"),
        InlineKeyboardButton(text="❌ Отмена", callback_data="docs:confirm:no"),
    ]])
    await callback.message.edit_text(
        f"Заменить инструкцию <b>«{target}»</b>?",
        reply_markup=kb,
        parse_mode="HTML",
    )
    await callback.answer()


@router.callback_query(DocsUploadFSM.waiting_for_confirm, F.data == "docs:confirm:no")
async def cb_cancel(callback: types.CallbackQuery, state: FSMContext) -> None:
    await state.clear()
    await callback.message.edit_text("Отменено.")
    await callback.answer()


@router.callback_query(DocsUploadFSM.waiting_for_confirm, F.data == "docs:confirm:yes")
async def cb_confirm_save(callback: types.CallbackQuery, state: FSMContext) -> None:
    data = await state.get_data()
    await state.clear()
    await callback.message.edit_text("⏳ Конвертирую и сохраняю...")
    await callback.answer()

    target_name: str = data["target_name"]
    file_id: str = data["docx_file_id"]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_docx = Path(tmpdir) / f"{target_name}.docx"
        await _bot.download(file_id, destination=str(tmp_docx))

        try:
            _docs_svc.docx_to_pdf(tmp_docx, dest_name=target_name)
        except Exception as e:
            logger.exception("Ошибка DOCX→PDF: %s", e)
            await callback.message.answer(f"❌ Ошибка конвертации: {e}")
            return

    await callback.message.answer(
        f"✅ Инструкция <b>«{target_name}»</b> сохранена.\n⏳ Обновляю базу знаний...",
        parse_mode="HTML",
    )

    try:
        await _docs_svc.rebuild_kb()
        await callback.message.answer("✅ База знаний обновлена!")
    except Exception as e:
        logger.exception("Ошибка пересборки KB: %s", e)
        await callback.message.answer(f"⚠️ PDF сохранён, но KB не обновлена: {e}")
