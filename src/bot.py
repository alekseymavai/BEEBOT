"""Telegram bot for BEEBOT — AI assistant for a beekeeper blog."""

import asyncio
import logging

from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import CommandStart, Command
from aiogram.enums import ParseMode, ChatType

from src.config import TELEGRAM_BOT_TOKEN
from src.knowledge_base import KnowledgeBase
from src.llm_client import LLMClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher()

# Initialize knowledge base and LLM client
kb = KnowledgeBase()
llm = LLMClient()

WELCOME_MESSAGE = """Привет! Я бот-помощник Александра Дмитрова — пчеловода и автора блога о продуктах пчеловодства.

Задай мне любой вопрос о:
- Продуктах пчеловодства (мёд, перга, прополис, пыльца)
- Рецептах и дозировках
- Пчеловодстве и здоровье

Просто напиши свой вопрос!"""

HELP_MESSAGE = """Как пользоваться ботом:

В личных сообщениях — просто напиши вопрос.

В группе — обратись ко мне по имени или ответь на моё сообщение:
• @AleksandrDmitrov_BEEBOT чем полезна перга?
• Или reply на моё сообщение с вопросом

Примеры вопросов:
- Как принимать настойку прополиса?
- Чем полезна перга?
- Как укрепить иммунитет ребёнка?

/start — начать
/help — эта справка"""

BOT_USERNAME = "AleksandrDmitrov_BEEBOT"


@dp.message(CommandStart())
async def cmd_start(message: types.Message):
    await message.answer(WELCOME_MESSAGE)


@dp.message(Command("help"))
async def cmd_help(message: types.Message):
    await message.answer(HELP_MESSAGE)


def _should_respond(message: types.Message) -> bool:
    """Check if bot should respond to this message."""
    # Always respond in private chats
    if message.chat.type == ChatType.PRIVATE:
        return True

    # In groups: respond if mentioned or replied to
    text = message.text or ""
    if f"@{BOT_USERNAME}" in text:
        return True
    if message.reply_to_message and message.reply_to_message.from_user:
        if message.reply_to_message.from_user.id == bot.id:
            return True

    return False


@dp.message()
async def handle_question(message: types.Message):
    """Handle user questions: search KB → generate response via Groq."""
    if not _should_respond(message):
        return

    query = (message.text or "").replace(f"@{BOT_USERNAME}", "").strip()
    if len(query) < 3:
        await message.reply("Напиши вопрос подлиннее, чтобы я мог помочь.")
        return

    logger.info(f"Question from {message.from_user.id} in {message.chat.type}: {query}")

    # Show typing indicator
    await bot.send_chat_action(message.chat.id, "typing")

    try:
        # Search knowledge base
        chunks = kb.search(query)
        logger.info(f"Found {len(chunks)} relevant chunks")

        # Generate response
        response = llm.generate(query, chunks)
        await message.reply(response)

    except Exception as e:
        logger.error(f"Error handling question: {e}")
        await message.reply(
            "Извини, что-то пошло не так. Попробуй спросить ещё раз чуть позже."
        )


async def main():
    logger.info("Starting BEEBOT...")

    # Load knowledge base
    try:
        kb.load()
        logger.info(f"Knowledge base loaded: {len(kb.chunks)} chunks")
    except FileNotFoundError:
        logger.error("Knowledge base not found! Run `python -m src.build_kb` first.")
        return

    # Start polling
    logger.info("Bot is running!")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
