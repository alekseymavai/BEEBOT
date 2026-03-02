"""pytest configuration — set required environment variables before any import."""

import os

# Set dummy credentials so module-level initialisation in bot.py doesn't fail
os.environ.setdefault("GROQ_API_KEY", "test-key-placeholder")
os.environ.setdefault("TELEGRAM_BOT_TOKEN", "0000000000:AABBCCDDEEFFaabbccddeeff-placeholder")
