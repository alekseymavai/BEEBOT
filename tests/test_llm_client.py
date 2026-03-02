"""Unit tests for src/llm_client.py — Groq LLM client module."""

from unittest.mock import MagicMock, patch, call
import time

import pytest

from src.llm_client import LLMClient, build_prompt, SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

SAMPLE_CHUNKS = [
    {
        "source": "pdf:Прополис",
        "text": "Настойку прополиса принимают по 20–30 капель в стакане воды.",
    },
    {
        "source": "pdf:Перга",
        "text": "Перга — богатый источник белков, витаминов и минералов.",
    },
]


# ---------------------------------------------------------------------------
# build_prompt tests
# ---------------------------------------------------------------------------

class TestBuildPrompt:
    """Tests for the build_prompt helper function."""

    def test_prompt_contains_query(self):
        """The generated prompt should include the user's question."""
        prompt = build_prompt("Как принимать настойку прополиса?", SAMPLE_CHUNKS)
        assert "Как принимать настойку прополиса?" in prompt

    def test_prompt_contains_chunk_texts(self):
        """The generated prompt should contain all chunk texts."""
        prompt = build_prompt("вопрос", SAMPLE_CHUNKS)
        for chunk in SAMPLE_CHUNKS:
            assert chunk["text"] in prompt

    def test_prompt_contains_sources(self):
        """The generated prompt should reference chunk sources."""
        prompt = build_prompt("вопрос", SAMPLE_CHUNKS)
        for chunk in SAMPLE_CHUNKS:
            assert chunk["source"] in prompt

    def test_prompt_with_empty_chunks(self):
        """build_prompt should not crash with empty chunk list."""
        prompt = build_prompt("вопрос", [])
        assert "вопрос" in prompt
        assert isinstance(prompt, str)

    def test_prompt_separator_present(self):
        """Chunks should be separated with the --- delimiter."""
        prompt = build_prompt("вопрос", SAMPLE_CHUNKS)
        assert "---" in prompt

    def test_prompt_is_string(self):
        """build_prompt must return a string."""
        result = build_prompt("вопрос", SAMPLE_CHUNKS)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# SYSTEM_PROMPT tests
# ---------------------------------------------------------------------------

class TestSystemPrompt:
    """Sanity checks on the SYSTEM_PROMPT constant."""

    def test_system_prompt_is_in_russian(self):
        """System prompt should contain Russian text."""
        russian_chars = set("абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ")
        assert any(c in russian_chars for c in SYSTEM_PROMPT)

    def test_system_prompt_not_empty(self):
        """System prompt must not be empty."""
        assert len(SYSTEM_PROMPT) > 50

    def test_system_prompt_instructs_russian_only(self):
        """System prompt should contain instruction to reply in Russian only."""
        assert "русском" in SYSTEM_PROMPT.lower() or "russian" in SYSTEM_PROMPT.lower()


# ---------------------------------------------------------------------------
# LLMClient tests
# ---------------------------------------------------------------------------

class TestLLMClient:
    """Tests for LLMClient.generate method."""

    def _make_mock_response(self, text: str):
        """Create a mock Groq API response object."""
        choice = MagicMock()
        choice.message.content = text
        response = MagicMock()
        response.choices = [choice]
        return response

    def test_generate_returns_response_text(self):
        """generate() should return the text from the LLM response."""
        expected = "Настойку прополиса нужно принимать по 20–30 капель."

        with patch("src.llm_client.Groq") as mock_groq_cls:
            mock_client = MagicMock()
            mock_groq_cls.return_value = mock_client
            mock_client.chat.completions.create.return_value = self._make_mock_response(expected)

            llm = LLMClient()
            result = llm.generate("Как принимать прополис?", SAMPLE_CHUNKS)

        assert result == expected

    def test_generate_sends_system_and_user_messages(self):
        """generate() should pass both system and user messages to the API."""
        with patch("src.llm_client.Groq") as mock_groq_cls:
            mock_client = MagicMock()
            mock_groq_cls.return_value = mock_client
            mock_client.chat.completions.create.return_value = self._make_mock_response("ответ")

            llm = LLMClient()
            llm.generate("вопрос", SAMPLE_CHUNKS)

        call_kwargs = mock_client.chat.completions.create.call_args
        messages = call_kwargs.kwargs.get("messages") or call_kwargs.args[0] if call_kwargs.args else None
        if messages is None:
            messages = call_kwargs[1].get("messages", call_kwargs[0])

        # Extract messages from however they were passed
        all_kwargs = call_kwargs[1] if call_kwargs[1] else {}
        messages = all_kwargs.get("messages")

        assert messages is not None
        roles = [m["role"] for m in messages]
        assert "system" in roles
        assert "user" in roles

    def test_generate_system_message_uses_system_prompt(self):
        """The system message content must be the global SYSTEM_PROMPT."""
        with patch("src.llm_client.Groq") as mock_groq_cls:
            mock_client = MagicMock()
            mock_groq_cls.return_value = mock_client
            mock_client.chat.completions.create.return_value = self._make_mock_response("ответ")

            llm = LLMClient()
            llm.generate("вопрос", SAMPLE_CHUNKS)

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        messages = call_kwargs["messages"]
        system_msgs = [m for m in messages if m["role"] == "system"]
        assert len(system_msgs) == 1
        assert system_msgs[0]["content"] == SYSTEM_PROMPT

    def test_generate_retries_on_api_failure(self):
        """generate() should retry up to 3 times on API errors."""
        with patch("src.llm_client.Groq") as mock_groq_cls, \
             patch("src.llm_client.time.sleep"):
            mock_client = MagicMock()
            mock_groq_cls.return_value = mock_client
            mock_client.chat.completions.create.side_effect = Exception("API error")

            llm = LLMClient()
            result = llm.generate("вопрос", SAMPLE_CHUNKS)

        assert mock_client.chat.completions.create.call_count == 3
        assert "техническая проблема" in result.lower() or "не могу ответить" in result.lower()

    def test_generate_succeeds_after_one_failure(self):
        """generate() should succeed if the API succeeds on second attempt."""
        expected = "Ответ на второй попытке"

        with patch("src.llm_client.Groq") as mock_groq_cls, \
             patch("src.llm_client.time.sleep"):
            mock_client = MagicMock()
            mock_groq_cls.return_value = mock_client
            mock_client.chat.completions.create.side_effect = [
                Exception("Временная ошибка"),
                self._make_mock_response(expected),
            ]

            llm = LLMClient()
            result = llm.generate("вопрос", SAMPLE_CHUNKS)

        assert result == expected
        assert mock_client.chat.completions.create.call_count == 2

    def test_generate_returns_fallback_after_all_failures(self):
        """generate() should return a Russian fallback message after 3 failures."""
        with patch("src.llm_client.Groq") as mock_groq_cls, \
             patch("src.llm_client.time.sleep"):
            mock_client = MagicMock()
            mock_groq_cls.return_value = mock_client
            mock_client.chat.completions.create.side_effect = Exception("Error")

            llm = LLMClient()
            result = llm.generate("вопрос", [])

        # Fallback must be non-empty and in Russian
        assert isinstance(result, str)
        assert len(result) > 5
        russian_chars = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
        assert any(c in russian_chars for c in result.lower())

    def test_generate_uses_max_response_length(self):
        """generate() should pass max_tokens to the API call."""
        from src.config import MAX_RESPONSE_LENGTH

        with patch("src.llm_client.Groq") as mock_groq_cls:
            mock_client = MagicMock()
            mock_groq_cls.return_value = mock_client
            mock_client.chat.completions.create.return_value = self._make_mock_response("ok")

            llm = LLMClient()
            llm.generate("вопрос", [])

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs.get("max_tokens") == MAX_RESPONSE_LENGTH

    def test_generate_with_empty_chunks(self):
        """generate() should not crash when given an empty context."""
        with patch("src.llm_client.Groq") as mock_groq_cls:
            mock_client = MagicMock()
            mock_groq_cls.return_value = mock_client
            mock_client.chat.completions.create.return_value = self._make_mock_response("ok")

            llm = LLMClient()
            result = llm.generate("вопрос", [])

        assert isinstance(result, str)

    def test_init_passes_groq_base_url_when_set(self):
        """LLMClient should pass base_url to Groq when GROQ_BASE_URL is configured."""
        with patch("src.llm_client.GROQ_BASE_URL", "http://proxy.example.com"), \
             patch("src.llm_client.Groq") as mock_groq_cls:
            mock_groq_cls.return_value = MagicMock()
            LLMClient()

        call_kwargs = mock_groq_cls.call_args[1]
        assert call_kwargs.get("base_url") == "http://proxy.example.com"

    def test_init_no_base_url_when_not_set(self):
        """LLMClient should not pass base_url to Groq when GROQ_BASE_URL is None."""
        with patch("src.llm_client.GROQ_BASE_URL", None), \
             patch("src.llm_client.Groq") as mock_groq_cls:
            mock_groq_cls.return_value = MagicMock()
            LLMClient()

        call_kwargs = mock_groq_cls.call_args[1]
        assert "base_url" not in call_kwargs
