"""Integration tests — verify that modules work together end-to-end."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from src.pdf_loader import process_all_pdfs
from src.knowledge_base import KnowledgeBase
from src.llm_client import build_prompt, LLMClient
from src.bot import handle_question, _should_respond, BOT_USERNAME


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_PDF_TEXTS = {
    "Прополис.pdf": (
        "Прополис — природный антибиотик с мощными антибактериальными свойствами. "
        "Настойку прополиса принимают по 20–30 капель, разбавленных в стакане тёплой воды. "
        "Курс лечения составляет 2–3 недели. "
        "Прополис помогает при простуде, гриппе и воспалительных процессах. "
        "Детям до 3 лет приём не рекомендуется."
    ),
    "Перга.pdf": (
        "Перга — это пчелиный хлеб, богатый белками, витаминами и минеральными веществами. "
        "Принимают по 1 чайной ложке натощак за 30 минут до еды. "
        "Перга укрепляет иммунитет, улучшает пищеварение и борется с анемией. "
        "Противопоказана при аллергии на продукты пчеловодства. "
        "Хранить в холодном и тёмном месте."
    ),
    "Мёд.pdf": (
        "Крем-мёд — это мёд взбитый при температуре 14 градусов до кремообразной консистенции. "
        "Процесс занимает несколько дней непрерывного помешивания. "
        "Крем-мёд не засахаривается и легко намазывается на хлеб. "
        "Используется в кулинарии и косметологии. "
        "Польза та же, что у обычного мёда."
    ),
}


@pytest.fixture
def pdf_dir_with_content(tmp_path):
    """Fixture: create a temp directory with mock PDFs returning known content."""
    for pdf_name in SAMPLE_PDF_TEXTS:
        (tmp_path / pdf_name).write_bytes(b"%PDF-1.4 placeholder")
    return tmp_path


@pytest.fixture
def knowledge_base_from_docs(tmp_path):
    """Fixture: build a KnowledgeBase from SAMPLE_PDF_TEXTS and return it."""
    documents = [
        {"source": f"pdf:{Path(name).stem}", "text": text}
        for name, text in SAMPLE_PDF_TEXTS.items()
    ]
    kb = KnowledgeBase()
    with (
        patch("src.knowledge_base.FAISS_INDEX_PATH", tmp_path / "index.faiss"),
        patch("src.knowledge_base.CHUNKS_PATH", tmp_path / "chunks.json"),
        patch("src.knowledge_base.PROCESSED_DIR", tmp_path),
    ):
        kb.build(documents)
    return kb, tmp_path


# ---------------------------------------------------------------------------
# PDF → KnowledgeBase pipeline
# ---------------------------------------------------------------------------

class TestPdfToKnowledgeBasePipeline:
    """Test that PDF extraction feeds correctly into KnowledgeBase."""

    def test_pdf_docs_feed_into_kb_build(self, tmp_path):
        """Documents from process_all_pdfs should be accepted by KnowledgeBase.build()."""
        for pdf_name, text in SAMPLE_PDF_TEXTS.items():
            mock_page = MagicMock()
            mock_page.extract_text.return_value = text

            with patch("src.pdf_loader.PdfReader") as mock_cls:
                mock_reader = MagicMock()
                mock_reader.pages = [mock_page]
                mock_cls.return_value = mock_reader
                (tmp_path / pdf_name).write_bytes(b"%PDF placeholder")

        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Т" * 200

        with patch("src.pdf_loader.PdfReader") as mock_cls:
            mock_reader = MagicMock()
            mock_reader.pages = [mock_page]
            mock_cls.return_value = mock_reader

            with patch("src.pdf_loader.TEXTS_DIR", tmp_path / "texts"):
                docs = process_all_pdfs(pdf_dir=tmp_path)

        assert len(docs) > 0

        kb = KnowledgeBase()
        with (
            patch("src.knowledge_base.FAISS_INDEX_PATH", tmp_path / "index.faiss"),
            patch("src.knowledge_base.CHUNKS_PATH", tmp_path / "chunks.json"),
            patch("src.knowledge_base.PROCESSED_DIR", tmp_path),
        ):
            n = kb.build(docs)

        assert n > 0

    def test_pdf_extraction_and_search_end_to_end(self, tmp_path):
        """Full pipeline: extract PDFs → build KB → search → get relevant results."""
        documents = [
            {"source": f"pdf:{Path(name).stem}", "text": text}
            for name, text in SAMPLE_PDF_TEXTS.items()
        ]

        kb = KnowledgeBase()
        with (
            patch("src.knowledge_base.FAISS_INDEX_PATH", tmp_path / "index.faiss"),
            patch("src.knowledge_base.CHUNKS_PATH", tmp_path / "chunks.json"),
            patch("src.knowledge_base.PROCESSED_DIR", tmp_path),
        ):
            kb.build(documents)
            results = kb.search("Как принимать настойку прополиса?", top_k=3)

        sources = [r["source"] for r in results]
        assert "pdf:Прополис" in sources

    def test_all_five_test_queries_return_results(self, tmp_path):
        """All test queries from the issue should return at least 1 result."""
        documents = [
            {"source": f"pdf:{Path(name).stem}", "text": text}
            for name, text in SAMPLE_PDF_TEXTS.items()
        ]

        kb = KnowledgeBase()
        with (
            patch("src.knowledge_base.FAISS_INDEX_PATH", tmp_path / "index.faiss"),
            patch("src.knowledge_base.CHUNKS_PATH", tmp_path / "chunks.json"),
            patch("src.knowledge_base.PROCESSED_DIR", tmp_path),
        ):
            kb.build(documents)

            test_queries = [
                "Как принимать настойку прополиса?",
                "Чем полезна перга?",
                "Как сделать крем-мёд?",
            ]

            for query in test_queries:
                results = kb.search(query, top_k=3)
                assert len(results) > 0, f"No results for query: {query!r}"


# ---------------------------------------------------------------------------
# KnowledgeBase → LLMClient pipeline
# ---------------------------------------------------------------------------

class TestKbToLlmPipeline:
    """Test that KB results are correctly passed to LLM for response generation."""

    def test_kb_results_used_in_prompt(self, knowledge_base_from_docs):
        """KB search results should appear in the LLM prompt."""
        kb, _ = knowledge_base_from_docs
        query = "Чем полезна перга?"
        chunks = kb.search(query, top_k=3)
        assert len(chunks) > 0

        prompt = build_prompt(query, chunks)
        assert query in prompt
        # At least one chunk text should appear in the prompt
        assert any(c["text"][:30] in prompt for c in chunks)

    def test_full_pipeline_with_mocked_groq(self, knowledge_base_from_docs):
        """End-to-end: KB search → LLM generate → get response."""
        kb, _ = knowledge_base_from_docs
        query = "Как принимать настойку прополиса?"

        with patch("src.llm_client.Groq") as mock_groq_cls:
            mock_client = MagicMock()
            mock_groq_cls.return_value = mock_client

            expected = "Принимайте настойку прополиса по 20 капель."
            choice = MagicMock()
            choice.message.content = expected
            mock_response = MagicMock()
            mock_response.choices = [choice]
            mock_client.chat.completions.create.return_value = mock_response

            llm = LLMClient()
            chunks = kb.search(query, top_k=3)
            result = llm.generate(query, chunks)

        assert result == expected
        # Verify the API was called
        assert mock_client.chat.completions.create.call_count == 1


# ---------------------------------------------------------------------------
# Bot end-to-end flow
# ---------------------------------------------------------------------------

class TestBotEndToEnd:
    """Tests for the complete bot message handling flow."""

    def _make_message(self, text: str, chat_type: str = "private"):
        message = AsyncMock()
        message.text = text
        message.chat.type = chat_type
        message.chat.id = 100
        message.from_user = MagicMock()
        message.from_user.id = 42
        message.reply_to_message = None
        message.reply = AsyncMock()
        return message

    @pytest.mark.asyncio
    async def test_private_message_gets_reply(self, knowledge_base_from_docs):
        """Private message should always get a reply."""
        kb, _ = knowledge_base_from_docs
        msg = self._make_message("Чем полезна перга?", chat_type="private")

        with (
            patch("src.bot.kb", kb),
            patch("src.bot.llm") as mock_llm,
            patch("src.bot.bot") as mock_bot,
        ):
            mock_llm.generate.return_value = "Перга богата белками и витаминами!"
            mock_bot.send_chat_action = AsyncMock()
            await handle_question(msg)

        msg.reply.assert_called_once()
        reply_text = msg.reply.call_args[0][0]
        assert len(reply_text) > 5

    @pytest.mark.asyncio
    async def test_group_message_without_mention_ignored(self):
        """Group message without @mention should be silently ignored."""
        msg = self._make_message("Просто разговор", chat_type="group")
        await handle_question(msg)
        msg.reply.assert_not_called()

    @pytest.mark.asyncio
    async def test_group_mentioned_message_gets_reply(self, knowledge_base_from_docs):
        """Group message with @mention should get a reply."""
        kb, _ = knowledge_base_from_docs
        msg = self._make_message(
            f"@{BOT_USERNAME} Как принимать прополис?",
            chat_type="group",
        )

        with (
            patch("src.bot.kb", kb),
            patch("src.bot.llm") as mock_llm,
            patch("src.bot.bot") as mock_bot,
        ):
            mock_llm.generate.return_value = "Принимайте по 20 капель."
            mock_bot.send_chat_action = AsyncMock()
            await handle_question(msg)

        msg.reply.assert_called_once()

    @pytest.mark.asyncio
    async def test_short_query_returns_hint(self):
        """A very short query (< 3 chars) should prompt user to write more."""
        msg = self._make_message("??", chat_type="private")

        with patch("src.bot._should_respond", return_value=True):
            await handle_question(msg)

        msg.reply.assert_called_once()
        reply = msg.reply.call_args[0][0]
        assert len(reply) > 10


# ---------------------------------------------------------------------------
# KB persistence tests
# ---------------------------------------------------------------------------

class TestKnowledgeBasePersistence:
    """Verify KB can be saved and reloaded correctly."""

    def test_save_and_reload_produces_same_search_results(self, tmp_path):
        """Search results after reload should match those from the original build."""
        documents = [
            {"source": f"pdf:{Path(name).stem}", "text": text}
            for name, text in SAMPLE_PDF_TEXTS.items()
        ]

        kb1 = KnowledgeBase()
        with (
            patch("src.knowledge_base.FAISS_INDEX_PATH", tmp_path / "index.faiss"),
            patch("src.knowledge_base.CHUNKS_PATH", tmp_path / "chunks.json"),
            patch("src.knowledge_base.PROCESSED_DIR", tmp_path),
        ):
            kb1.build(documents)
            original_results = kb1.search("Как принимать прополис?", top_k=3)

        kb2 = KnowledgeBase()
        with (
            patch("src.knowledge_base.FAISS_INDEX_PATH", tmp_path / "index.faiss"),
            patch("src.knowledge_base.CHUNKS_PATH", tmp_path / "chunks.json"),
            patch("src.knowledge_base.PROCESSED_DIR", tmp_path),
        ):
            kb2.load()
            reloaded_results = kb2.search("Как принимать прополис?", top_k=3)

        assert len(original_results) == len(reloaded_results)
        original_sources = {r["source"] for r in original_results}
        reloaded_sources = {r["source"] for r in reloaded_results}
        assert original_sources == reloaded_sources

    def test_chunks_json_is_valid_json(self, tmp_path):
        """The chunks.json file written by build() should be valid JSON."""
        documents = [
            {"source": "pdf:Test", "text": "Тестовый документ. " * 10},
        ]
        chunks_path = tmp_path / "chunks.json"

        kb = KnowledgeBase()
        with (
            patch("src.knowledge_base.FAISS_INDEX_PATH", tmp_path / "index.faiss"),
            patch("src.knowledge_base.CHUNKS_PATH", chunks_path),
            patch("src.knowledge_base.PROCESSED_DIR", tmp_path),
        ):
            kb.build(documents)

        data = json.loads(chunks_path.read_text(encoding="utf-8"))
        assert isinstance(data, list)
        assert len(data) > 0
