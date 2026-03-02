"""Unit tests for src/knowledge_base.py — FAISS knowledge base module."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

from src.knowledge_base import StyleAnalyzer, KnowledgeBase


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SAMPLE_DOCUMENTS = [
    {
        "source": "pdf:Прополис",
        "text": (
            "Прополис — природный антибиотик. "
            "Настойку принимают по 20–30 капель в стакане воды. "
            "Курс лечения 2–3 недели. "
            "Усиливает иммунитет и борется с вирусами."
        ),
    },
    {
        "source": "pdf:Перга",
        "text": (
            "Перга — пчелиный хлеб, богатый белками и витаминами. "
            "Принимают по 1 чайной ложке натощак. "
            "Помогает при усталости и анемии. "
            "Улучшает работу кишечника."
        ),
    },
    {
        "source": "pdf:Мёд",
        "text": (
            "Крем-мёд делается из жидкого мёда путём взбивания. "
            "Температура должна быть 14 градусов. "
            "Хранится дольше обычного мёда. "
            "Используется в кулинарии."
        ),
    },
    {
        "source": "pdf:ПЖВМ",
        "text": (
            "ПЖВМ — подмор живых восковых молей. "
            "Применяется для укрепления здоровья. "
            "Настойка на спирту 40 градусов. "
            "Дозировка: 20 капель 2 раза в день."
        ),
    },
    {
        "source": "pdf:Зима",
        "text": (
            "Подготовка пчёл к зиме начинается в августе. "
            "Нужно убедиться, что мёда достаточно на зимовку. "
            "Ульи утепляют и закрывают летки. "
            "Пчёлы формируют клуб при температуре ниже +8."
        ),
    },
]


# ---------------------------------------------------------------------------
# StyleAnalyzer tests
# ---------------------------------------------------------------------------

class TestStyleAnalyzer:
    """Tests for StyleAnalyzer feature extractor."""

    def setup_method(self):
        self.analyzer = StyleAnalyzer()

    def test_extract_features_returns_dict(self):
        """extract_features should return a dict with all expected keys."""
        text = "Привет! Как дела? Всё хорошо, друг мой."
        features = self.analyzer.extract_features(text)
        assert isinstance(features, dict)
        expected_keys = {
            "avg_sentence_len", "avg_word_len",
            "exclamation_ratio", "question_ratio", "comma_ratio"
        }
        assert set(features.keys()) == expected_keys

    def test_extract_features_all_float(self):
        """All feature values must be floats."""
        features = self.analyzer.extract_features("Тест теста тестового.")
        for v in features.values():
            assert isinstance(v, float)

    def test_to_vector_returns_numpy_array(self):
        """to_vector should return a 1-D numpy float32 array of length 5."""
        vec = self.analyzer.to_vector("Тест текста для проверки.")
        assert isinstance(vec, np.ndarray)
        assert vec.dtype == np.float32
        assert vec.shape == (5,)

    def test_extract_features_empty_string(self):
        """Should handle empty string without exceptions."""
        features = self.analyzer.extract_features("")
        for v in features.values():
            assert v == 0.0

    def test_exclamation_ratio_nonzero_for_exclamations(self):
        """Exclamation ratio should be > 0 for text with ! marks."""
        features = self.analyzer.extract_features("Отлично! Прекрасно! Замечательно!")
        assert features["exclamation_ratio"] > 0

    def test_question_ratio_nonzero_for_questions(self):
        """Question ratio should be > 0 for text with ? marks."""
        features = self.analyzer.extract_features("Как дела? Всё хорошо?")
        assert features["question_ratio"] > 0

    def test_comma_ratio_nonzero_for_commas(self):
        """Comma ratio should be > 0 for text with commas."""
        features = self.analyzer.extract_features("Мёд, прополис, перга — продукты пчеловодства.")
        assert features["comma_ratio"] > 0


# ---------------------------------------------------------------------------
# KnowledgeBase tests
# ---------------------------------------------------------------------------

class TestKnowledgeBase:
    """Tests for KnowledgeBase (build / search / save / load)."""

    def _make_kb(self):
        return KnowledgeBase()

    def test_build_returns_chunk_count(self, tmp_path):
        """build() should return the number of chunks created."""
        kb = self._make_kb()
        with (
            patch("src.knowledge_base.FAISS_INDEX_PATH", tmp_path / "index.faiss"),
            patch("src.knowledge_base.CHUNKS_PATH", tmp_path / "chunks.json"),
            patch("src.knowledge_base.PROCESSED_DIR", tmp_path),
        ):
            count = kb.build(SAMPLE_DOCUMENTS)
        assert isinstance(count, int)
        assert count > 0

    def test_build_populates_chunks(self, tmp_path):
        """After build(), kb.chunks should be non-empty and contain expected fields."""
        kb = self._make_kb()
        with (
            patch("src.knowledge_base.FAISS_INDEX_PATH", tmp_path / "index.faiss"),
            patch("src.knowledge_base.CHUNKS_PATH", tmp_path / "chunks.json"),
            patch("src.knowledge_base.PROCESSED_DIR", tmp_path),
        ):
            kb.build(SAMPLE_DOCUMENTS)

        assert len(kb.chunks) > 0
        for chunk in kb.chunks:
            assert "text" in chunk
            assert "source" in chunk
            assert "chunk_index" in chunk

    def test_build_creates_faiss_index_file(self, tmp_path):
        """build() should create an index.faiss file on disk."""
        import faiss

        kb = self._make_kb()
        index_path = tmp_path / "index.faiss"
        with (
            patch("src.knowledge_base.FAISS_INDEX_PATH", index_path),
            patch("src.knowledge_base.CHUNKS_PATH", tmp_path / "chunks.json"),
            patch("src.knowledge_base.PROCESSED_DIR", tmp_path),
        ):
            kb.build(SAMPLE_DOCUMENTS)

        assert index_path.exists()

    def test_build_creates_chunks_json_file(self, tmp_path):
        """build() should create a chunks.json file on disk."""
        kb = self._make_kb()
        chunks_path = tmp_path / "chunks.json"
        with (
            patch("src.knowledge_base.FAISS_INDEX_PATH", tmp_path / "index.faiss"),
            patch("src.knowledge_base.CHUNKS_PATH", chunks_path),
            patch("src.knowledge_base.PROCESSED_DIR", tmp_path),
        ):
            kb.build(SAMPLE_DOCUMENTS)

        assert chunks_path.exists()
        data = json.loads(chunks_path.read_text(encoding="utf-8"))
        assert isinstance(data, list)
        assert len(data) > 0

    def test_build_raises_on_empty_documents(self, tmp_path):
        """build() should raise ValueError when given no documents."""
        kb = self._make_kb()
        with (
            patch("src.knowledge_base.FAISS_INDEX_PATH", tmp_path / "index.faiss"),
            patch("src.knowledge_base.CHUNKS_PATH", tmp_path / "chunks.json"),
            patch("src.knowledge_base.PROCESSED_DIR", tmp_path),
        ):
            with pytest.raises(ValueError, match="No chunks"):
                kb.build([])

    def test_search_returns_list_of_dicts(self, tmp_path):
        """search() should return a list of chunk dicts."""
        kb = self._make_kb()
        with (
            patch("src.knowledge_base.FAISS_INDEX_PATH", tmp_path / "index.faiss"),
            patch("src.knowledge_base.CHUNKS_PATH", tmp_path / "chunks.json"),
            patch("src.knowledge_base.PROCESSED_DIR", tmp_path),
        ):
            kb.build(SAMPLE_DOCUMENTS)
            results = kb.search("Как принимать настойку прополиса?")

        assert isinstance(results, list)
        assert len(results) > 0
        for r in results:
            assert "text" in r
            assert "source" in r
            assert "score" in r

    def test_search_score_is_float(self, tmp_path):
        """Each result's 'score' field should be a float."""
        kb = self._make_kb()
        with (
            patch("src.knowledge_base.FAISS_INDEX_PATH", tmp_path / "index.faiss"),
            patch("src.knowledge_base.CHUNKS_PATH", tmp_path / "chunks.json"),
            patch("src.knowledge_base.PROCESSED_DIR", tmp_path),
        ):
            kb.build(SAMPLE_DOCUMENTS)
            results = kb.search("Перга польза")

        for r in results:
            assert isinstance(r["score"], float)

    def test_search_top_k_limits_results(self, tmp_path):
        """search(top_k=2) should return at most 2 results."""
        kb = self._make_kb()
        with (
            patch("src.knowledge_base.FAISS_INDEX_PATH", tmp_path / "index.faiss"),
            patch("src.knowledge_base.CHUNKS_PATH", tmp_path / "chunks.json"),
            patch("src.knowledge_base.PROCESSED_DIR", tmp_path),
        ):
            kb.build(SAMPLE_DOCUMENTS)
            results = kb.search("мёд", top_k=2)

        assert len(results) <= 2

    def test_search_propolis_returns_relevant_chunk(self, tmp_path):
        """Querying about прополис should return the прополис chunk as top result."""
        kb = self._make_kb()
        with (
            patch("src.knowledge_base.FAISS_INDEX_PATH", tmp_path / "index.faiss"),
            patch("src.knowledge_base.CHUNKS_PATH", tmp_path / "chunks.json"),
            patch("src.knowledge_base.PROCESSED_DIR", tmp_path),
        ):
            kb.build(SAMPLE_DOCUMENTS)
            results = kb.search("Как принимать настойку прополиса?", top_k=3)

        sources = [r["source"] for r in results]
        assert "pdf:Прополис" in sources

    def test_search_perga_returns_relevant_chunk(self, tmp_path):
        """Querying about перга should return the перга chunk."""
        kb = self._make_kb()
        with (
            patch("src.knowledge_base.FAISS_INDEX_PATH", tmp_path / "index.faiss"),
            patch("src.knowledge_base.CHUNKS_PATH", tmp_path / "chunks.json"),
            patch("src.knowledge_base.PROCESSED_DIR", tmp_path),
        ):
            kb.build(SAMPLE_DOCUMENTS)
            results = kb.search("Чем полезна перга?", top_k=3)

        sources = [r["source"] for r in results]
        assert "pdf:Перга" in sources

    def test_search_krem_med_returns_relevant_chunk(self, tmp_path):
        """Querying about крем-мёд should return the мёд chunk."""
        kb = self._make_kb()
        with (
            patch("src.knowledge_base.FAISS_INDEX_PATH", tmp_path / "index.faiss"),
            patch("src.knowledge_base.CHUNKS_PATH", tmp_path / "chunks.json"),
            patch("src.knowledge_base.PROCESSED_DIR", tmp_path),
        ):
            kb.build(SAMPLE_DOCUMENTS)
            results = kb.search("Как сделать крем-мёд?", top_k=3)

        sources = [r["source"] for r in results]
        assert "pdf:Мёд" in sources

    def test_search_pzhvm_returns_relevant_chunk(self, tmp_path):
        """Querying about ПЖВМ should return the ПЖВМ chunk."""
        kb = self._make_kb()
        with (
            patch("src.knowledge_base.FAISS_INDEX_PATH", tmp_path / "index.faiss"),
            patch("src.knowledge_base.CHUNKS_PATH", tmp_path / "chunks.json"),
            patch("src.knowledge_base.PROCESSED_DIR", tmp_path),
        ):
            kb.build(SAMPLE_DOCUMENTS)
            results = kb.search("Что такое ПЖВМ?", top_k=3)

        sources = [r["source"] for r in results]
        assert "pdf:ПЖВМ" in sources

    def test_search_winter_returns_relevant_chunk(self, tmp_path):
        """Querying about зимовка пчёл should return the зима chunk."""
        kb = self._make_kb()
        with (
            patch("src.knowledge_base.FAISS_INDEX_PATH", tmp_path / "index.faiss"),
            patch("src.knowledge_base.CHUNKS_PATH", tmp_path / "chunks.json"),
            patch("src.knowledge_base.PROCESSED_DIR", tmp_path),
        ):
            kb.build(SAMPLE_DOCUMENTS)
            results = kb.search("Как подготовить пчёл к зиме?", top_k=3)

        sources = [r["source"] for r in results]
        assert "pdf:Зима" in sources

    def test_load_from_disk(self, tmp_path):
        """load() should restore chunks and index from disk."""
        # Build and save
        kb1 = self._make_kb()
        with (
            patch("src.knowledge_base.FAISS_INDEX_PATH", tmp_path / "index.faiss"),
            patch("src.knowledge_base.CHUNKS_PATH", tmp_path / "chunks.json"),
            patch("src.knowledge_base.PROCESSED_DIR", tmp_path),
        ):
            kb1.build(SAMPLE_DOCUMENTS)
            n_chunks = len(kb1.chunks)

        # Load fresh instance from disk
        kb2 = self._make_kb()
        with (
            patch("src.knowledge_base.FAISS_INDEX_PATH", tmp_path / "index.faiss"),
            patch("src.knowledge_base.CHUNKS_PATH", tmp_path / "chunks.json"),
            patch("src.knowledge_base.PROCESSED_DIR", tmp_path),
        ):
            kb2.load()

        assert len(kb2.chunks) == n_chunks

    def test_load_then_search(self, tmp_path):
        """After load(), search should work correctly."""
        kb1 = self._make_kb()
        with (
            patch("src.knowledge_base.FAISS_INDEX_PATH", tmp_path / "index.faiss"),
            patch("src.knowledge_base.CHUNKS_PATH", tmp_path / "chunks.json"),
            patch("src.knowledge_base.PROCESSED_DIR", tmp_path),
        ):
            kb1.build(SAMPLE_DOCUMENTS)

        kb2 = self._make_kb()
        with (
            patch("src.knowledge_base.FAISS_INDEX_PATH", tmp_path / "index.faiss"),
            patch("src.knowledge_base.CHUNKS_PATH", tmp_path / "chunks.json"),
            patch("src.knowledge_base.PROCESSED_DIR", tmp_path),
        ):
            kb2.load()
            results = kb2.search("Чем полезна перга?", top_k=3)

        assert len(results) > 0
        sources = [r["source"] for r in results]
        assert "pdf:Перга" in sources

    def test_search_does_not_mutate_original_chunks(self, tmp_path):
        """search() should not modify the stored chunks list."""
        kb = self._make_kb()
        with (
            patch("src.knowledge_base.FAISS_INDEX_PATH", tmp_path / "index.faiss"),
            patch("src.knowledge_base.CHUNKS_PATH", tmp_path / "chunks.json"),
            patch("src.knowledge_base.PROCESSED_DIR", tmp_path),
        ):
            kb.build(SAMPLE_DOCUMENTS)
            before = [c.copy() for c in kb.chunks]
            kb.search("прополис")
            after = kb.chunks

        assert len(before) == len(after)
        for b, a in zip(before, after):
            assert "score" not in a  # score must not pollute stored chunks
