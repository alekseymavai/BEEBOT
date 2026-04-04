"""Тесты ConsultService — консультации через KB + LLM."""

from unittest.mock import MagicMock

from src.services.consult_service import ConsultService


def _make_service(tunnel_healthy=True):
    kb = MagicMock()
    kb.search.return_value = [
        {"text": "Перга — пчелиный хлеб", "source": "txt:Перга"},
        {"text": "Принимать по 1 ч.л.", "source": "txt:Перга"},
    ]
    llm = MagicMock()
    llm.generate.return_value = "Перга — один из самых ценных продуктов пчеловодства."

    tunnel = None
    if not tunnel_healthy:
        tunnel = MagicMock()
        tunnel.is_healthy = False

    return ConsultService(kb=kb, llm=llm, tunnel_monitor=tunnel), kb, llm


class TestAnswer:
    def test_normal_response(self):
        svc, kb, llm = _make_service()
        response, chunks = svc.answer("что такое перга?")
        assert "Перга" in response
        assert len(chunks) == 2
        kb.search.assert_called_once_with("что такое перга?")
        llm.generate.assert_called_once()

    def test_passes_style_and_history(self):
        svc, _kb, llm = _make_service()
        svc.answer("вопрос", history=[{"role": "user", "content": "hi"}], style="founder")
        call_kwargs = llm.generate.call_args
        assert call_kwargs.kwargs.get("style") == "founder"
        assert call_kwargs.kwargs.get("history") == [{"role": "user", "content": "hi"}]


class TestFaqFallback:
    def test_tunnel_down_uses_fallback(self):
        svc, kb, llm = _make_service(tunnel_healthy=False)
        response, chunks = svc.answer("как принимать пергу?")
        assert "автономном режиме" in response
        assert len(chunks) <= 3
        llm.generate.assert_not_called()

    def test_tunnel_down_no_results(self):
        svc, kb, _llm = _make_service(tunnel_healthy=False)
        kb.search.return_value = []
        response, chunks = svc.answer("абракадабра")
        assert "ничего не найдено" in response
        assert chunks == []

    def test_direct_faq_fallback(self):
        svc, _kb, _llm = _make_service()
        response, chunks = svc.faq_fallback("перга")
        assert "автономном режиме" in response
        assert len(chunks) <= 3
