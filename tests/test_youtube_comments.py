"""Unit tests for src/youtube_comments.py — YouTube comment downloader."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_thread(question: str, author_channel_id: str, reply_text: str | None = None):
    """Build a fake commentThread API response item."""
    thread = {
        "snippet": {
            "topLevelComment": {
                "snippet": {
                    "textDisplay": question,
                    "authorChannelId": {"value": "subscriber_ch"},
                }
            }
        },
        "replies": {"comments": []},
    }
    if reply_text:
        thread["replies"]["comments"].append({
            "snippet": {
                "textDisplay": reply_text,
                "authorChannelId": {"value": author_channel_id},
            }
        })
    return thread


def _make_api_page(items: list, next_page_token: str | None = None) -> dict:
    data = {"items": items}
    if next_page_token:
        data["nextPageToken"] = next_page_token
    return data


# ---------------------------------------------------------------------------
# _resolve_channel_id
# ---------------------------------------------------------------------------

class TestResolveChannelId:
    def test_returns_channel_id_on_success(self):
        from src.youtube_comments import _resolve_channel_id

        response_data = {"items": [{"id": "UCxyz123"}]}
        with patch("httpx.Client") as MockClient:
            mock_resp = MagicMock()
            mock_resp.json.return_value = response_data
            mock_resp.raise_for_status = MagicMock()
            MockClient.return_value.__enter__.return_value.get.return_value = mock_resp

            result = _resolve_channel_id("api_key", "@a.dmitrov")

        assert result == "UCxyz123"

    def test_returns_none_when_no_items(self):
        from src.youtube_comments import _resolve_channel_id

        with patch("httpx.Client") as MockClient:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"items": []}
            mock_resp.raise_for_status = MagicMock()
            MockClient.return_value.__enter__.return_value.get.return_value = mock_resp

            result = _resolve_channel_id("api_key", "@unknown")

        assert result is None

    def test_returns_none_on_api_error(self):
        from src.youtube_comments import _resolve_channel_id

        with patch("httpx.Client") as MockClient:
            MockClient.return_value.__enter__.return_value.get.side_effect = Exception("timeout")
            result = _resolve_channel_id("api_key", "@a.dmitrov")

        assert result is None

    def test_strips_at_sign_from_handle(self):
        """Handle with @ should still work — the @ is stripped before the API call."""
        from src.youtube_comments import _resolve_channel_id

        called_params = {}

        def fake_get(url, params=None):
            called_params.update(params or {})
            resp = MagicMock()
            resp.json.return_value = {"items": [{"id": "UCabc"}]}
            resp.raise_for_status = MagicMock()
            return resp

        with patch("httpx.Client") as MockClient:
            MockClient.return_value.__enter__.return_value.get.side_effect = fake_get
            _resolve_channel_id("key", "@a.dmitrov")

        assert called_params.get("forHandle") == "a.dmitrov"


# ---------------------------------------------------------------------------
# fetch_video_comments
# ---------------------------------------------------------------------------

class TestFetchVideoComments:
    def test_returns_qa_pair_when_author_replied(self):
        from src.youtube_comments import fetch_video_comments

        channel_id = "UCowner"
        thread = _make_thread(
            "Как принимать прополис?",
            author_channel_id=channel_id,
            reply_text="20 капель на стакан воды, 3 раза в день.",
        )
        page = _make_api_page([thread])

        with patch("httpx.Client") as MockClient:
            mock_resp = MagicMock()
            mock_resp.json.return_value = page
            mock_resp.raise_for_status = MagicMock()
            MockClient.return_value.__enter__.return_value.get.return_value = mock_resp

            pairs = fetch_video_comments("vid001", channel_id, "api_key")

        assert len(pairs) == 1
        q, a = pairs[0]
        assert "прополис" in q
        assert "капель" in a

    def test_skips_thread_without_author_reply(self):
        from src.youtube_comments import fetch_video_comments

        channel_id = "UCowner"
        thread = _make_thread(
            "Хороший видос!",
            author_channel_id="subscriber_ch",
            reply_text=None,  # no reply
        )
        page = _make_api_page([thread])

        with patch("httpx.Client") as MockClient:
            mock_resp = MagicMock()
            mock_resp.json.return_value = page
            mock_resp.raise_for_status = MagicMock()
            MockClient.return_value.__enter__.return_value.get.return_value = mock_resp

            pairs = fetch_video_comments("vid001", channel_id, "api_key")

        assert pairs == []

    def test_skips_thread_when_reply_is_from_another_channel(self):
        from src.youtube_comments import fetch_video_comments

        channel_id = "UCowner"
        thread = _make_thread(
            "Отличное видео!",
            author_channel_id="UCsomeone_else",  # not the channel owner
            reply_text="Спасибо!",
        )
        page = _make_api_page([thread])

        with patch("httpx.Client") as MockClient:
            mock_resp = MagicMock()
            mock_resp.json.return_value = page
            mock_resp.raise_for_status = MagicMock()
            MockClient.return_value.__enter__.return_value.get.return_value = mock_resp

            pairs = fetch_video_comments("vid001", channel_id, "api_key")

        assert pairs == []

    def test_skips_very_short_texts(self):
        """Q&A pairs where question or answer is ≤ 10 chars are skipped."""
        from src.youtube_comments import fetch_video_comments

        channel_id = "UCowner"
        thread = _make_thread("Ок", author_channel_id=channel_id, reply_text="👍")
        page = _make_api_page([thread])

        with patch("httpx.Client") as MockClient:
            mock_resp = MagicMock()
            mock_resp.json.return_value = page
            mock_resp.raise_for_status = MagicMock()
            MockClient.return_value.__enter__.return_value.get.return_value = mock_resp

            pairs = fetch_video_comments("vid001", channel_id, "api_key")

        assert pairs == []

    def test_paginates_multiple_pages(self):
        """Should follow nextPageToken until exhausted."""
        from src.youtube_comments import fetch_video_comments

        channel_id = "UCowner"

        page1 = _make_api_page(
            [_make_thread("Как правильно принимать прополис?", channel_id, "20 капель на стакан воды утром.")],
            next_page_token="token2",
        )
        page2 = _make_api_page(
            [_make_thread("Можно ли давать пергу детям?", channel_id, "Да, с 3 лет в небольших дозах.")],
        )
        pages = [page1, page2]

        call_count = 0

        def fake_get(url, params=None):
            nonlocal call_count
            resp = MagicMock()
            resp.json.return_value = pages[call_count]
            resp.raise_for_status = MagicMock()
            call_count += 1
            return resp

        with patch("httpx.Client") as MockClient:
            MockClient.return_value.__enter__.return_value.get.side_effect = fake_get
            pairs = fetch_video_comments("vid001", channel_id, "api_key")

        assert len(pairs) == 2
        assert call_count == 2

    def test_returns_empty_on_api_error(self):
        from src.youtube_comments import fetch_video_comments

        with patch("httpx.Client") as MockClient:
            MockClient.return_value.__enter__.return_value.get.side_effect = Exception("network error")
            pairs = fetch_video_comments("vid001", "UCowner", "api_key")

        assert pairs == []


# ---------------------------------------------------------------------------
# download_all_comments
# ---------------------------------------------------------------------------

class TestDownloadAllComments:
    def test_saves_qa_file_per_video(self, tmp_path):
        from src.youtube_comments import download_all_comments

        channel_id = "UCowner"

        with (
            patch("src.youtube_comments._resolve_channel_id", return_value=channel_id),
            patch("src.youtube_comments.fetch_video_comments") as mock_fetch,
            patch("src.youtube_comments.time.sleep"),
        ):
            mock_fetch.return_value = [
                ("Как принимать?", "20 капель на воду."),
            ]
            results = download_all_comments(
                video_ids=["vid001"],
                api_key="api_key",
                channel_handle="@a.dmitrov",
                output_dir=tmp_path,
                delay=0,
            )

        assert results["vid001"] == 1
        txt_path = tmp_path / "vid001.txt"
        assert txt_path.exists()
        content = txt_path.read_text(encoding="utf-8")
        assert "Вопрос подписчика" in content
        assert "Ответ Александра Дмитрова" in content
        assert "20 капель" in content

    def test_skips_videos_with_no_qa_pairs(self, tmp_path):
        from src.youtube_comments import download_all_comments

        with (
            patch("src.youtube_comments._resolve_channel_id", return_value="UCowner"),
            patch("src.youtube_comments.fetch_video_comments", return_value=[]),
            patch("src.youtube_comments.time.sleep"),
        ):
            results = download_all_comments(
                video_ids=["vid_no_qa"],
                api_key="api_key",
                output_dir=tmp_path,
                delay=0,
            )

        assert results["vid_no_qa"] == 0
        assert not (tmp_path / "vid_no_qa.txt").exists()

    def test_returns_empty_when_no_api_key(self, tmp_path):
        from src.youtube_comments import download_all_comments

        with patch("src.youtube_comments.YOUTUBE_API_KEY", None):
            results = download_all_comments(
                video_ids=["vid001"],
                api_key=None,
                output_dir=tmp_path,
            )

        assert results == {}

    def test_returns_empty_when_channel_not_resolved(self, tmp_path):
        from src.youtube_comments import download_all_comments

        with (
            patch("src.youtube_comments._resolve_channel_id", return_value=None),
        ):
            results = download_all_comments(
                video_ids=["vid001"],
                api_key="api_key",
                output_dir=tmp_path,
            )

        assert results == {}

    def test_creates_output_directory(self, tmp_path):
        """Should create output_dir if it doesn't exist."""
        from src.youtube_comments import download_all_comments

        nested = tmp_path / "deep" / "comments"
        assert not nested.exists()

        with (
            patch("src.youtube_comments._resolve_channel_id", return_value="UCowner"),
            patch("src.youtube_comments.fetch_video_comments", return_value=[]),
            patch("src.youtube_comments.time.sleep"),
        ):
            download_all_comments(
                video_ids=["vid001"],
                api_key="api_key",
                output_dir=nested,
                delay=0,
            )

        assert nested.exists()

    def test_formats_multiple_qa_pairs_with_separator(self, tmp_path):
        """Multiple Q&A pairs should be separated by blank lines."""
        from src.youtube_comments import download_all_comments

        with (
            patch("src.youtube_comments._resolve_channel_id", return_value="UCowner"),
            patch("src.youtube_comments.fetch_video_comments", return_value=[
                ("Вопрос первый?", "Ответ первый достаточно длинный."),
                ("Вопрос второй?", "Ответ второй достаточно длинный."),
            ]),
            patch("src.youtube_comments.time.sleep"),
        ):
            download_all_comments(
                video_ids=["vid001"],
                api_key="api_key",
                output_dir=tmp_path,
                delay=0,
            )

        content = (tmp_path / "vid001.txt").read_text(encoding="utf-8")
        # Two Q&A blocks separated by \n\n
        assert content.count("Вопрос подписчика:") == 2
        assert "\n\n" in content

    def test_processes_multiple_videos(self, tmp_path):
        from src.youtube_comments import download_all_comments

        video_ids = ["vid001", "vid002", "vid003"]
        qa_by_video = {
            "vid001": [("Вопрос?", "Ответ достаточно длинный.")],
            "vid002": [],
            "vid003": [("Другой вопрос?", "Другой ответ достаточно длинный.")],
        }

        with (
            patch("src.youtube_comments._resolve_channel_id", return_value="UCowner"),
            patch("src.youtube_comments.fetch_video_comments", side_effect=lambda v, c, k: qa_by_video[v]),
            patch("src.youtube_comments.time.sleep"),
        ):
            results = download_all_comments(
                video_ids=video_ids,
                api_key="api_key",
                output_dir=tmp_path,
                delay=0,
            )

        assert results["vid001"] == 1
        assert results["vid002"] == 0
        assert results["vid003"] == 1
        assert (tmp_path / "vid001.txt").exists()
        assert not (tmp_path / "vid002.txt").exists()
        assert (tmp_path / "vid003.txt").exists()
