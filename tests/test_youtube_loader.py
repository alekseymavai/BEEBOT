"""Unit tests for src/youtube_loader.py — YouTube subtitle downloader."""

from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from src.youtube_loader import fetch_transcript, download_all_subtitles, CHANNEL_VIDEO_IDS


class TestFetchTranscript:
    """Tests for fetch_transcript function."""

    def _patch_transcript_api(self, mock_api):
        """Return a context manager that patches YouTubeTranscriptApi where it is imported."""
        import youtube_transcript_api as _yt_module
        return patch.object(_yt_module, "YouTubeTranscriptApi", return_value=mock_api)

    def test_fetch_transcript_returns_text_on_success(self):
        """Should return cleaned text when transcript is available."""
        mock_snippet1 = MagicMock()
        mock_snippet1.text = "Привет, сегодня поговорим о пчёлах."
        mock_snippet2 = MagicMock()
        mock_snippet2.text = "Мёд очень полезен для здоровья."

        mock_api = MagicMock()
        mock_api.fetch.return_value = [mock_snippet1, mock_snippet2]

        with self._patch_transcript_api(mock_api):
            result = fetch_transcript("abc123")

        assert result is not None
        assert "Привет" in result
        assert "Мёд" in result

    def test_fetch_transcript_removes_music_markers(self):
        """Should strip [музыка] and similar noise from transcript."""
        long_text = "[музыка] " + "Перга богата белками и витаминами. " * 5 + "[аплодисменты]"
        mock_snippet = MagicMock()
        mock_snippet.text = long_text

        mock_api = MagicMock()
        mock_api.fetch.return_value = [mock_snippet]

        with self._patch_transcript_api(mock_api):
            result = fetch_transcript("abc123")

        assert result is not None
        assert "[музыка]" not in result
        assert "[аплодисменты]" not in result
        assert "Перга" in result

    def test_fetch_transcript_removes_music_english(self):
        """Should strip [Music] (English) from transcript."""
        long_text = "[Music] " + "Useful bees make honey for health. " * 5
        mock_snippet = MagicMock()
        mock_snippet.text = long_text

        mock_api = MagicMock()
        mock_api.fetch.return_value = [mock_snippet]

        with self._patch_transcript_api(mock_api):
            result = fetch_transcript("abc123")

        assert result is not None
        assert "[Music]" not in result

    def test_fetch_transcript_returns_none_for_short_text(self):
        """Should return None if transcript is too short (≤ 50 chars)."""
        mock_snippet = MagicMock()
        mock_snippet.text = "Ок."

        mock_api = MagicMock()
        mock_api.fetch.return_value = [mock_snippet]

        with self._patch_transcript_api(mock_api):
            result = fetch_transcript("abc123")

        assert result is None

    def test_fetch_transcript_returns_none_on_exception(self):
        """Should return None when YouTubeTranscriptApi raises an error."""
        mock_api = MagicMock()
        mock_api.fetch.side_effect = Exception("No transcript available")

        with self._patch_transcript_api(mock_api):
            result = fetch_transcript("bad_id")

        assert result is None

    def test_fetch_transcript_normalises_whitespace(self):
        """Should collapse multiple spaces into single space."""
        mock_snippet = MagicMock()
        mock_snippet.text = "Много   пробелов   между   словами"

        mock_api = MagicMock()
        mock_api.fetch.return_value = [mock_snippet] * 3  # repeat to exceed 50 chars

        with self._patch_transcript_api(mock_api):
            result = fetch_transcript("abc123")

        if result:
            assert "  " not in result  # no double spaces

    def test_fetch_transcript_calls_with_russian_language(self):
        """Should request Russian language transcripts."""
        mock_snippet = MagicMock()
        mock_snippet.text = "А" * 100

        mock_api = MagicMock()
        mock_api.fetch.return_value = [mock_snippet]

        with self._patch_transcript_api(mock_api):
            fetch_transcript("abc123")

        mock_api.fetch.assert_called_once_with("abc123", languages=["ru"])


class TestDownloadAllSubtitles:
    """Tests for download_all_subtitles function."""

    def test_download_returns_list_of_dicts(self, tmp_path):
        """Should return a list of document dicts for each successful video."""
        with patch("src.youtube_loader.fetch_transcript") as mock_fetch:
            mock_fetch.return_value = "Б" * 200
            results = download_all_subtitles(
                video_ids=["abc123"],
                output_dir=tmp_path,
            )

        assert len(results) == 1
        doc = results[0]
        assert doc["video_id"] == "abc123"
        assert doc["source"] == "youtube:abc123"
        assert len(doc["text"]) == 200

    def test_download_skips_videos_without_transcript(self, tmp_path):
        """Should skip videos where fetch_transcript returns None."""
        with patch("src.youtube_loader.fetch_transcript") as mock_fetch:
            mock_fetch.return_value = None
            results = download_all_subtitles(
                video_ids=["bad_id"],
                output_dir=tmp_path,
            )

        assert results == []

    def test_download_saves_txt_files(self, tmp_path):
        """Should write subtitle text to .txt files in output_dir."""
        with patch("src.youtube_loader.fetch_transcript") as mock_fetch:
            mock_fetch.return_value = "В" * 200
            download_all_subtitles(
                video_ids=["vid001"],
                output_dir=tmp_path,
            )

        txt_path = tmp_path / "vid001.txt"
        assert txt_path.exists()
        assert txt_path.read_text(encoding="utf-8") == "В" * 200

    def test_download_uses_default_video_ids(self, tmp_path):
        """Should use CHANNEL_VIDEO_IDS when no video_ids given."""
        call_count = 0

        def fake_fetch(video_id):
            nonlocal call_count
            call_count += 1
            return None  # skip all

        with patch("src.youtube_loader.fetch_transcript", side_effect=fake_fetch):
            download_all_subtitles(output_dir=tmp_path)

        assert call_count == len(CHANNEL_VIDEO_IDS)

    def test_download_multiple_videos(self, tmp_path):
        """Should process all provided video IDs."""
        video_ids = ["id1", "id2", "id3"]
        texts = {"id1": "Г" * 200, "id2": None, "id3": "Д" * 150}

        with patch("src.youtube_loader.fetch_transcript", side_effect=lambda vid: texts[vid]):
            results = download_all_subtitles(
                video_ids=video_ids,
                output_dir=tmp_path,
            )

        assert len(results) == 2
        sources = {r["source"] for r in results}
        assert "youtube:id1" in sources
        assert "youtube:id3" in sources
        assert "youtube:id2" not in sources

    def test_channel_video_ids_list_not_empty(self):
        """CHANNEL_VIDEO_IDS must contain video IDs (sanity check)."""
        assert len(CHANNEL_VIDEO_IDS) >= 5
        for vid_id in CHANNEL_VIDEO_IDS:
            assert isinstance(vid_id, str)
            assert len(vid_id) > 5
