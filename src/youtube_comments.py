"""Download YouTube comments with author replies for knowledge base.

For each video fetches all comment threads, filters threads where the channel
owner replied, and saves Q&A pairs to data/comments/{video_id}.txt.

Format:
    Вопрос подписчика: {question}
    Ответ Александра Дмитрова: {answer}

    Вопрос подписчика: ...
    ...

Quota cost: ~1 unit per page (100 comments) per video.
"""

import logging
import time
from pathlib import Path

import httpx

from src.config import COMMENTS_DIR, YOUTUBE_API_KEY, YOUTUBE_CHANNEL_HANDLE

logger = logging.getLogger(__name__)

_YOUTUBE_API_BASE = "https://www.googleapis.com/youtube/v3"


def _resolve_channel_id(api_key: str, handle: str) -> str | None:
    """Получить channel_id по handle через YouTube Data API."""
    handle_clean = handle.lstrip("@")
    try:
        with httpx.Client(timeout=10) as client:
            r = client.get(
                f"{_YOUTUBE_API_BASE}/channels",
                params={"part": "id", "forHandle": handle_clean, "key": api_key},
            )
            r.raise_for_status()
            items = r.json().get("items", [])
            return items[0]["id"] if items else None
    except Exception as e:
        logger.error("YouTube API: не удалось получить channel_id для %s: %s", handle, e)
        return None


def fetch_video_comments(
    video_id: str,
    channel_id: str,
    api_key: str,
) -> list[tuple[str, str]]:
    """Скачать все Q&A пары (вопрос подписчика + ответ автора) для одного видео.

    Returns:
        Список (question, answer) где answer принадлежит владельцу канала.
    """
    qa_pairs: list[tuple[str, str]] = []
    params: dict = {
        "part": "snippet,replies",
        "videoId": video_id,
        "maxResults": 100,
        "order": "relevance",
        "textFormat": "plainText",
        "key": api_key,
    }

    with httpx.Client(timeout=15) as client:
        while True:
            try:
                r = client.get(f"{_YOUTUBE_API_BASE}/commentThreads", params=params)
                r.raise_for_status()
                data = r.json()
            except Exception as e:
                logger.error("Comments API error for %s: %s", video_id, e)
                break

            for thread in data.get("items", []):
                top = thread["snippet"]["topLevelComment"]["snippet"]
                question = top["textDisplay"].strip()

                replies = thread.get("replies", {}).get("comments", [])
                author_reply = None
                for reply in replies:
                    s = reply["snippet"]
                    if s.get("authorChannelId", {}).get("value") == channel_id:
                        author_reply = s["textDisplay"].strip()
                        break

                if author_reply and len(question) > 10 and len(author_reply) > 10:
                    qa_pairs.append((question, author_reply))

            next_page = data.get("nextPageToken")
            if not next_page:
                break
            params["pageToken"] = next_page

    return qa_pairs


def download_all_comments(
    video_ids: list[str],
    api_key: str | None = None,
    channel_handle: str | None = None,
    output_dir: Path | None = None,
    delay: float = 2.0,
) -> dict[str, int]:
    """Скачать комментарии с ответами автора для всех видео.

    Args:
        video_ids: Список video_id.
        api_key: YouTube Data API ключ (по умолчанию из YOUTUBE_API_KEY).
        channel_handle: Хэндл канала (по умолчанию из YOUTUBE_CHANNEL_HANDLE).
        output_dir: Директория для сохранения (по умолчанию data/comments/).
        delay: Пауза между запросами в секундах.

    Returns:
        dict {video_id: количество_Q&A_пар}.
    """
    api_key = api_key or YOUTUBE_API_KEY
    channel_handle = channel_handle or YOUTUBE_CHANNEL_HANDLE
    output_dir = output_dir or COMMENTS_DIR

    if not api_key:
        logger.error("YOUTUBE_API_KEY не задан")
        return {}

    output_dir.mkdir(parents=True, exist_ok=True)

    channel_id = _resolve_channel_id(api_key, channel_handle)
    if not channel_id:
        logger.error("Не удалось получить channel_id для %s", channel_handle)
        return {}

    logger.info(
        "Скачиваю комментарии для %d видео (channel_id: %s)", len(video_ids), channel_id
    )

    results: dict[str, int] = {}
    for i, video_id in enumerate(video_ids):
        qa_pairs = fetch_video_comments(video_id, channel_id, api_key)

        if qa_pairs:
            text = "\n\n".join(
                f"Вопрос подписчика: {q}\nОтвет Александра Дмитрова: {a}"
                for q, a in qa_pairs
            )
            (output_dir / f"{video_id}.txt").write_text(text, encoding="utf-8")
            logger.info("Comments: %s — %d Q&A пар сохранено", video_id, len(qa_pairs))
        else:
            logger.info("Comments: %s — нет ответов автора", video_id)

        results[video_id] = len(qa_pairs)

        if i < len(video_ids) - 1:
            time.sleep(delay)

    total = sum(results.values())
    logger.info("Итого: %d Q&A пар по %d видео", total, len(video_ids))
    return results
