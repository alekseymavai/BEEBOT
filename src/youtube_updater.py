"""Автоматическое обновление базы знаний из YouTube-канала.

Логика:
  1. YouTube Data API v3 — получить список последних видео с канала
  2. Сравнить с известным списком CHANNEL_VIDEO_IDS
  3. Для новых видео скачать субтитры через youtube-transcript-api
  4. Пересобрать FAISS-индекс (build_kb.build())

Требует: YOUTUBE_API_KEY в .env
Опционально: YOUTUBE_CHANNEL_HANDLE (по умолчанию @a.dmitrov)

Команды в боте (только ADMIN_CHAT_ID):
  /yt_check  — проверить наличие новых видео
  /yt_update — скачать субтитры для новых видео + пересобрать KB
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import httpx

from src.config import SUBTITLES_DIR
from src.youtube_loader import fetch_transcript

logger = logging.getLogger(__name__)

# Настройки канала
_YOUTUBE_API_BASE = "https://www.googleapis.com/youtube/v3"
_DEFAULT_CHANNEL_HANDLE = "@a.dmitrov"
_MAX_RESULTS = 500  # получить все видео канала (постранично по 50)


async def _resolve_channel_id(api_key: str, handle: str) -> str | None:
    """Получить channel_id по handle через YouTube Data API."""
    handle_clean = handle.lstrip("@")
    url = f"{_YOUTUBE_API_BASE}/channels"
    params = {"part": "id", "forHandle": handle_clean, "key": api_key}
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url, params=params)
            r.raise_for_status()
            items = r.json().get("items", [])
            return items[0]["id"] if items else None
    except Exception as e:
        logger.error("YouTube API: не удалось получить channel_id для %s: %s", handle, e)
        return None


async def _get_upload_playlist_id(api_key: str, channel_id: str) -> str | None:
    """Получить ID плейлиста загрузок (uploads) для канала."""
    url = f"{_YOUTUBE_API_BASE}/channels"
    params = {"part": "contentDetails", "id": channel_id, "key": api_key}
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url, params=params)
            r.raise_for_status()
            items = r.json().get("items", [])
            if not items:
                return None
            return items[0]["contentDetails"]["relatedPlaylists"]["uploads"]
    except Exception as e:
        logger.error("YouTube API: не удалось получить playlist_id: %s", e)
        return None


async def _list_playlist_videos(
    api_key: str, playlist_id: str, max_results: int = _MAX_RESULTS
) -> list[str]:
    """Получить список video_id из плейлиста (последние max_results видео)."""
    url = f"{_YOUTUBE_API_BASE}/playlistItems"
    params = {
        "part": "contentDetails",
        "playlistId": playlist_id,
        "maxResults": min(max_results, 50),
        "key": api_key,
    }
    video_ids: list[str] = []
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            while True:
                r = await client.get(url, params=params)
                r.raise_for_status()
                data = r.json()
                for item in data.get("items", []):
                    vid = item["contentDetails"].get("videoId")
                    if vid:
                        video_ids.append(vid)
                next_page = data.get("nextPageToken")
                if not next_page or len(video_ids) >= max_results:
                    break
                params["pageToken"] = next_page
    except Exception as e:
        logger.error("YouTube API: ошибка при получении видео из плейлиста: %s", e)
    return video_ids


def _get_known_ids() -> set[str]:
    """Получить ID видео, субтитры которых уже есть в data/subtitles/."""
    return {p.stem for p in SUBTITLES_DIR.glob("*.txt")}


async def check_new_videos(
    api_key: str,
    channel_handle: str = _DEFAULT_CHANNEL_HANDLE,
    known_ids: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Проверить новые видео на канале.

    Returns:
        (all_channel_ids, new_ids) — все видео канала и только новые (нет субтитров в data/subtitles/).
    """
    known = set(known_ids) if known_ids is not None else _get_known_ids()

    channel_id = await _resolve_channel_id(api_key, channel_handle)
    if not channel_id:
        return [], []

    playlist_id = await _get_upload_playlist_id(api_key, channel_id)
    if not playlist_id:
        return [], []

    all_ids = await _list_playlist_videos(api_key, playlist_id)
    new_ids = [vid for vid in all_ids if vid not in known]
    return all_ids, new_ids


async def download_new_subtitles(new_ids: list[str]) -> tuple[int, int]:
    """Скачать субтитры для новых видео.

    Returns:
        (downloaded, failed) — количество успешных и неудачных загрузок.
    """
    SUBTITLES_DIR.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    failed = 0
    for vid in new_ids:
        txt_path = SUBTITLES_DIR / f"{vid}.txt"
        if txt_path.exists():
            logger.info("YouTube: %s — субтитры уже есть, пропускаем", vid)
            downloaded += 1
            continue
        text = fetch_transcript(vid)
        if text:
            txt_path.write_text(text, encoding="utf-8")
            logger.info("YouTube: %s — субтитры скачаны (%d символов)", vid, len(text))
            downloaded += 1
        else:
            logger.warning("YouTube: %s — субтитры недоступны", vid)
            failed += 1
    return downloaded, failed


def rebuild_knowledge_base() -> int:
    """Пересобрать FAISS-индекс с новыми субтитрами.

    Returns:
        Количество чанков в новом индексе.
    """
    from src.build_kb import build as build_kb

    logger.info("YouTube updater: пересборка базы знаний...")
    build_kb()
    # Прочитать кол-во чанков из файла
    from src.config import CHUNKS_PATH
    try:
        with open(CHUNKS_PATH) as f:
            chunks = json.load(f)
        return len(chunks)
    except Exception:
        return 0


async def run_update(
    api_key: str,
    channel_handle: str = _DEFAULT_CHANNEL_HANDLE,
    rebuild_kb: bool = True,
) -> str:
    """Полный цикл обновления: проверка → загрузка → пересборка KB.

    Returns:
        Отчёт в виде строки (Markdown).
    """
    logger.info("YouTube updater: запуск обновления канала %s", channel_handle)

    all_ids, new_ids = await check_new_videos(api_key, channel_handle)
    if not all_ids:
        return "❌ Не удалось получить список видео. Проверь YOUTUBE_API_KEY."

    known_count = len(_get_known_ids())
    lines = [
        f"📺 *Канал {channel_handle}*\n",
        f"Всего видео: {len(all_ids)}",
        f"Субтитров в KB: {known_count}",
        f"Новых (без субтитров): *{len(new_ids)}*",
    ]

    if not new_ids:
        lines.append("\n✅ База знаний актуальна — новых видео нет.")
        return "\n".join(lines)

    lines.append(f"\nСкачиваю субтитры для {len(new_ids)} новых видео...")
    downloaded, failed = await download_new_subtitles(new_ids)
    lines.append(f"✅ Скачано: {downloaded}, ❌ Недоступно: {failed}")

    if downloaded > 0 and rebuild_kb:
        lines.append("\nПересобираю базу знаний...")
        n_chunks = rebuild_knowledge_base()
        lines.append(f"✅ База знаний пересобрана: *{n_chunks} чанков*")

    return "\n".join(lines)
