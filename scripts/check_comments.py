"""
Проверка топ-5 видео по комментариям с ответами автора.
Квота: ~6 единиц из 10 000.
"""
import os
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from googleapiclient.discovery import build
from src.youtube_loader import CHANNEL_VIDEO_IDS

API_KEY = os.environ.get("YOUTUBE_API_KEY") or sys.argv[1] if len(sys.argv) > 1 else None
if not API_KEY:
    print("Нужен YOUTUBE_API_KEY в .env или как аргумент")
    sys.exit(1)

youtube = build("youtube", "v3", developerKey=API_KEY)

# 1. Батч-запрос статистики всех видео (1 единица квоты)
print(f"Запрашиваю статистику {len(CHANNEL_VIDEO_IDS)} видео...")
resp = youtube.videos().list(
    part="snippet,statistics",
    id=",".join(CHANNEL_VIDEO_IDS),
    maxResults=50,
).execute()

videos = []
channel_id = None
for item in resp["items"]:
    stats = item.get("statistics", {})
    count = int(stats.get("commentCount", 0))
    videos.append({
        "id": item["id"],
        "title": item["snippet"]["title"][:60],
        "comments": count,
        "channel_id": item["snippet"]["channelId"],
    })
    if not channel_id:
        channel_id = item["snippet"]["channelId"]

videos.sort(key=lambda x: x["comments"], reverse=True)
top5 = videos[:5]

print(f"\nТоп-5 видео по комментариям (channel_id автора: {channel_id}):\n")
for i, v in enumerate(top5, 1):
    print(f"  {i}. [{v['comments']} комм] {v['title']}")
    print(f"     https://youtu.be/{v['id']}")

# 2. Для каждого из топ-5 — первая страница комментариев с ответами автора
print("\n" + "="*60)
print("Ищу ответы автора...\n")

total_qa = 0
for v in top5:
    print(f"▶ {v['title']}")
    time.sleep(2)  # пауза между запросами

    try:
        resp = youtube.commentThreads().list(
            part="snippet,replies",
            videoId=v["id"],
            maxResults=100,
            order="relevance",
            textFormat="plainText",
        ).execute()
    except Exception as e:
        print(f"  Ошибка: {e}\n")
        continue

    qa_pairs = []
    for thread in resp["items"]:
        top = thread["snippet"]["topLevelComment"]["snippet"]
        question = top["textDisplay"].strip()

        # Ищем ответ автора в replies
        replies = thread.get("replies", {}).get("comments", [])
        author_reply = None
        for reply in replies:
            s = reply["snippet"]
            if s.get("authorChannelId", {}).get("value") == channel_id:
                author_reply = s["textDisplay"].strip()
                break

        if author_reply:
            qa_pairs.append((question, author_reply))

    print(f"  Найдено Q&A пар: {len(qa_pairs)} (из {len(resp['items'])} тредов)")
    for q, a in qa_pairs[:3]:  # показываем первые 3
        print(f"\n  Q: {q[:120]}")
        print(f"  A: {a[:200]}")
    if len(qa_pairs) > 3:
        print(f"  ... и ещё {len(qa_pairs) - 3} пар")
    total_qa += len(qa_pairs)
    print()

print("="*60)
print(f"Итого Q&A пар в топ-5 видео (первые 100 комм каждого): {total_qa}")
print("Примечание: это только первая страница, у популярных видео комментариев больше.")
