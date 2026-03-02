# BEEBOT — Контекст сессии

> Последнее обновление: 2 марта 2026

## Статус проекта

**Бот развёрнут и работает в production.**

| Компонент | Статус | Детали |
|-----------|--------|--------|
| Telegram-бот | ✅ Работает | @AleksandrDmitrov_BEEBOT |
| База знаний | ✅ 410 чанков | 13 PDF + 26 YouTube видео |
| Groq LLM | ✅ Через прокси | llama-3.3-70b-versatile |
| VPS Docker | ✅ Запущен | 185.233.200.13, container `beebot` |
| Группы Telegram | ✅ Поддержка добавлена | По упоминанию @bot или reply |

## Архитектура

```
Telegram → aiogram бот (VPS Docker, network_mode: host)
  → FAISS (семантика 70% + стилометрия 30%, 410 чанков)
  → SSH-туннель (VPS:8990 → hive:8990)
  → Groq Proxy (hive, порт 8990)
  → Groq API (llama-3.3-70b-versatile)
  → Ответ в стиле Александра Дмитрова
```

## Доступы и серверы

| Ресурс | Адрес | Пользователь |
|--------|-------|-------------|
| VPS | 185.233.200.13 | ai-agent (SSH-ключ) |
| GitHub (upstream) | github.com/alekseymavai/BEEBOT | alekseymavai |
| GitHub (fork) | github.com/unidel2035/BEEBOT | unidel2035 |
| Локальный путь | /home/hive/BEEBOT/ | hive |
| Groq Console | console.groq.com | alekseymavai |

**Важно:** аккаунт `unidel2035` не имеет push-доступа к `alekseymavai/BEEBOT`. Все изменения — через fork + PR.

## Ключевые файлы

```
/home/hive/BEEBOT/
├── src/
│   ├── bot.py              # Telegram-бот (aiogram 3, группы + личка)
│   ├── config.py           # Конфигурация из .env
│   ├── knowledge_base.py   # FAISS + стилометрия, гибридный поиск
│   ├── llm_client.py       # Groq API (с поддержкой GROQ_BASE_URL прокси)
│   ├── pdf_loader.py       # Извлечение текста из PDF
│   ├── youtube_loader.py   # Загрузка субтитров YouTube
│   └── build_kb.py         # Сборка базы знаний
├── data/
│   ├── subtitles/          # 26 файлов .txt (131K символов)
│   ├── texts/              # 13 файлов из PDF (52K символов)
│   └── processed/
│       ├── index.faiss     # FAISS-индекс (623 KB)
│       └── chunks.json     # Метаданные чанков (424 KB)
├── groq_proxy.py           # Reverse proxy для Groq API (порт 8990)
├── Dockerfile
├── docker-compose.yml      # network_mode: host
├── deploy.sh               # Скрипт деплоя на VPS
├── beebot.service          # systemd unit
├── .env                    # Секреты (НЕ в git)
└── .env.example
```

## Фоновые процессы на hive (требуют перезапуска при перезагрузке)

```bash
# 1. Groq Proxy (обязательно для работы бота!)
source /home/hive/BEEBOT/.venv/bin/activate
nohup python /home/hive/BEEBOT/groq_proxy.py > /home/hive/BEEBOT/proxy.log 2>&1 &

# 2. SSH-туннель (обязательно для работы бота!)
ssh -f -N -R 8990:localhost:8990 ai-agent@185.233.200.13
```

**ВНИМАНИЕ:** Если hive перезагрузится — бот перестанет отвечать (403 от Groq). Нужно заново запустить прокси и туннель.

## Секреты (.env на VPS)

```
TELEGRAM_BOT_TOKEN=8762491951:AAGvmx8YCJcGaq6HEf8xMGV3NOPehr38H84
GROQ_API_KEY=gsk_...ounD (последние 4: ounD)
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_BASE_URL=http://localhost:8990
```

## Известные проблемы и ограничения

1. **Groq блокирует IP VPS** — решено через SSH-туннель + прокси на hive
2. **YouTube блокирует IP hive** — субтитры скачаны через Docker на VPS
3. **Прокси + туннель на hive** — не переживёт перезагрузку, нужно автоматизировать (systemd или cron @reboot)
4. **llama-3.3-70b-versatile** иногда вставляет иноязычные слова — частично решено промптом, но может повторяться
5. **unidel2035 нет push-доступа** к alekseymavai/BEEBOT — только fork + PR

## GitHub Issues (все открыты)

| # | Название | Статус по факту |
|---|----------|----------------|
| 1 | Настроить окружение | ✅ Сделано |
| 2 | Загрузчик YouTube | ✅ Сделано (26/27 видео) |
| 3 | Гибридная база знаний | ✅ Сделано (410 чанков) |
| 4 | Интеграция с Groq | ✅ Сделано (llama-3.3-70b-versatile) |
| 5 | Telegram-бот | ✅ Сделано (личка + группы) |
| 6 | Деплой на VPS | ✅ Сделано (Docker) |
| 7 | Тестирование | ⚠️ Частично (нужно больше тестов) |

## Git-история

```
1f184bc fix: русскоязычный промпт + прокси для Groq API
57bfaf0 Merge pull request #8 from unidel2035/main
d5e8916 fix: заменить llama3-70b-8192 на llama-3.3-70b-versatile
ae42b4f feat: MVP бота-помощника для блога о пчеловодстве
4d0bb0c Инструкции препаратов на основе пчелопродуктов
aae9beb Initial commit
```

**Незакоммиченные изменения:** `src/bot.py` (поддержка групп), `SESSION_CONTEXT.md`

## Что делать дальше

### Приоритет 1 — Стабильность
- [ ] Автоматизировать прокси + туннель на hive (systemd)
- [ ] Или найти LLM-провайдер без IP-блокировки (OpenRouter, Together AI)
- [ ] Закрыть выполненные Issues на GitHub

### Приоритет 2 — Качество ответов
- [ ] Тонкая настройка системного промпта (больше примеров стиля автора)
- [ ] Улучшить чанкинг (по смысловым блокам, а не по символам)
- [ ] Добавить больше данных: комментарии YouTube, посты в соцсетях

### Приоритет 3 — Функциональность
- [ ] Кнопки (inline keyboard) для популярных вопросов
- [ ] Команда /products — список продуктов
- [ ] Лог вопросов для аналитики
- [ ] Мониторинг (uptime, ошибки)

## Команды для быстрого старта сессии

```bash
# Перейти в проект
cd /home/hive/BEEBOT && source .venv/bin/activate

# Проверить бота на VPS
ssh ai-agent@185.233.200.13 "docker logs --tail 5 beebot"

# Проверить прокси
curl -s -o /dev/null -w "%{http_code}" http://localhost:8990/

# Проверить туннель
ssh ai-agent@185.233.200.13 "curl -s -o /dev/null -w '%{http_code}' http://localhost:8990/"

# Перезапустить бота на VPS
ssh ai-agent@185.233.200.13 "cd /home/ai-agent/BEEBOT && docker restart beebot"

# Пересобрать базу знаний
python -m src.build_kb

# Обновить код на VPS (быстро, без пересборки Docker)
scp src/bot.py ai-agent@185.233.200.13:/home/ai-agent/BEEBOT/src/bot.py
ssh ai-agent@185.233.200.13 "cd /home/ai-agent/BEEBOT && docker rm -f beebot && docker compose up -d --build"

# Восстановить прокси и туннель (после перезагрузки hive)
source /home/hive/BEEBOT/.venv/bin/activate
nohup python /home/hive/BEEBOT/groq_proxy.py > /home/hive/BEEBOT/proxy.log 2>&1 &
ssh -f -N -R 8990:localhost:8990 ai-agent@185.233.200.13
```
