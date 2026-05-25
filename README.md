# Стенограмма

FastAPI-сервис для обработки аудио- и видеофайлов: загрузка файла, извлечение аудио через `ffmpeg`, транскрибация через `faster-whisper`, генерация summary через LLM и скачивание результатов.

## Возможности

- `POST /login` — авторизация и получение access token.
- `POST /add-user` — добавление нового пользователя администратором.
- `POST /process` — загрузка аудио/видео и запуск фоновой обработки.
- `GET /transcription-models` — список доступных моделей транскрибации.
- `GET /result/{job_id}` — получение результата обработки по ID задачи.
- `POST /result/{job_id}/edit` — ИИ-редактура готовой стенограммы через OpenRouter.
- `GET /download/{job_id}/{filename}` — скачивание отдельных файлов результата.
- Token-based авторизация через `Authorization: Bearer <token>`.
- Пользователи хранятся в PostgreSQL, пароль сохраняется в виде hash.
- Поддержка аудио и видеоформатов, совместимых с `ffmpeg`.
- Выбор модели транскрибации: локальная `faster-whisper` или OpenRouter STT модели из `audio_transcribator/transcription_models.json`.
- Опциональная diarization через `pyannote`, если включен `ENABLE_DIARIZATION=true`.

## Запуск локально

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
New-Item -ItemType File .env
uvicorn app:app --reload --port 8001
```

Если приложение запускается нативно из `.venv`, а PostgreSQL нужен в Docker, можно поднять только контейнер базы данных:

```powershell
docker compose up -d db
```

Это удобно для локальной разработки: база работает в контейнере и доступна локальному приложению на `localhost:5432`, а FastAPI-приложение запускается как обычный локальный процесс через `uvicorn app:app --reload --port 8001`.

## Docker Compose

```powershell
New-Item -ItemType File .env
docker compose up --build
```

После запуска через Docker Compose API доступен на `http://localhost:8001`.
Простой веб-интерфейс доступен на `http://localhost:8001/ui`.
PostgreSQL поднимается отдельным контейнером `db`; при первом запуске приложение создает таблицу `users` и добавляет пользователя из `API_USERNAME`/`API_PASSWORD`, если такого login еще нет.

Добавить пользователя можно через admin token:

```powershell
$headers = @{
    "Authorization" = "Bearer change-me-add-user-token"
    "Content-Type" = "application/json"
}

$body = '{"username":"newuser","password":"newpassword"}'

Invoke-WebRequest -Uri "http://localhost:8000/add-user" `
    -Method Post `
    -Headers $headers `
    -Body $body
```

То же самое через `curl`:

```bash
curl -X POST http://localhost:8001/add-user \
  -H "Authorization: Bearer change-me-add-user-token" \
  -H "Content-Type: application/json" \
  -d '{"username":"user1","password":"strong-password"}'
```

## Структура

```text
audio_transcribator/
  api/routes.py          # FastAPI endpoint'ы
  ui/routes.py           # web UI: login, upload, result, downloads
  auth.py                # token-based авторизация
  config.py              # настройки из env
  db.py                  # PostgreSQL users table и проверка паролей
  models.py              # Pydantic-схемы
  services/
    audio.py             # ffmpeg и подготовка аудио
    transcription.py     # faster-whisper
    summary.py           # LLM summary
    editor.py            # ИИ-редактура стенограммы
    diarization.py       # pyannote diarization
  utils/files.py         # файловые helper'ы
  worker.py              # CLI/background pipeline
app.py                   # совместимый ASGI entrypoint
process_audio_fast.py    # совместимый CLI wrapper
data/
  uploads/               # входные файлы, не коммитить
  api_results/           # результаты задач, не коммитить
```

## Настройки

Основные переменные окружения задаются в `.env`:

- `API_TOKEN`, `API_USERNAME`, `API_PASSWORD`, `ADD_USER_ADMIN_TOKEN`
- `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `DATABASE_URL`
- `OPENROUTER_API_KEY`, `OPENROUTER_BASE_URL`, `SUMMARY_MODEL`, `EDITOR_MODEL`, `EDITOR_TEMPERATURE`
- `WHISPER_MODEL`, `WHISPER_COMPUTE_TYPE`
- `TRANSCRIPTION_MODELS_FILE`
- `ENABLE_DIARIZATION`, `HF_TOKEN`

