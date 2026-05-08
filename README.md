# Audio Transcribator MVP

FastAPI-сервис для обработки аудио- и видеофайлов: загрузка файла, извлечение аудио через `ffmpeg`, транскрибация через `faster-whisper`, генерация summary через LLM и скачивание результатов.

## Возможности

- `POST /login` — авторизация и получение access token.
- `POST /process` — загрузка аудио/видео и запуск фоновой обработки.
- `GET /result/{job_id}` — получение результата обработки по ID задачи.
- `GET /download/{job_id}/{filename}` — скачивание отдельных файлов результата.
- Token-based авторизация через `Authorization: Bearer <token>`.
- Поддержка аудио и видеоформатов, совместимых с `ffmpeg`.
- Опциональная diarization через `pyannote`, если включен `ENABLE_DIARIZATION=true`.

## Запуск локально

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
uvicorn app:app --reload
```

## Docker Compose

```powershell
Copy-Item .env.example .env
docker compose up --build
```

После запуска через Docker Compose API доступен на `http://localhost:8001`.

## Структура

```text
audio_transcribator/
  api/routes.py          # FastAPI endpoint'ы
  auth.py                # token-based авторизация
  config.py              # настройки из env
  models.py              # Pydantic-схемы
  services/
    audio.py             # ffmpeg и подготовка аудио
    transcription.py     # faster-whisper
    summary.py           # LLM summary
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

Основные переменные окружения лежат в `.env.example`:

- `API_TOKEN`, `API_USERNAME`, `API_PASSWORD`
- `OPENROUTER_API_KEY`, `OPENROUTER_BASE_URL`, `SUMMARY_MODEL`
- `WHISPER_MODEL`, `WHISPER_COMPUTE_TYPE`
- `ENABLE_DIARIZATION`, `HF_TOKEN`

## Что не коммитить

- входные медиафайлы: `*.mp3`, `*.mp4`, `*.wav`
- результаты обработки: `transcript.txt`, `summary.txt`, `diarization.txt`, `speaker_transcript.txt`, `work_audio.wav`
- логи: `*.log`
- backup/черновики: `*_backup.py`, `*_before_api.py`
- секреты: `.env`, API/HF/OpenRouter tokens
