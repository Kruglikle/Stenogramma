# Audio Transcribator MVP

FastAPI-сервис для загрузки аудио/видео, извлечения аудио через ffmpeg, транскрибации через faster-whisper и генерации summary через LLM.

## Запуск

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app:app --reload
```

API сохраняет прежние endpoint'ы:

- `POST /login`
- `POST /process`
- `GET /result/{job_id}`
- `GET /download/{job_id}/{filename}`

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

## Docker Compose

```powershell
Copy-Item .env.example .env
docker compose up --build
```

## Файлы, которые не нужно коммитить

- входные медиафайлы: `*.mp3`, `*.mp4`, `*.wav`
- результаты обработки: `transcript.txt`, `summary.txt`, `diarization.txt`, `speaker_transcript.txt`, `work_audio.wav`
- логи: `*.log`
- backup/черновики: `*_backup.py`, `*_before_api.py`
- секреты: `.env`, API/HF/OpenRouter tokens

