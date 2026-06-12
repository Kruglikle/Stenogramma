# Стенограмма

FastAPI-сервис для обработки аудио- и видеофайлов: загрузка файла, извлечение аудио через `ffmpeg`, транскрибация через `faster-whisper`, генерация summary через LLM и скачивание результатов.

## Возможности

- `POST /login` — авторизация и получение access token.
- `POST /add-user` — добавление нового пользователя администратором.
- `POST /process` — загрузка аудио/видео и запуск фоновой обработки.
- `GET /transcription-models` — список доступных моделей транскрибации.
- `GET /editor-models` — список доступных моделей ИИ-редактуры.
- `GET /result/{job_id}` — получение результата обработки по ID задачи.
- `POST /result/{job_id}/edit` — ИИ-редактура готовой стенограммы через локальную Ollama-модель.
- `GET /download/{job_id}/{filename}` — скачивание отдельных файлов результата.
- Token-based авторизация через `Authorization: Bearer <token>`.
- Пользователи хранятся в PostgreSQL, пароль сохраняется в виде hash.
- Поддержка аудио и видеоформатов, совместимых с `ffmpeg`.
- Транскрибация через локальную `faster-whisper`.
- Выбор модели ИИ-редактуры: локальные `gemma3:4b`, `qwen3:8b` через Ollama.
- Резюме стенограммы опционально генерируется локальной Ollama-моделью из `SUMMARY_MODEL`.
- Диаризация опциональна и выполняется локально через `pyannote.audio` speaker-diarization 3.1; модели скачиваются один раз и затем используются из локального каталога.

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

Invoke-WebRequest -Uri "http://localhost:8001/add-user" `
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
    summary.py           # local Ollama summary
    editor.py            # ИИ-редактура стенограммы
    diarization.py       # local pyannote diarization
  utils/files.py         # файловые helper'ы
  worker.py              # CLI/background pipeline
app.py                   # совместимый ASGI entrypoint
process_audio_fast.py    # совместимый CLI wrapper
data/
  uploads/               # входные файлы, не коммитить
  api_results/           # результаты задач, не коммитить
```

## Ollama на Ubuntu

Установка Ollama:

```bash
curl -fsSL https://ollama.com/install.sh | sh
sudo systemctl enable --now ollama
systemctl status ollama
```

Скачать локальные модели:

```bash
ollama pull gemma3:4b
ollama pull qwen3:8b
ollama list
```

Проверить, что модели отвечают:

```bash
ollama run gemma3:4b "Привет. Ответь одной короткой фразой."
ollama run qwen3:8b "Исправь текст: это тестовая строка без знаков препинания"
curl http://localhost:11434/api/tags
```

Чтобы Ollama был доступен не только этому проекту, но и другим backend-сервисам на сервере, запускайте Ollama как системный сервис и слушайте не только `127.0.0.1`. Для systemd:

```bash
sudo systemctl edit ollama
```

Добавьте:

```ini
[Service]
Environment="OLLAMA_HOST=0.0.0.0:11434"
```

Затем перезапустите сервис:

```bash
sudo systemctl daemon-reload
sudo systemctl restart ollama
ss -ltnp | grep 11434
```

Если сервер открыт в сеть, ограничьте порт firewall'ом, например разрешите доступ только с Docker-сети или внутренних адресов.

Из Docker-контейнеров и backend-сервисов обращайтесь к Ollama по HTTP:

```bash
curl http://host.docker.internal:11434/api/tags
curl http://<server-ip>:11434/api/tags
```

В Linux Docker `host.docker.internal` может потребовать настройку в `docker-compose.yml`:

```yaml
extra_hosts:
  - "host.docker.internal:host-gateway"
```

Для этого проекта в `.env` укажите адрес Ollama, доступный из процесса приложения:

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_API_KEY=ollama
SUMMARY_MODEL=gemma3:4b
EDITOR_MODEL=gemma3:4b
```

Если приложение запущено в Docker, обычно нужен один из вариантов:

```env
OLLAMA_BASE_URL=http://host.docker.internal:11434
```

или:

```env
OLLAMA_BASE_URL=http://<server-ip>:11434
```

Резюме и ИИ-редактура идут через локальные Ollama-модели из `SUMMARY_MODEL` и `EDITOR_MODEL`. Для локальных моделей через Ollama ключ не нужен; `OLLAMA_API_KEY=ollama` используется как техническое значение для OpenAI-compatible endpoint Ollama.

## WhisperX large-v3

В списке `Модель распознавания` доступен вариант `WhisperX large-v3`. Он использует пакет `whisperx==3.1.1`, `faster-whisper==1.2.1` и `transformers==4.39.3`, потому что эта связка совместима с текущими `torch==2.1.2`, `torchaudio==2.1.2`, `numpy<2` и `pyannote.audio==3.1.1`. Последние версии WhisperX/Transformers требуют более новую связку `torch`/`numpy`/`pyannote` и могут сломать текущую локальную диаризацию.

Настройки в `.env`:

```env
WHISPERX_MODEL=large-v3
WHISPERX_DEVICE=auto
WHISPERX_BATCH_SIZE=16
WHISPER_COMPUTE_TYPE=int8
WHISPER_LOCAL_FILES_ONLY=true
```

`WHISPERX_DEVICE=auto` выбирает CUDA, если она доступна, иначе CPU. На CPU `large-v3` может работать очень медленно; если памяти не хватает, уменьшите `WHISPERX_BATCH_SIZE`, например до `4` или `1`.

Для первого скачивания модели на сервере можно временно разрешить загрузку из Hugging Face:

```bash
sudo docker-compose run --rm api python -c "import importlib.metadata as m; print(m.version('whisperx'))"
```

```bash
sudo docker-compose run --rm \
  -e WHISPER_LOCAL_FILES_ONLY=false \
  api python -c "from audio_transcribator.config import settings; from audio_transcribator.services.transcription import resolve_auto_device; import whisperx; whisperx.load_model(settings.whisperx_model, resolve_auto_device(settings.whisperx_device), compute_type=settings.whisper_compute_type, language=settings.transcription_language, download_root=str(settings.model_cache_dir / 'whisperx')); print('WhisperX model cached')"
```

После того как модель окажется в `data/model_cache/whisperx`, верните `WHISPER_LOCAL_FILES_ONLY=true`. В runtime приложение будет работать из локального cache-каталога.

## Локальная диаризация pyannote

Диаризация остается опциональной: глобально она включается через `ENABLE_DIARIZATION=true`, а для каждой задачи отдельно через чекбокс `Включить диаризацию`.

Один раз скачайте модели на сервере. Для скачивания нужен Hugging Face token с принятыми условиями моделей `pyannote/speaker-diarization-3.1` и `pyannote/segmentation-3.0`; WeSpeaker embedding берется из открытого `hbredin/wespeaker-voxceleb-resnet34-LM`. После скачивания token приложению в runtime не нужен.

```bash
source .venv/bin/activate
python --version
pip install -r requirements.txt
export HF_TOKEN=hf_...
python scripts/download_pyannote_models.py --target-dir data/model_cache/pyannote
```

Для pyannote 3.1 используйте Python 3.11 и совместимую связку `torch==2.1.2`/`torchaudio==2.1.2` из `requirements.txt`. Dockerfile уже использует `python:3.11-slim`.

В `.env`:

```env
ENABLE_DIARIZATION=true
PYANNOTE_MODEL_DIR=/app/data/model_cache/pyannote
PYANNOTE_PIPELINE_CONFIG=/app/data/model_cache/pyannote/speaker-diarization-3.1/config.yaml
PYANNOTE_SEGMENTATION_MODEL=/app/data/model_cache/pyannote/segmentation-3.0
PYANNOTE_EMBEDDING_MODEL=/app/data/model_cache/pyannote/hbredin-wespeaker-voxceleb-resnet34-LM
PYANNOTE_DEVICE=auto
DIARIZATION_SPEAKERS=0
DIARIZATION_MIN_SPEAKERS=0
DIARIZATION_MAX_SPEAKERS=0
DIARIZATION_MIN_TURN_SECONDS=0.25
DIARIZATION_MIN_SPEAKER_RATIO=0.02
DIARIZATION_LOW_CONFIDENCE_RATIO=0.05
```

Для native-запуска вне Docker замените `/app/data/...` на абсолютный путь вашего проекта, например `/opt/audio-transcribator/data/...`.

`DIARIZATION_SPEAKERS=0`, `DIARIZATION_MIN_SPEAKERS=0` и `DIARIZATION_MAX_SPEAKERS=0` включают автооценку числа спикеров без системного ограничения сверху. Если известно точное число участников для конкретной задачи, укажите его в форме загрузки в поле `Точное число спикеров для диаризации`, например `2`; это передаст `num_speakers=2` только для этой обработки. Глобальный `DIARIZATION_SPEAKERS=2` используйте только если все задачи в установке всегда имеют ровно двух участников.

`DIARIZATION_MIN_TURN_SECONDS` отсекает слишком короткие шумовые фрагменты спикеров. `DIARIZATION_MIN_SPEAKER_RATIO` отсекает кластеры, которые занимают слишком маленькую долю речи. `DIARIZATION_LOW_CONFIDENCE_RATIO` помечает диаризацию как неуверенную в диагностике, если один из найденных спикеров слишком мал.

В runtime pipeline принудительно работает через локальный `config.yaml`; он указывает на локальные директории segmentation и embedding моделей. Скрытые обращения к Hugging Face отключаются через `HF_HUB_OFFLINE=1` при загрузке pyannote pipeline.

## Настройки

Основные переменные окружения задаются в `.env`:

- `API_TOKEN`, `API_USERNAME`, `API_PASSWORD`, `ADD_USER_ADMIN_TOKEN`
- `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `DATABASE_URL`
- `USER_STORAGE_QUOTA_BYTES`
- `DATA_RETENTION_DAYS`, `DATA_CLEANUP_INTERVAL_SECONDS`
- `OLLAMA_BASE_URL`, `OLLAMA_API_KEY`, `SUMMARY_MODEL`, `SUMMARY_CHUNK_CHARS`, `EDITOR_MODEL`, `EDITOR_CHUNK_CHARS`, `EDITOR_TEMPERATURE`
- `WHISPER_MODEL`, `WHISPER_COMPUTE_TYPE`, `WHISPER_LOCAL_FILES_ONLY`
- `TRANSCRIPTION_MODELS_FILE`
- `ENABLE_DIARIZATION`, `PYANNOTE_MODEL_DIR`, `PYANNOTE_PIPELINE_CONFIG`, `PYANNOTE_SEGMENTATION_MODEL`, `PYANNOTE_EMBEDDING_MODEL`, `PYANNOTE_DEVICE`
- `DIARIZATION_SPEAKERS`, `DIARIZATION_MIN_SPEAKERS`, `DIARIZATION_MAX_SPEAKERS`, `DIARIZATION_MIN_TURN_SECONDS`, `DIARIZATION_MIN_SPEAKER_RATIO`, `DIARIZATION_LOW_CONFIDENCE_RATIO`

