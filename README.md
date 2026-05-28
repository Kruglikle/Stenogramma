# Стенограмма

FastAPI-сервис для обработки аудио- и видеофайлов: загрузка файла, извлечение аудио через `ffmpeg`, транскрибация через `faster-whisper`, генерация summary через LLM и скачивание результатов.

## Возможности

- `POST /login` — авторизация и получение access token.
- `POST /add-user` — добавление нового пользователя администратором.
- `POST /process` — загрузка аудио/видео и запуск фоновой обработки.
- `GET /transcription-models` — список доступных моделей транскрибации.
- `GET /editor-models` — список доступных моделей ИИ-редактуры.
- `GET /result/{job_id}` — получение результата обработки по ID задачи.
- `POST /result/{job_id}/edit` — ИИ-редактура готовой стенограммы через OpenRouter или локальную Ollama-модель.
- `GET /download/{job_id}/{filename}` — скачивание отдельных файлов результата.
- Token-based авторизация через `Authorization: Bearer <token>`.
- Пользователи хранятся в PostgreSQL, пароль сохраняется в виде hash.
- Поддержка аудио и видеоформатов, совместимых с `ffmpeg`.
- Выбор модели транскрибации: локальная `faster-whisper` или OpenRouter STT модели из `audio_transcribator/transcription_models.json`.
- Выбор модели ИИ-редактуры: локальные `gemma3:4b`, `qwen3:8b` через Ollama или `qwen/qwen3.6-35b-a3b` через OpenRouter.
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
EDITOR_MODEL=qwen/qwen3.6-35b-a3b
OPENROUTER_API_KEY=
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

Если приложение запущено в Docker, обычно нужен один из вариантов:

```env
OLLAMA_BASE_URL=http://host.docker.internal:11434
```

или:

```env
OLLAMA_BASE_URL=http://<server-ip>:11434
```

Для OpenRouter-редактуры заполните `OPENROUTER_API_KEY`. Для локальных моделей через Ollama ключ не нужен; `OLLAMA_API_KEY=ollama` используется как техническое значение для OpenAI-compatible endpoint Ollama.

## Настройки

Основные переменные окружения задаются в `.env`:

- `API_TOKEN`, `API_USERNAME`, `API_PASSWORD`, `ADD_USER_ADMIN_TOKEN`
- `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `DATABASE_URL`
- `OPENROUTER_API_KEY`, `OPENROUTER_BASE_URL`, `OLLAMA_BASE_URL`, `OLLAMA_API_KEY`, `SUMMARY_MODEL`, `EDITOR_MODEL`, `EDITOR_TEMPERATURE`
- `WHISPER_MODEL`, `WHISPER_COMPUTE_TYPE`
- `TRANSCRIPTION_MODELS_FILE`
- `ENABLE_DIARIZATION`, `HF_TOKEN`

