# Стенограмма

Сервис для обработки аудио и видео: извлекает аудио, делает транскрибацию, диаризацию по спикерам и краткое резюме текста.

Основной пайплайн:

- подготовка аудио через `ffmpeg`;
- транскрибация через локальный `WhisperX large-v3`;
- диаризация через локальный `pyannote.audio`;
- суммаризация через локальную Ollama-модель, например `qwen3:8b`;
- сохранение результатов в `data/api_results`.

## Возможности

- загрузка аудио и видео через веб-интерфейс;
- обработка файлов в фоне;
- просмотр статуса, логов и времени этапов;
- скачивание стенограммы, резюме и файлов диаризации;
- авторизация пользователей;
- benchmark для оценки качества и скорости диаризации;
- benchmark для оценки качества и скорости транскрибации.

## Быстрый запуск

Создайте `.env` на основе `.env.example`, затем запустите сервис:

```bash
sudo docker-compose up -d --build
```

Остановить сервис:

```bash
sudo docker-compose down
```

Посмотреть логи:

```bash
sudo docker-compose logs -f api
```

Веб-интерфейс:

```text
http://localhost:8001/ui
```

Benchmark-интерфейс:

```text
http://localhost:8001/ui/benchmark
```

## Основные URL

- `/ui` - загрузка файлов и просмотр результатов;
- `/ui/benchmark` - запуск benchmark диаризации и транскрибации;
- `/login` - получение токена API;
- `/process` - загрузка файла через API;
- `/result/{job_id}` - результат обработки;
- `/download/{job_id}/{filename}` - скачивание файла результата.

## Модели

Транскрибация:

```text
WhisperX large-v3
```

Диаризация:

```text
pyannote speaker-diarization 3.1
```

Суммаризация:

```text
Ollama, модель из SUMMARY_MODEL
```

Пример настроек `.env` для моделей внутри Docker:

```env
WHISPERX_MODEL=large-v3
WHISPERX_DEVICE=auto
WHISPER_COMPUTE_TYPE=int8
WHISPER_LOCAL_FILES_ONLY=true

ENABLE_DIARIZATION=true
PYANNOTE_MODEL_DIR=/app/data/model_cache/pyannote
PYANNOTE_PIPELINE_CONFIG=/app/data/model_cache/pyannote/speaker-diarization-3.1/config.yaml
PYANNOTE_SEGMENTATION_MODEL=/app/data/model_cache/pyannote/segmentation-3.0
PYANNOTE_EMBEDDING_MODEL=/app/data/model_cache/pyannote/hbredin-wespeaker-voxceleb-resnet34-LM
PYANNOTE_DEVICE=auto

OLLAMA_BASE_URL=http://host.docker.internal:11434
SUMMARY_MODEL=qwen3:8b
```

## Результаты обработки

Для каждой задачи создается папка:

```text
data/api_results/<job_id>
```

Основные файлы:

- `metadata.json` - статус, настройки задачи и время этапов;
- `run.log` - технический лог обработки;
- `stenogramma.txt` - распознанный текст;
- `transcript_segments.json` - сегменты транскрибации;
- `summary.txt` - резюме, если оно успешно создано;
- `partial_summary.txt` - частичное резюме, если суммаризация упала;
- `diarization.txt` - человекочитаемая разметка спикеров;
- `diarization.json` - сегменты диаризации;
- `diarization.rttm` - RTTM-файл для внешних инструментов;
- `diarized_transcript.txt` - стенограмма с привязкой к спикерам.

## Benchmark диаризации

Benchmark проверяет текущую диаризацию на датасете:

```text
ivkond/synthetic-speech-diarization-ru
```

Запуск на CPU:

```bash
python -m audio_transcribator.benchmarks.diarization --limit 10 --device cpu
```

Запуск на GPU:

```bash
python -m audio_transcribator.benchmarks.diarization --limit 10 --device cuda
```

Основные параметры:

- `--limit` - сколько файлов обработать;
- `--offset` - с какого файла начать;
- `--device` - `cpu`, `cuda` или `auto`;
- `--output-dir` - папка для отчетов;
- `--save-artifacts` - сохранить аудио, эталон и предсказания по каждому файлу.

Результаты:

- `per_file_metrics.csv` - метрики по каждому файлу;
- `aggregate_metrics.json` - агрегированные значения;
- `summary.md` - краткий отчет;
- `errors.jsonl` - ошибки обработки.

Считаются:

- `DER`;
- `JER`;
- `Miss`;
- `False Alarm`;
- `Speaker Confusion`;
- реальное и найденное число спикеров;
- длительность аудио;
- время обработки;
- `RTF`.

`DER` и `JER` считаются через `pyannote.metrics`. Ошибка одного файла не останавливает весь benchmark.

## Benchmark транскрибации

Benchmark транскрибации использует тот же датасет:

```text
ivkond/synthetic-speech-diarization-ru
```

Он сравнивает эталонный текст из датасета с текстом, который выдал текущий сервис транскрибации.

Запуск на CPU:

```bash
python -m audio_transcribator.benchmarks.transcription --limit 10 --device cpu
```

Запуск на GPU:

```bash
python -m audio_transcribator.benchmarks.transcription --limit 10 --device cuda
```

Результаты:

- `asr_per_file_metrics.csv` - метрики по каждому файлу;
- `asr_aggregate_metrics.json` - агрегированные значения;
- `asr_summary.md` - краткий отчет;
- `asr_errors.jsonl` - ошибки обработки.

Считаются:

- `WER`;
- `CER`;
- нормализованные `WER` и `CER`;
- длительность аудио;
- время обработки;
- `RTF`;
- статус и текст ошибки.

## Как читать RTF

`RTF` показывает, сколько времени занимает обработка относительно длительности аудио.

```text
RTF = время обработки / длительность аудио
```

Примеры:

- `RTF = 0.5` - один час аудио обработается примерно за 30 минут;
- `RTF = 1.0` - обработка идет примерно в реальном времени;
- `RTF = 3.0` - один час аудио обработается примерно за 3 часа.

## Структура проекта

```text
audio_transcribator/
  api/                 # API
  ui/                  # веб-интерфейс
  services/            # транскрибация, диаризация, суммаризация, jobs
  benchmarks/          # benchmark-модули
  templates/           # HTML-шаблоны
  static/              # CSS и статика
  worker.py            # фоновый обработчик задач

data/
  uploads/             # загруженные файлы
  api_results/         # результаты обработки
  model_cache/         # локальные модели
  diarization_benchmarks/ # отчеты benchmark из UI
```

## Тесты

```bash
pytest
```

## Важные замечания

- Директории `data/uploads`, `data/api_results` и `data/model_cache` не нужно коммитить.
- Для CPU-замеров не запускайте несколько тяжелых задач одновременно.
- Для корректного сравнения CPU и GPU запускайте одинаковые тесты с одинаковым `limit` и `offset`.
- Для длинных видео основной вклад во время обработки могут давать WhisperX, pyannote и локальная LLM-суммаризация.
