import json
from pathlib import Path

from fastapi import FastAPI


def test_entrypoint_files_exist() -> None:
    """Совместимые entrypoint-файлы должны оставаться на месте для Docker и старых команд."""
    root = Path(__file__).resolve().parents[1]

    assert (root / "app.py").is_file()
    assert (root / "process_audio_fast.py").is_file()
    assert (root / "Dockerfile").is_file()
    assert (root / "docker-compose.yml").is_file()


def test_settings_load_without_model_initialization() -> None:
    """Импорт настроек не должен скачивать модели или обращаться к внешним сервисам."""
    from audio_transcribator.config import settings

    assert settings.base_dir.exists()
    assert settings.upload_dir.name == "uploads"
    assert settings.results_dir.name == "api_results"
    assert settings.model_cache_dir.name == "model_cache"


def test_fastapi_app_import_does_not_start_server() -> None:
    """Импорт app.py должен создать ASGI-объект без запуска uvicorn."""
    from app import app

    assert isinstance(app, FastAPI)


def test_worker_entrypoint_importable() -> None:
    """process_audio_fast.py остается совместимой оберткой над worker-ом."""
    from process_audio_fast import main

    assert callable(main)


def test_core_modules_importable() -> None:
    """Основные service-модули должны импортироваться без загрузки больших ML-моделей."""
    import audio_transcribator.services.audio
    import audio_transcribator.services.diarization
    import audio_transcribator.services.job_formatting
    import audio_transcribator.services.job_launch
    import audio_transcribator.services.job_metadata
    import audio_transcribator.services.job_results
    import audio_transcribator.services.job_storage
    import audio_transcribator.services.jobs
    import audio_transcribator.services.pipeline_steps
    import audio_transcribator.services.pipeline_progress
    import audio_transcribator.services.progress
    import audio_transcribator.services.summary
    import audio_transcribator.services.transcription

    assert audio_transcribator.services.jobs.STATUS_LABELS


def test_transcription_models_default_to_whisperx() -> None:
    """Пользовательский выбор распознавания сведен к локальному WhisperX."""
    from audio_transcribator.services.transcription_models import (
        DEFAULT_TRANSCRIPTION_MODEL_ID,
        list_transcription_models,
    )

    models = list_transcription_models()
    assert DEFAULT_TRANSCRIPTION_MODEL_ID == "local:whisperx-large"
    assert [model["id"] for model in models] == ["local:whisperx-large"]


def test_job_queue_dispatches_fifo_with_concurrency_limit(tmp_path, monkeypatch) -> None:
    from audio_transcribator.config import settings
    from audio_transcribator.services import job_queue

    def write_job(name: str, status: str, started_at: str) -> Path:
        job_dir = tmp_path / name
        job_dir.mkdir()
        (job_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "job_id": name,
                    "status": status,
                    "input_file": str(tmp_path / f"{name}.wav"),
                    "started_at": started_at,
                    "transcription_model": "local:whisperx-large",
                    "enable_transcription": True,
                    "enable_summary": False,
                    "enable_diarization": False,
                }
            ),
            encoding="utf-8",
        )
        return job_dir

    write_job("active", "running", "2026-01-01T00:00:00+00:00")
    write_job("newer", "queued", "2026-01-01T00:02:00+00:00")
    write_job("older", "queued", "2026-01-01T00:01:00+00:00")
    launched = []

    def fake_launch(job_dir: Path, metadata: dict) -> None:
        launched.append(job_dir.name)

    monkeypatch.setattr(settings, "results_dir", tmp_path)
    monkeypatch.setattr(settings, "max_concurrent_jobs", 2)
    monkeypatch.setattr(job_queue, "launch_worker_for_job", fake_launch)

    assert job_queue.dispatch_queued_jobs() == ["older"]
    assert launched == ["older"]
