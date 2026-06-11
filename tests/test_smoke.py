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
    import audio_transcribator.services.jobs
    import audio_transcribator.services.pipeline_progress
    import audio_transcribator.services.progress
    import audio_transcribator.services.summary
    import audio_transcribator.services.transcription

    assert audio_transcribator.services.jobs.STATUS_LABELS
