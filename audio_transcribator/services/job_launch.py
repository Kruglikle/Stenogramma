import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

from fastapi import UploadFile

from audio_transcribator.config import settings
from audio_transcribator.services.job_metadata import save_job_metadata, save_job_timing
from audio_transcribator.services.job_storage import ensure_user_storage_quota, get_upload_size
from audio_transcribator.services.transcription_models import resolve_transcription_model


def start_uploaded_file(
    file: UploadFile,
    transcription_model_id: str | None = None,
    user_login: str | None = None,
    title: str | None = None,
    enable_transcription: bool = True,
    enable_summary: bool = True,
    enable_diarization: bool = True,
    diarization_speakers: int = 0,
) -> dict:
    """Сохранить upload-файл и запустить обработку в отдельном процессе."""
    transcription_model = resolve_transcription_model(transcription_model_id)
    _validate_requested_steps(enable_transcription, enable_diarization, enable_summary)
    upload_size = get_upload_size(file)
    ensure_user_storage_quota(user_login, upload_size)

    job_id = str(uuid4())
    job_dir = settings.results_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    safe_filename = file.filename or "upload"
    input_path = settings.upload_dir / f"{job_id}_{safe_filename}"

    started = time.perf_counter()
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    save_job_metadata(
        job_dir,
        input_path,
        status="started",
        transcription_model_id=transcription_model["id"],
        user_login=user_login,
        title=title or safe_filename,
        enable_transcription=enable_transcription,
        enable_summary=enable_summary,
        enable_diarization=enable_diarization,
        diarization_speakers=diarization_speakers,
    )
    save_job_timing(job_dir, "upload", time.perf_counter() - started)
    _launch_worker(
        job_dir,
        _build_worker_command(
            input_arg=str(input_path),
            job_dir=job_dir,
            transcription_model_id=transcription_model["id"],
            enable_transcription=enable_transcription,
            enable_summary=enable_summary,
            enable_diarization=enable_diarization,
            diarization_speakers=diarization_speakers,
        ),
    )

    return _start_response(
        job_id,
        transcription_model["id"],
        enable_transcription,
        enable_summary,
        enable_diarization,
        diarization_speakers,
        "File uploaded and processing started",
    )


def start_url(
    source_url: str,
    transcription_model_id: str | None = None,
    user_login: str | None = None,
    title: str | None = None,
    enable_transcription: bool = True,
    enable_summary: bool = True,
    enable_diarization: bool = True,
    diarization_speakers: int = 0,
) -> dict:
    """Создать задачу для внешней http/https-ссылки и запустить worker."""
    parsed_url = urlparse(source_url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise ValueError("Only http/https media links are supported")

    transcription_model = resolve_transcription_model(transcription_model_id)
    _validate_requested_steps(enable_transcription, enable_diarization, enable_summary)
    ensure_user_storage_quota(user_login, 0)

    job_id = str(uuid4())
    job_dir = settings.results_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    save_job_metadata(
        job_dir,
        source_url,
        status="started",
        transcription_model_id=transcription_model["id"],
        user_login=user_login,
        title=title or source_url,
        enable_transcription=enable_transcription,
        enable_summary=enable_summary,
        enable_diarization=enable_diarization,
        diarization_speakers=diarization_speakers,
    )
    _launch_worker(
        job_dir,
        _build_worker_command(
            input_arg="remote-media",
            job_dir=job_dir,
            transcription_model_id=transcription_model["id"],
            enable_transcription=enable_transcription,
            enable_summary=enable_summary,
            enable_diarization=enable_diarization,
            diarization_speakers=diarization_speakers,
            source_url=source_url,
        ),
    )

    return _start_response(
        job_id,
        transcription_model["id"],
        enable_transcription,
        enable_summary,
        enable_diarization,
        diarization_speakers,
        "Media URL queued and processing started",
    )


def _validate_requested_steps(enable_transcription: bool, enable_diarization: bool, enable_summary: bool) -> None:
    """Проверить комбинацию этапов перед созданием задачи."""
    if not any((enable_transcription, enable_diarization, enable_summary)):
        raise ValueError("At least one processing step must be enabled")
    if enable_summary and not enable_transcription:
        raise ValueError("Summary requires transcription to be enabled")


def _build_worker_command(
    input_arg: str,
    job_dir: Path,
    transcription_model_id: str,
    enable_transcription: bool,
    enable_summary: bool,
    enable_diarization: bool,
    diarization_speakers: int,
    source_url: str | None = None,
) -> list[str]:
    """Собрать команду process_audio_fast.py, сохраняя старый CLI-контракт."""
    command = [
        sys.executable,
        "process_audio_fast.py",
        input_arg,
        str(job_dir),
    ]
    if source_url:
        command.extend(["--source-url", source_url])
    command.extend(["--transcription-model", transcription_model_id])
    if not enable_transcription:
        command.append("--no-transcription")
    if not enable_summary:
        command.append("--no-summary")
    if enable_diarization:
        command.append("--diarization")
        if diarization_speakers > 0:
            command.extend(["--diarization-speakers", str(diarization_speakers)])
    return command


def _launch_worker(job_dir: Path, command: list[str]) -> None:
    """Запустить обработчик в фоне и направить stdout/stderr в run.log."""
    with open(job_dir / "run.log", "w", encoding="utf-8") as log_file:
        subprocess.Popen(
            command,
            cwd=str(settings.base_dir),
            stdout=log_file,
            stderr=log_file,
        )


def _start_response(
    job_id: str,
    transcription_model_id: str,
    enable_transcription: bool,
    enable_summary: bool,
    enable_diarization: bool,
    diarization_speakers: int,
    message: str,
) -> dict:
    """Вернуть тот же формат ответа, который ожидают API и UI."""
    return {
        "status": "started",
        "job_id": job_id,
        "transcription_model": transcription_model_id,
        "enable_transcription": enable_transcription,
        "enable_summary": enable_summary,
        "enable_diarization": enable_diarization,
        "diarization_speakers": diarization_speakers,
        "message": message,
    }
