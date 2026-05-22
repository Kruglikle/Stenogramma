import json
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

from fastapi import UploadFile

from audio_transcribator.config import settings
from audio_transcribator.services.transcription_models import (
    DEFAULT_TRANSCRIPTION_MODEL_ID,
    resolve_transcription_model,
)
from audio_transcribator.utils.files import tail


STATUS_LABELS = {
    "started": "Запущено",
    "running": "В обработке",
    "completed": "Готово",
    "completed_without_summary": "Готово без резюме",
    "failed": "Ошибка",
}


def save_job_metadata(
    job_dir: Path,
    input_file: Path | str,
    status: str,
    transcription_model_id: str | None = None,
) -> None:
    existing_metadata = load_job_metadata(job_dir)
    metadata = {
        "job_id": job_dir.name,
        "status": status,
        "input_file": str(input_file),
        "transcription_model": transcription_model_id
        or existing_metadata.get("transcription_model")
        or DEFAULT_TRANSCRIPTION_MODEL_ID,
        "files": sorted(p.name for p in job_dir.iterdir() if p.is_file()),
    }
    with open(job_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def load_job_metadata(job_dir: Path) -> dict:
    metadata_file = job_dir / "metadata.json"
    if not metadata_file.exists():
        return {}

    try:
        return json.loads(metadata_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def resolve_status(metadata: dict, files: list[str], log_tail: str) -> str:
    status = metadata.get("status")
    if status == "completed" and "summary.txt" not in files:
        return "completed_without_summary"
    if status in STATUS_LABELS:
        return status
    if "failed" in log_tail.lower() or "traceback" in log_tail.lower():
        return "failed"
    if "transcript.txt" in files:
        return "completed_without_summary"
    return "running"


def build_job_result(job_id: str) -> dict:
    job_dir = settings.results_dir / job_id
    summary_file = job_dir / "summary.txt"
    transcript_file = job_dir / "transcript.txt"
    edited_transcript_file = job_dir / "edited_transcript.txt"
    log_file = job_dir / "run.log"

    if not job_dir.exists():
        raise FileNotFoundError("Job not found")

    files = [p.name for p in job_dir.iterdir() if p.is_file()]
    metadata = load_job_metadata(job_dir)
    log_tail = tail(log_file)
    status = resolve_status(metadata, files, log_tail)

    result = {
        "job_id": job_id,
        "files": files,
        "status": status,
        "status_label": STATUS_LABELS.get(status, status.title()),
    }

    if transcript_file.exists():
        result["transcript"] = transcript_file.read_text(encoding="utf-8", errors="replace")

    if edited_transcript_file.exists():
        result["edited_transcript"] = edited_transcript_file.read_text(encoding="utf-8", errors="replace")

    if summary_file.exists():
        result["summary"] = summary_file.read_text(encoding="utf-8", errors="replace")

    if log_tail:
        result["log_tail"] = log_tail

    return result


def get_job_file(job_id: str, filename: str) -> Path:
    job_dir = settings.results_dir / job_id
    file_path = job_dir / filename

    if not job_dir.exists():
        raise FileNotFoundError("Job not found")

    if not file_path.exists():
        raise FileNotFoundError("File not found")

    return file_path


def start_uploaded_file(file: UploadFile, transcription_model_id: str | None = None) -> dict:
    transcription_model = resolve_transcription_model(transcription_model_id)
    job_id = str(uuid4())
    job_dir = settings.results_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    safe_filename = file.filename or "upload"
    input_path = settings.upload_dir / f"{job_id}_{safe_filename}"

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    save_job_metadata(job_dir, input_path, status="started", transcription_model_id=transcription_model["id"])

    log_path = job_dir / "run.log"
    command = [
        sys.executable,
        "process_audio_fast.py",
        str(input_path),
        str(job_dir),
        "--transcription-model",
        transcription_model["id"],
    ]

    with open(log_path, "w", encoding="utf-8") as log_file:
        subprocess.Popen(
            command,
            cwd=str(settings.base_dir),
            stdout=log_file,
            stderr=log_file,
        )

    return {
        "status": "started",
        "job_id": job_id,
        "transcription_model": transcription_model["id"],
        "message": "File uploaded and processing started",
    }


def start_url(source_url: str, transcription_model_id: str | None = None) -> dict:
    parsed_url = urlparse(source_url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise ValueError("Only http/https media links are supported")

    transcription_model = resolve_transcription_model(transcription_model_id)
    job_id = str(uuid4())
    job_dir = settings.results_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    save_job_metadata(job_dir, source_url, status="started", transcription_model_id=transcription_model["id"])

    log_path = job_dir / "run.log"
    command = [
        sys.executable,
        "process_audio_fast.py",
        "remote-media",
        str(job_dir),
        "--source-url",
        source_url,
        "--transcription-model",
        transcription_model["id"],
    ]

    with open(log_path, "w", encoding="utf-8") as log_file:
        subprocess.Popen(
            command,
            cwd=str(settings.base_dir),
            stdout=log_file,
            stderr=log_file,
        )

    return {
        "status": "started",
        "job_id": job_id,
        "transcription_model": transcription_model["id"],
        "message": "Media URL queued and processing started",
    }
