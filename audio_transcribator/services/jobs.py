import json
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

from fastapi import UploadFile

from audio_transcribator.config import settings
from audio_transcribator.services.transcription_models import (
    DEFAULT_TRANSCRIPTION_MODEL_ID,
    resolve_transcription_model,
)
from audio_transcribator.utils.files import tail, write_text_atomic


STATUS_LABELS = {
    "started": "Запущено",
    "running": "В обработке",
    "completed": "Готово",
    "completed_without_summary": "Готово без резюме",
    "failed": "Ошибка",
}

TIMING_LABELS = {
    "upload": "Загрузка файла",
    "download": "Скачивание медиа",
    "prepare_audio": "Подготовка аудио",
    "transcription": "Транскрибация",
    "diarization": "Диаризация",
    "summary": "Резюме",
    "editing": "ИИ-редактура",
}
TIMING_STATUS_LABELS = {
    "completed": "готово",
    "failed": "ошибка",
    "skipped": "пропущено",
}

PROGRESS_LABELS = {
    "queued": "Задача поставлена в очередь",
    "upload": "Загрузка файла",
    "download": "Скачивание медиа",
    "prepare_audio": "Подготовка аудио",
    "transcription": "Транскрибация",
    "diarization": "Диаризация",
    "summary": "Резюме",
    "editing": "ИИ-редактура",
    "completed": "Готово",
    "failed": "Ошибка",
}


def utc_now_iso() -> str: 
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def format_duration(seconds: float | int | None) -> str:
    if seconds is None:
        return ""

    total_seconds = int(round(float(seconds)))
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)

    if hours:
        return f"{hours} ч {minutes} мин {seconds} сек"
    if minutes:
        return f"{minutes} мин {seconds} сек"
    return f"{seconds} сек"


def format_bytes(size_bytes: int) -> str:
    value = float(size_bytes)
    for unit in ("Б", "КБ", "МБ", "ГБ", "ТБ"):
        if value < 1024 or unit == "ТБ":
            return f"{value:.1f} {unit}" if unit != "Б" else f"{int(value)} {unit}"
        value /= 1024
    return f"{size_bytes} Б"


def path_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


class StorageQuotaExceeded(ValueError):
    pass


def clean_job_title(value: str | None, fallback: str) -> str:
    title = " ".join((value or "").strip().split())
    if not title:
        title = fallback
    return title[:160]


def get_upload_size(file: UploadFile) -> int:
    size = getattr(file, "size", None)
    if isinstance(size, int) and size >= 0:
        return size

    current_position = file.file.tell()
    file.file.seek(0, 2)
    size = file.file.tell()
    file.file.seek(current_position)
    return size


def list_user_job_dirs(user_login: str) -> list[Path]:
    if not settings.results_dir.exists():
        return []

    job_dirs = []
    for job_dir in settings.results_dir.iterdir():
        if not job_dir.is_dir():
            continue
        metadata = load_job_metadata(job_dir)
        if metadata.get("user_login") == user_login:
            job_dirs.append(job_dir)
    return job_dirs


def user_storage_usage_bytes(user_login: str) -> int:
    total = 0
    counted_job_dirs = []
    for job_dir in list_user_job_dirs(user_login):
        counted_job_dirs.append(job_dir.resolve())
        total += path_size(job_dir)

        metadata = load_job_metadata(job_dir)
        input_file = metadata.get("input_file")
        if not input_file or str(input_file).startswith(("http://", "https://")):
            continue

        input_path = Path(input_file)
        if not input_path.is_absolute():
            input_path = (settings.base_dir / input_path).resolve()
        else:
            input_path = input_path.resolve()

        if any(is_relative_to(input_path, counted_dir) for counted_dir in counted_job_dirs):
            continue
        if is_relative_to(input_path, settings.upload_dir):
            total += path_size(input_path)

    return total


def user_storage_quota(user_login: str) -> dict:
    used = user_storage_usage_bytes(user_login)
    limit = settings.user_storage_quota_bytes
    remaining = max(limit - used, 0)
    return {
        "used_bytes": used,
        "limit_bytes": limit,
        "remaining_bytes": remaining,
        "used": format_bytes(used),
        "limit": format_bytes(limit),
        "remaining": format_bytes(remaining),
        "percent": round((used / limit) * 100, 1) if limit else 0,
    }


def ensure_user_storage_quota(user_login: str | None, incoming_bytes: int = 0) -> None:
    if not user_login or settings.user_storage_quota_bytes <= 0:
        return

    quota = user_storage_quota(user_login)
    if quota["used_bytes"] + incoming_bytes <= quota["limit_bytes"]:
        return

    raise StorageQuotaExceeded(
        "Недостаточно места в кабинете: занято "
        f"{quota['used']} из {quota['limit']}, новый файл {format_bytes(incoming_bytes)}."
    )


def save_job_metadata(
    job_dir: Path,
    input_file: Path | str,
    status: str,
    transcription_model_id: str | None = None,
    user_login: str | None = None,
    title: str | None = None,
    enable_summary: bool | None = None,
    enable_diarization: bool | None = None,
    diarization_speakers: int | None = None,
) -> None:
    existing_metadata = load_job_metadata(job_dir)
    started_at = existing_metadata.get("started_at") or utc_now_iso()
    input_value = str(input_file)
    fallback_title = input_value if input_value.startswith(("http://", "https://")) else Path(input_value).name
    metadata = {
        "job_id": job_dir.name,
        "status": status,
        "input_file": input_value,
        "title": clean_job_title(title or existing_metadata.get("title"), fallback_title),
        "user_login": user_login or existing_metadata.get("user_login"),
        "started_at": started_at,
        "transcription_model": transcription_model_id
        or existing_metadata.get("transcription_model")
        or DEFAULT_TRANSCRIPTION_MODEL_ID,
        "enable_summary": enable_summary if enable_summary is not None else existing_metadata.get("enable_summary", True),
        "enable_diarization": enable_diarization
        if enable_diarization is not None
        else existing_metadata.get("enable_diarization", False),
        "diarization_speakers": diarization_speakers
        if diarization_speakers is not None
        else existing_metadata.get("diarization_speakers", 0),
        "timings": existing_metadata.get("timings", {}),
        "files": sorted(p.name for p in job_dir.iterdir() if p.is_file()),
    }
    if status in {"completed", "completed_without_summary", "failed"}:
        metadata["finished_at"] = existing_metadata.get("finished_at") or utc_now_iso()
        started = parse_iso_datetime(started_at)
        finished = parse_iso_datetime(metadata["finished_at"])
        if started and finished:
            metadata["total_seconds"] = max((finished - started).total_seconds(), 0)
    with open(job_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def save_job_timing(job_dir: Path, step: str, elapsed_seconds: float, status: str = "completed") -> None:
    metadata = load_job_metadata(job_dir)
    timings = metadata.setdefault("timings", {})
    timings[step] = {
        "label": TIMING_LABELS.get(step, step),
        "seconds": round(elapsed_seconds, 3),
        "duration": format_duration(elapsed_seconds),
        "status": status,
        "finished_at": utc_now_iso(),
    }
    metadata["files"] = sorted(p.name for p in job_dir.iterdir() if p.is_file())
    with open(job_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def save_job_progress(
    job_dir: Path,
    percent: float,
    step: str,
    label: str | None = None,
    detail: str | None = None,
    status: str = "running",
) -> None:
    normalized = max(0, min(100, round(float(percent), 1)))
    payload = {
        "percent": normalized,
        "step": step,
        "label": label or PROGRESS_LABELS.get(step, step),
        "detail": detail or "",
        "status": status,
        "updated_at": utc_now_iso(),
    }
    write_text_atomic(job_dir / "progress.json", json.dumps(payload, ensure_ascii=False, indent=2))


def load_job_progress(job_dir: Path, status: str) -> dict:
    progress_file = job_dir / "progress.json"
    if progress_file.exists():
        try:
            progress = json.loads(progress_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            progress = {}
    else:
        progress = {}

    if status in {"completed", "completed_without_summary"}:
        progress.update(
            {
                "percent": 100,
                "step": "completed",
                "label": PROGRESS_LABELS["completed"],
                "detail": "",
                "status": "completed",
            }
        )
    elif status == "failed":
        progress.update(
            {
                "percent": progress.get("percent", 0),
                "step": progress.get("step", "failed"),
                "label": PROGRESS_LABELS["failed"],
                "detail": progress.get("detail", ""),
                "status": "failed",
            }
        )

    return {
        "percent": progress.get("percent", 0),
        "step": progress.get("step", "queued"),
        "label": progress.get("label", PROGRESS_LABELS["queued"]),
        "detail": progress.get("detail", ""),
        "status": progress.get("status", status),
    }


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
    if status == "completed" and metadata.get("enable_summary", True) and "summary.txt" not in files:
        return "completed_without_summary"
    if status in STATUS_LABELS:
        return status
    if "failed" in log_tail.lower() or "traceback" in log_tail.lower():
        return "failed"
    if "stenogramma.txt" in files or "transcript.txt" in files:
        return "completed_without_summary"
    return "running"


def build_job_result(job_id: str) -> dict:
    job_dir = settings.results_dir / job_id
    summary_file = job_dir / "summary.txt"
    diarization_file = job_dir / "diarization.txt"
    diarized_transcript_file = job_dir / "diarized_transcript.txt"
    transcript_file = job_dir / "stenogramma.txt"
    legacy_transcript_file = job_dir / "transcript.txt"
    edited_transcript_file = job_dir / "edited_transcript.txt"
    editing_error_file = job_dir / "editing_error.txt"
    editing_lock_file = job_dir / "editing.lock"
    log_file = job_dir / "run.log"

    if not job_dir.exists():
        raise FileNotFoundError("Job not found")

    files = [p.name for p in job_dir.iterdir() if p.is_file()]
    metadata = load_job_metadata(job_dir)
    log_tail = tail(log_file)
    status = resolve_status(metadata, files, log_tail)

    result = {
        "job_id": job_id,
        "title": metadata.get("title") or job_id,
        "user_login": metadata.get("user_login"),
        "files": files,
        "status": status,
        "status_label": STATUS_LABELS.get(status, status.title()),
        "enable_summary": metadata.get("enable_summary", True),
        "enable_diarization": metadata.get("enable_diarization", False),
        "timings": build_timing_result(metadata),
        "progress": load_job_progress(job_dir, status),
    }

    if transcript_file.exists():
        result["transcript"] = transcript_file.read_text(encoding="utf-8", errors="replace")
    elif legacy_transcript_file.exists():
        result["transcript"] = legacy_transcript_file.read_text(encoding="utf-8", errors="replace")

    if edited_transcript_file.exists():
        result["edited_transcript"] = edited_transcript_file.read_text(encoding="utf-8", errors="replace")
        result["editing_timing"] = result["timings"].get("by_step", {}).get("editing")

    if editing_lock_file.exists():
        result["editing_in_progress"] = True

    if editing_error_file.exists():
        result["editor_error"] = editing_error_file.read_text(encoding="utf-8", errors="replace")

    if summary_file.exists():
        result["summary"] = summary_file.read_text(encoding="utf-8", errors="replace")

    if diarization_file.exists():
        result["diarization"] = diarization_file.read_text(encoding="utf-8", errors="replace")

    if diarized_transcript_file.exists():
        result["diarized_transcript"] = diarized_transcript_file.read_text(encoding="utf-8", errors="replace")

    if log_tail:
        result["log_tail"] = log_tail

    return result


def choose_history_download(files: list[str]) -> str | None:
    for filename in ("edited_transcript.txt", "diarized_transcript.txt", "stenogramma.txt", "summary.txt"):
        if filename in files:
            return filename
    return None


def list_user_jobs(user_login: str, limit: int = 40) -> list[dict]:
    if not settings.results_dir.exists():
        return []

    jobs = []
    for job_dir in settings.results_dir.iterdir():
        if not job_dir.is_dir():
            continue

        metadata = load_job_metadata(job_dir)
        if metadata.get("user_login") != user_login:
            continue

        files = metadata.get("files")
        if not isinstance(files, list):
            files = [p.name for p in job_dir.iterdir() if p.is_file()]

        status = resolve_status(metadata, files, tail(job_dir / "run.log", lines=8))
        jobs.append(
            {
                "job_id": job_dir.name,
                "title": metadata.get("title") or job_dir.name,
                "status": status,
                "status_label": STATUS_LABELS.get(status, status.title()),
                "started_at": metadata.get("started_at") or "",
                "finished_at": metadata.get("finished_at") or "",
                "download_file": choose_history_download(files),
            }
        )

    jobs.sort(key=lambda item: item.get("started_at") or "", reverse=True)
    return jobs[:limit]


def update_job_title(job_id: str, title: str) -> dict:
    job_dir = settings.results_dir / job_id
    if not job_dir.exists():
        raise FileNotFoundError("Job not found")

    metadata = load_job_metadata(job_dir)
    metadata["title"] = clean_job_title(title, metadata.get("title") or job_id)
    metadata["files"] = sorted(p.name for p in job_dir.iterdir() if p.is_file())
    with open(job_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    return metadata


def delete_job(job_id: str) -> None:
    job_dir = settings.results_dir / job_id
    if not job_dir.exists():
        raise FileNotFoundError("Job not found")

    metadata = load_job_metadata(job_dir)
    input_file = metadata.get("input_file")
    if input_file and not str(input_file).startswith(("http://", "https://")):
        input_path = Path(input_file)
        if not input_path.is_absolute():
            input_path = (settings.base_dir / input_path).resolve()
        else:
            input_path = input_path.resolve()

        if is_relative_to(input_path, settings.upload_dir):
            input_path.unlink(missing_ok=True)

    shutil.rmtree(job_dir)


def cleanup_expired_jobs() -> list[str]:
    if settings.data_retention_days <= 0 or not settings.results_dir.exists():
        return []

    cutoff = datetime.now(UTC) - timedelta(days=settings.data_retention_days)
    deleted_job_ids = []
    for job_dir in settings.results_dir.iterdir():
        if not job_dir.is_dir():
            continue

        metadata = load_job_metadata(job_dir)
        created_at = parse_iso_datetime(metadata.get("started_at") or metadata.get("finished_at"))
        if created_at is None:
            created_at = datetime.fromtimestamp(job_dir.stat().st_mtime, tz=UTC)

        if created_at > cutoff:
            continue

        delete_job(job_dir.name)
        deleted_job_ids.append(job_dir.name)

    if deleted_job_ids:
        print(f"Deleted expired jobs: {', '.join(deleted_job_ids)}", flush=True)
    return deleted_job_ids


def build_timing_result(metadata: dict) -> dict:
    timings = metadata.get("timings") or {}
    ordered_steps = [
        "upload",
        "download",
        "prepare_audio",
        "transcription",
        "diarization",
        "summary",
        "editing",
    ]
    items = []
    for step in ordered_steps:
        timing = timings.get(step)
        if not timing:
            continue
        seconds = timing.get("seconds")
        items.append(
            {
                "step": step,
                "label": timing.get("label") or TIMING_LABELS.get(step, step),
                "seconds": seconds,
                "duration": timing.get("duration") or format_duration(seconds),
                "status": timing.get("status", "completed"),
                "status_label": TIMING_STATUS_LABELS.get(timing.get("status", "completed"), timing.get("status")),
            }
        )

    total_seconds = metadata.get("total_seconds")
    item_total_seconds = sum(float(item.get("seconds") or 0) for item in items)
    if total_seconds is None and items:
        total_seconds = item_total_seconds
    elif total_seconds is not None:
        total_seconds = max(float(total_seconds), item_total_seconds)

    return {
        "items": items,
        "by_step": {item["step"]: item for item in items},
        "total_seconds": total_seconds,
        "total_duration": format_duration(total_seconds),
    }


def get_job_file(job_id: str, filename: str) -> Path:
    job_dir = settings.results_dir / job_id
    file_path = job_dir / filename

    if not job_dir.exists():
        raise FileNotFoundError("Job not found")

    if not file_path.exists():
        raise FileNotFoundError("File not found")

    return file_path


def start_uploaded_file(
    file: UploadFile,
    transcription_model_id: str | None = None,
    user_login: str | None = None,
    title: str | None = None,
    enable_summary: bool = True,
    enable_diarization: bool = False,
    diarization_speakers: int = 0,
) -> dict:
    transcription_model = resolve_transcription_model(transcription_model_id)
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
        enable_summary=enable_summary,
        enable_diarization=enable_diarization,
        diarization_speakers=diarization_speakers,
    )
    save_job_timing(job_dir, "upload", time.perf_counter() - started)

    log_path = job_dir / "run.log"
    command = [
        sys.executable,
        "process_audio_fast.py",
        str(input_path),
        str(job_dir),
        "--transcription-model",
        transcription_model["id"],
    ]
    if not enable_summary:
        command.append("--no-summary")
    if enable_diarization:
        command.append("--diarization")
        if diarization_speakers > 0:
            command.extend(["--diarization-speakers", str(diarization_speakers)])

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
        "enable_summary": enable_summary,
        "enable_diarization": enable_diarization,
        "diarization_speakers": diarization_speakers,
        "message": "File uploaded and processing started",
    }


def start_url(
    source_url: str,
    transcription_model_id: str | None = None,
    user_login: str | None = None,
    title: str | None = None,
    enable_summary: bool = True,
    enable_diarization: bool = False,
    diarization_speakers: int = 0,
) -> dict:
    parsed_url = urlparse(source_url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise ValueError("Only http/https media links are supported")

    transcription_model = resolve_transcription_model(transcription_model_id)
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
        enable_summary=enable_summary,
        enable_diarization=enable_diarization,
        diarization_speakers=diarization_speakers,
    )

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
    if not enable_summary:
        command.append("--no-summary")
    if enable_diarization:
        command.append("--diarization")
        if diarization_speakers > 0:
            command.extend(["--diarization-speakers", str(diarization_speakers)])

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
        "enable_summary": enable_summary,
        "enable_diarization": enable_diarization,
        "diarization_speakers": diarization_speakers,
        "message": "Media URL queued and processing started",
    }
