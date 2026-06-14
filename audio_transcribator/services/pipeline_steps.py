import time
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from audio_transcribator.config import settings
from audio_transcribator.services.diarization import diarize
from audio_transcribator.services.editor import edit_transcript
from audio_transcribator.services.jobs import (
    ensure_user_storage_quota,
    load_job_metadata,
    save_job_metadata,
    save_job_timing,
)
from audio_transcribator.services.pipeline_progress import (
    ProgressPlan,
    run_progress_step as run_pipeline_progress_step,
)
from audio_transcribator.services.summary import summarize


T = TypeVar("T")


def timed_step(job_dir: Path, step: str, action: Callable[[], T], skipped: Callable[[T], bool] | None = None) -> T:
    """Выполнить этап пайплайна и сохранить его длительность в metadata.json."""
    started = time.perf_counter()
    try:
        result = action()
    except Exception:
        save_job_timing(job_dir, step, time.perf_counter() - started, status="failed")
        raise

    status = "skipped" if skipped and skipped(result) else "completed"
    save_job_timing(job_dir, step, time.perf_counter() - started, status=status)
    return result


def save_metadata(
    job_dir: Path,
    input_file: Path | str,
    status: str = "completed",
    transcription_model_id: str | None = None,
    enable_transcription: bool | None = None,
    enable_summary: bool | None = None,
    enable_diarization: bool | None = None,
    diarization_speakers: int | None = None,
) -> None:
    """Локальная обертка worker-а, сохраняющая существующую схему metadata.json."""
    save_job_metadata(
        job_dir,
        input_file,
        status,
        transcription_model_id=transcription_model_id,
        enable_transcription=enable_transcription,
        enable_summary=enable_summary,
        enable_diarization=enable_diarization,
        diarization_speakers=diarization_speakers,
    )


def maybe_diarize(
    audio_file: Path,
    job_dir: Path,
    enable_diarization: bool,
    diarization_speakers: int = 0,
    progress_plan: ProgressPlan | None = None,
) -> None:
    """Запустить диаризацию только если она включена в задаче и в общей конфигурации."""
    if not enable_diarization:
        return
    if not settings.enable_diarization:
        print("Diarization was requested but ENABLE_DIARIZATION is disabled.")
        save_job_timing(job_dir, "diarization", 0, status="skipped")
        return
    if progress_plan:
        run_pipeline_progress_step(
            job_dir,
            progress_plan,
            "diarization",
            lambda progress: diarize(
                audio_file,
                job_dir,
                diarization_speakers=diarization_speakers,
                progress_callback=progress,
            ),
            timed_step,
            skipped=lambda result: not result,
        )
        return

    timed_step(
        job_dir,
        "diarization",
        lambda: diarize(audio_file, job_dir, diarization_speakers=diarization_speakers),
        skipped=lambda result: not result,
    )


def maybe_summarize(
    transcript: str,
    job_dir: Path,
    enable_summary: bool,
    progress_plan: ProgressPlan | None = None,
) -> None:
    """Запустить опциональное локальное резюме и явно отметить пропуск этапа."""
    if not enable_summary:
        print("Summary was disabled for this job.")
        save_job_timing(job_dir, "summary", 0, status="skipped")
        return
    if progress_plan:
        run_pipeline_progress_step(
            job_dir,
            progress_plan,
            "summary",
            lambda progress: summarize(transcript, job_dir, progress_callback=progress),
            timed_step,
            skipped=lambda result: result is None,
        )
        return

    timed_step(job_dir, "summary", lambda: summarize(transcript, job_dir), skipped=lambda result: result is None)


def enforce_download_quota(job_dir: Path, input_file: Path) -> None:
    """Удалить скачанный файл, если его сохранение превышает квоту пользователя."""
    metadata = load_job_metadata(job_dir)
    user_login = metadata.get("user_login")
    if not user_login:
        return

    try:
        ensure_user_storage_quota(user_login, 0)
    except Exception:
        input_file.unlink(missing_ok=True)
        raise


def process_edit(job_dir: Path, editor_model: str | None = None, transcript_source: str = "transcript") -> None:
    """Фоновая точка входа для ИИ-редактуры уже готовой стенограммы."""
    transcript_file = job_dir / "diarized_transcript.txt" if transcript_source == "diarized" else job_dir / "stenogramma.txt"
    lock_path = job_dir / "editing.lock"
    if not transcript_file.exists():
        raise FileNotFoundError("Transcript is not ready")

    transcript = transcript_file.read_text(encoding="utf-8", errors="replace")
    started = time.perf_counter()
    try:
        edit_transcript(transcript, job_dir, model=editor_model)
        save_job_timing(job_dir, "editing", time.perf_counter() - started)
    except Exception as exc:
        save_job_timing(job_dir, "editing", time.perf_counter() - started, status="failed")
        (job_dir / "editing_error.txt").write_text(str(exc), encoding="utf-8")
        raise
    finally:
        lock_path.unlink(missing_ok=True)
