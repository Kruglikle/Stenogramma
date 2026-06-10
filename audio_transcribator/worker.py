import argparse
import time
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from audio_transcribator.config import settings
from audio_transcribator.services.audio import download_media, prepare_audio
from audio_transcribator.services.diarization import diarize
from audio_transcribator.services.editor import edit_transcript
from audio_transcribator.services.jobs import (
    ensure_user_storage_quota,
    load_job_progress,
    load_job_metadata,
    save_job_metadata,
    save_job_progress,
    save_job_timing,
)
from audio_transcribator.services.summary import summarize
from audio_transcribator.services.transcription import transcribe
from audio_transcribator.services.transcription_models import DEFAULT_TRANSCRIPTION_MODEL_ID


T = TypeVar("T")

STEP_WEIGHTS = {
    "download": 10,
    "prepare_audio": 10,
    "transcription": 45,
    "diarization": 25,
    "summary": 10,
}

STEP_LABELS = {
    "download": "Скачивание медиа",
    "prepare_audio": "Подготовка аудио",
    "transcription": "Транскрибация",
    "diarization": "Диаризация",
    "summary": "Резюме",
}


def build_progress_plan(source_is_url: bool, enable_diarization: bool, enable_summary: bool) -> dict[str, tuple[float, float]]:
    steps = []
    if source_is_url:
        steps.append("download")
    steps.extend(["prepare_audio", "transcription"])
    if enable_diarization:
        steps.append("diarization")
    if enable_summary:
        steps.append("summary")

    total_weight = sum(STEP_WEIGHTS[step] for step in steps) or 1
    current = 0.0
    plan = {}
    for step in steps:
        width = STEP_WEIGHTS[step] / total_weight * 100
        plan[step] = (current, width)
        current += width
    return plan


def update_progress(
    job_dir: Path,
    plan: dict[str, tuple[float, float]],
    step: str,
    stage_percent: float,
    detail: str | None = None,
) -> None:
    start, width = plan.get(step, (0, 0))
    overall = start + width * max(0, min(100, stage_percent)) / 100
    save_job_progress(job_dir, overall, step, label=STEP_LABELS.get(step), detail=detail)


def run_progress_step(
    job_dir: Path,
    plan: dict[str, tuple[float, float]],
    step: str,
    action: Callable[[Callable[[float, str | None], None]], T],
    skipped: Callable[[T], bool] | None = None,
) -> T:
    update_progress(job_dir, plan, step, 0)

    def stage_progress(percent: float, detail: str | None = None) -> None:
        update_progress(job_dir, plan, step, percent, detail=detail)

    result = timed_step(job_dir, step, lambda: action(stage_progress), skipped=skipped)
    update_progress(job_dir, plan, step, 100)
    return result


def mark_progress_failed(job_dir: Path, detail: str | None = None) -> None:
    current = load_job_progress(job_dir, "failed")
    save_job_progress(
        job_dir,
        current.get("percent", 0),
        current.get("step", "failed"),
        detail=detail or current.get("detail", ""),
        status="failed",
    )


def timed_step(job_dir: Path, step: str, action: Callable[[], T], skipped: Callable[[T], bool] | None = None) -> T:
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
    enable_summary: bool | None = None,
    enable_diarization: bool | None = None,
    diarization_speakers: int | None = None,
) -> None:
    save_job_metadata(
        job_dir,
        input_file,
        status,
        transcription_model_id=transcription_model_id,
        enable_summary=enable_summary,
        enable_diarization=enable_diarization,
        diarization_speakers=diarization_speakers,
    )


def maybe_diarize(
    audio_file: Path,
    job_dir: Path,
    enable_diarization: bool,
    diarization_speakers: int = 0,
    progress_plan: dict[str, tuple[float, float]] | None = None,
) -> None:
    if not enable_diarization:
        return
    if not settings.enable_diarization:
        print("Diarization was requested but ENABLE_DIARIZATION is disabled.")
        save_job_timing(job_dir, "diarization", 0, status="skipped")
        return
    if progress_plan:
        run_progress_step(
            job_dir,
            progress_plan,
            "diarization",
            lambda progress: diarize(
                audio_file,
                job_dir,
                diarization_speakers=diarization_speakers,
                progress_callback=progress,
            ),
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
    progress_plan: dict[str, tuple[float, float]] | None = None,
) -> None:
    if not enable_summary:
        print("Summary was disabled for this job.")
        save_job_timing(job_dir, "summary", 0, status="skipped")
        return
    if progress_plan:
        run_progress_step(
            job_dir,
            progress_plan,
            "summary",
            lambda progress: summarize(transcript, job_dir, progress_callback=progress),
            skipped=lambda result: result is None,
        )
        return

    timed_step(job_dir, "summary", lambda: summarize(transcript, job_dir), skipped=lambda result: result is None)


def enforce_download_quota(job_dir: Path, input_file: Path) -> None:
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


def process_file(
    input_file: Path,
    job_dir: Path,
    transcription_model_id: str = DEFAULT_TRANSCRIPTION_MODEL_ID,
    enable_summary: bool = True,
    enable_diarization: bool = False,
    diarization_speakers: int = 0,
) -> None:
    job_dir.mkdir(parents=True, exist_ok=True)
    save_metadata(
        job_dir,
        input_file,
        status="running",
        transcription_model_id=transcription_model_id,
        enable_summary=enable_summary,
        enable_diarization=enable_diarization,
        diarization_speakers=diarization_speakers,
    )
    progress_plan = build_progress_plan(False, enable_diarization, enable_summary)
    save_job_progress(job_dir, 0, "queued")

    try:
        audio_file = run_progress_step(
            job_dir,
            progress_plan,
            "prepare_audio",
            lambda progress: prepare_audio(input_file, job_dir),
        )
        transcript = run_progress_step(
            job_dir,
            progress_plan,
            "transcription",
            lambda progress: transcribe(
                audio_file,
                job_dir,
                transcription_model_id=transcription_model_id,
                progress_callback=progress,
            ),
        )
        maybe_diarize(audio_file, job_dir, enable_diarization, diarization_speakers, progress_plan)
        maybe_summarize(transcript, job_dir, enable_summary, progress_plan)
        save_metadata(job_dir, input_file, status="completed", transcription_model_id=transcription_model_id)
        save_job_progress(job_dir, 100, "completed", status="completed")
        print("Processing completed.")
    except Exception as exc:
        mark_progress_failed(job_dir, str(exc))
        save_metadata(job_dir, input_file, status="failed", transcription_model_id=transcription_model_id)
        raise


def process_url(
    source_url: str,
    job_dir: Path,
    transcription_model_id: str = DEFAULT_TRANSCRIPTION_MODEL_ID,
    enable_summary: bool = True,
    enable_diarization: bool = False,
    diarization_speakers: int = 0,
) -> None:
    job_dir.mkdir(parents=True, exist_ok=True)
    save_metadata(
        job_dir,
        source_url,
        status="running",
        transcription_model_id=transcription_model_id,
        enable_summary=enable_summary,
        enable_diarization=enable_diarization,
        diarization_speakers=diarization_speakers,
    )
    progress_plan = build_progress_plan(True, enable_diarization, enable_summary)
    save_job_progress(job_dir, 0, "queued")

    try:
        input_file = run_progress_step(
            job_dir,
            progress_plan,
            "download",
            lambda progress: download_media(source_url, job_dir),
        )
        enforce_download_quota(job_dir, input_file)
        save_metadata(job_dir, input_file, status="running", transcription_model_id=transcription_model_id)
        audio_file = run_progress_step(
            job_dir,
            progress_plan,
            "prepare_audio",
            lambda progress: prepare_audio(input_file, job_dir),
        )
        transcript = run_progress_step(
            job_dir,
            progress_plan,
            "transcription",
            lambda progress: transcribe(
                audio_file,
                job_dir,
                transcription_model_id=transcription_model_id,
                progress_callback=progress,
            ),
        )
        maybe_diarize(audio_file, job_dir, enable_diarization, diarization_speakers, progress_plan)
        maybe_summarize(transcript, job_dir, enable_summary, progress_plan)
        save_metadata(job_dir, input_file, status="completed", transcription_model_id=transcription_model_id)
        save_job_progress(job_dir, 100, "completed", status="completed")
        print("Processing completed.")
    except Exception as exc:
        mark_progress_failed(job_dir, str(exc))
        save_metadata(job_dir, source_url, status="failed", transcription_model_id=transcription_model_id)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Process uploaded audio/video file.")
    parser.add_argument("input_file", type=Path)
    parser.add_argument("job_dir", type=Path)
    parser.add_argument("--source-url")
    parser.add_argument("--transcription-model", default=DEFAULT_TRANSCRIPTION_MODEL_ID)
    parser.add_argument("--no-summary", action="store_true")
    parser.add_argument("--diarization", action="store_true")
    parser.add_argument("--diarization-speakers", type=int, default=0)
    parser.add_argument("--edit-model")
    parser.add_argument("--edit-source", default="transcript")
    args = parser.parse_args()

    if str(args.input_file) == "edit-transcript":
        process_edit(args.job_dir, editor_model=args.edit_model, transcript_source=args.edit_source)
    elif args.source_url:
        process_url(
            args.source_url,
            args.job_dir,
            transcription_model_id=args.transcription_model,
            enable_summary=not args.no_summary,
            enable_diarization=args.diarization,
            diarization_speakers=max(args.diarization_speakers, 0),
        )
    else:
        process_file(
            args.input_file,
            args.job_dir,
            transcription_model_id=args.transcription_model,
            enable_summary=not args.no_summary,
            enable_diarization=args.diarization,
            diarization_speakers=max(args.diarization_speakers, 0),
        )


if __name__ == "__main__":
    main()
