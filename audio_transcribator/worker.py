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
    load_job_metadata,
    save_job_metadata,
    save_job_timing,
)
from audio_transcribator.services.summary import summarize
from audio_transcribator.services.transcription import transcribe
from audio_transcribator.services.transcription_models import DEFAULT_TRANSCRIPTION_MODEL_ID


T = TypeVar("T")


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
) -> None:
    save_job_metadata(
        job_dir,
        input_file,
        status,
        transcription_model_id=transcription_model_id,
        enable_summary=enable_summary,
        enable_diarization=enable_diarization,
    )


def maybe_diarize(audio_file: Path, job_dir: Path, enable_diarization: bool) -> None:
    if not enable_diarization:
        return
    if not settings.enable_diarization:
        print("Diarization was requested but ENABLE_DIARIZATION is disabled.")
        save_job_timing(job_dir, "diarization", 0, status="skipped")
        return
    timed_step(job_dir, "diarization", lambda: diarize(audio_file, job_dir), skipped=lambda result: not result)


def maybe_summarize(transcript: str, job_dir: Path, enable_summary: bool) -> None:
    if not enable_summary:
        print("Summary was disabled for this job.")
        save_job_timing(job_dir, "summary", 0, status="skipped")
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
) -> None:
    job_dir.mkdir(parents=True, exist_ok=True)
    save_metadata(
        job_dir,
        input_file,
        status="running",
        transcription_model_id=transcription_model_id,
        enable_summary=enable_summary,
        enable_diarization=enable_diarization,
    )

    try:
        audio_file = timed_step(job_dir, "prepare_audio", lambda: prepare_audio(input_file, job_dir))
        transcript = timed_step(
            job_dir,
            "transcription",
            lambda: transcribe(audio_file, job_dir, transcription_model_id=transcription_model_id),
        )
        maybe_diarize(audio_file, job_dir, enable_diarization)
        maybe_summarize(transcript, job_dir, enable_summary)
        save_metadata(job_dir, input_file, status="completed", transcription_model_id=transcription_model_id)
        print("Processing completed.")
    except Exception:
        save_metadata(job_dir, input_file, status="failed", transcription_model_id=transcription_model_id)
        raise


def process_url(
    source_url: str,
    job_dir: Path,
    transcription_model_id: str = DEFAULT_TRANSCRIPTION_MODEL_ID,
    enable_summary: bool = True,
    enable_diarization: bool = False,
) -> None:
    job_dir.mkdir(parents=True, exist_ok=True)
    save_metadata(
        job_dir,
        source_url,
        status="running",
        transcription_model_id=transcription_model_id,
        enable_summary=enable_summary,
        enable_diarization=enable_diarization,
    )

    try:
        input_file = timed_step(job_dir, "download", lambda: download_media(source_url, job_dir))
        enforce_download_quota(job_dir, input_file)
        save_metadata(job_dir, input_file, status="running", transcription_model_id=transcription_model_id)
        audio_file = timed_step(job_dir, "prepare_audio", lambda: prepare_audio(input_file, job_dir))
        transcript = timed_step(
            job_dir,
            "transcription",
            lambda: transcribe(audio_file, job_dir, transcription_model_id=transcription_model_id),
        )
        maybe_diarize(audio_file, job_dir, enable_diarization)
        maybe_summarize(transcript, job_dir, enable_summary)
        save_metadata(job_dir, input_file, status="completed", transcription_model_id=transcription_model_id)
        print("Processing completed.")
    except Exception:
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
        )
    else:
        process_file(
            args.input_file,
            args.job_dir,
            transcription_model_id=args.transcription_model,
            enable_summary=not args.no_summary,
            enable_diarization=args.diarization,
        )


if __name__ == "__main__":
    main()
