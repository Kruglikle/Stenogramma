import argparse
from pathlib import Path

from audio_transcribator.config import settings
from audio_transcribator.services.audio import download_media, prepare_audio
from audio_transcribator.services.diarization import diarize
from audio_transcribator.services.jobs import save_job_metadata
from audio_transcribator.services.summary import summarize
from audio_transcribator.services.transcription import transcribe
from audio_transcribator.services.transcription_models import DEFAULT_TRANSCRIPTION_MODEL_ID


def save_metadata(
    job_dir: Path,
    input_file: Path | str,
    status: str = "completed",
    transcription_model_id: str | None = None,
) -> None:
    save_job_metadata(job_dir, input_file, status, transcription_model_id=transcription_model_id)


def process_file(
    input_file: Path,
    job_dir: Path,
    transcription_model_id: str = DEFAULT_TRANSCRIPTION_MODEL_ID,
) -> None:
    job_dir.mkdir(parents=True, exist_ok=True)
    save_metadata(job_dir, input_file, status="running", transcription_model_id=transcription_model_id)

    try:
        audio_file = prepare_audio(input_file, job_dir)
        transcript = transcribe(audio_file, job_dir, transcription_model_id=transcription_model_id)
        if settings.enable_diarization:
            diarize(audio_file, job_dir)
        summarize(transcript, job_dir)
        save_metadata(job_dir, input_file, status="completed", transcription_model_id=transcription_model_id)
        print("Processing completed.")
    except Exception:
        save_metadata(job_dir, input_file, status="failed", transcription_model_id=transcription_model_id)
        raise


def process_url(
    source_url: str,
    job_dir: Path,
    transcription_model_id: str = DEFAULT_TRANSCRIPTION_MODEL_ID,
) -> None:
    job_dir.mkdir(parents=True, exist_ok=True)
    save_metadata(job_dir, source_url, status="running", transcription_model_id=transcription_model_id)

    try:
        input_file = download_media(source_url, job_dir)
        save_metadata(job_dir, input_file, status="running", transcription_model_id=transcription_model_id)
        audio_file = prepare_audio(input_file, job_dir)
        transcript = transcribe(audio_file, job_dir, transcription_model_id=transcription_model_id)
        if settings.enable_diarization:
            diarize(audio_file, job_dir)
        summarize(transcript, job_dir)
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
    args = parser.parse_args()

    if args.source_url:
        process_url(args.source_url, args.job_dir, transcription_model_id=args.transcription_model)
    else:
        process_file(args.input_file, args.job_dir, transcription_model_id=args.transcription_model)


if __name__ == "__main__":
    main()
