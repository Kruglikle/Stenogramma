import argparse
import json
from pathlib import Path

from audio_transcribator.config import settings
from audio_transcribator.services.audio import prepare_audio
from audio_transcribator.services.diarization import diarize
from audio_transcribator.services.summary import summarize
from audio_transcribator.services.transcription import transcribe


def save_metadata(job_dir: Path, input_file: Path, status: str = "completed") -> None:
    metadata = {
        "job_id": job_dir.name,
        "status": status,
        "input_file": str(input_file),
        "files": sorted(p.name for p in job_dir.iterdir() if p.is_file()),
    }
    with open(job_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def process_file(input_file: Path, job_dir: Path) -> None:
    job_dir.mkdir(parents=True, exist_ok=True)
    save_metadata(job_dir, input_file, status="running")

    try:
        audio_file = prepare_audio(input_file, job_dir)
        transcript = transcribe(audio_file, job_dir)
        if settings.enable_diarization:
            diarize(audio_file, job_dir)
        summarize(transcript, job_dir)
        save_metadata(job_dir, input_file, status="completed")
        print("Processing completed.")
    except Exception:
        save_metadata(job_dir, input_file, status="failed")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Process uploaded audio/video file.")
    parser.add_argument("input_file", type=Path)
    parser.add_argument("job_dir", type=Path)
    args = parser.parse_args()

    process_file(args.input_file, args.job_dir)


if __name__ == "__main__":
    main()

