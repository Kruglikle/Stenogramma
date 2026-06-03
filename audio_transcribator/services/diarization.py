from pathlib import Path


def diarize(audio_file: Path, job_dir: Path) -> list[dict]:
    print("Diarization is disabled: Hugging Face pyannote integration was removed.")
    return []
