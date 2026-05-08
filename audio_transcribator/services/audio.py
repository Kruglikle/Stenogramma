import subprocess
from pathlib import Path


VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg"}


def prepare_audio(input_file: Path, job_dir: Path) -> Path:
    if not input_file.exists():
        raise FileNotFoundError(f"File not found: {input_file}")

    suffix = input_file.suffix.lower()
    if suffix in VIDEO_EXTENSIONS:
        audio_path = job_dir / "work_audio.wav"
        print(f"Extracting audio from video: {input_file}")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(input_file),
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ac",
                "1",
                "-ar",
                "16000",
                str(audio_path),
            ],
            check=True,
        )
        return audio_path

    if suffix in AUDIO_EXTENSIONS:
        print(f"Using audio file: {input_file}")
        return input_file

    raise ValueError(f"Unsupported file format: {suffix}")

