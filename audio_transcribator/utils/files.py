from pathlib import Path


ALLOWED_DOWNLOADS = {
    "transcript.txt",
    "edited_transcript.txt",
    "summary.txt",
    "run.log",
    "work_audio.wav",
    "diarization.txt",
}


def tail(path: Path, lines: int = 30) -> str:
    if not path.exists():
        return ""
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
