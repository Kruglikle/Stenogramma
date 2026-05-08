from pathlib import Path

from faster_whisper import WhisperModel

from audio_transcribator.config import settings


def transcribe(audio_file: Path, job_dir: Path) -> str:
    print("Transcribing...")
    model = WhisperModel(settings.whisper_model, compute_type=settings.whisper_compute_type)

    segments, _ = model.transcribe(str(audio_file))
    transcript_path = job_dir / "transcript.txt"

    lines = []
    with open(transcript_path, "w", encoding="utf-8") as f:
        for segment in segments:
            line = f"[{segment.start:.2f} - {segment.end:.2f}] {segment.text.strip()}"
            print(line)
            f.write(line + "\n")
            lines.append(line)

    return "\n".join(lines)

