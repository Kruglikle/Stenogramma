from pathlib import Path

from audio_transcribator.config import settings


def diarize(audio_file: Path, job_dir: Path) -> list[dict]:
    if not settings.hf_token:
        print("HF_TOKEN is not set, skipping diarization")
        return []

    from pyannote.audio import Pipeline

    print("Diarizing...")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=settings.hf_token)
    output = pipeline(str(audio_file))
    diarization = output.speaker_diarization

    segments = []
    with open(job_dir / "diarization.txt", "w", encoding="utf-8") as f:
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            item = {"start": turn.start, "end": turn.end, "speaker": speaker}
            segments.append(item)
            line = f"{speaker}: {turn.start:.2f} - {turn.end:.2f}"
            print(line)
            f.write(line + "\n")

    return segments

