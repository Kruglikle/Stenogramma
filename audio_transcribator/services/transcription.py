import base64
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from faster_whisper import WhisperModel

from audio_transcribator.config import settings
from audio_transcribator.services.transcription_models import resolve_transcription_model


def transcribe(audio_file: Path, job_dir: Path, transcription_model_id: str | None = None) -> str:
    model_config = resolve_transcription_model(transcription_model_id)
    print(f"Transcribing with {model_config['id']}...")

    if model_config["provider"] == "openrouter":
        return transcribe_openrouter(audio_file, job_dir, model_config["model"])

    return transcribe_local(audio_file, job_dir)


def transcribe_local(audio_file: Path, job_dir: Path) -> str:
    settings.model_cache_dir.mkdir(parents=True, exist_ok=True)
    model = WhisperModel(
        settings.whisper_model,
        compute_type=settings.whisper_compute_type,
        download_root=str(settings.model_cache_dir / "faster-whisper"),
    )

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


def transcribe_openrouter(audio_file: Path, job_dir: Path, model: str) -> str:
    if not settings.openrouter_api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required for OpenRouter transcription models")

    audio_format = audio_file.suffix.lower().lstrip(".") or "wav"
    payload = {
        "model": model,
        "input_audio": {
            "data": base64.b64encode(audio_file.read_bytes()).decode("ascii"),
            "format": audio_format,
        },
    }

    url = f"{settings.openrouter_base_url.rstrip('/')}/audio/transcriptions"
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {settings.openrouter_api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=300) as response:
            response_data = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenRouter transcription failed: {exc.code} {error_body}") from exc

    text = response_data.get("text")
    if not isinstance(text, str):
        raise RuntimeError(f"OpenRouter transcription response does not contain text: {response_data}")

    transcript_path = job_dir / "transcript.txt"
    transcript_path.write_text(text, encoding="utf-8")
    print(text)

    usage = response_data.get("usage")
    if usage:
        (job_dir / "transcription_usage.json").write_text(
            json.dumps(usage, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return text
