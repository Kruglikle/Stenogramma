import base64
import json
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from faster_whisper import WhisperModel

from audio_transcribator.config import settings
from audio_transcribator.services.transcription_models import resolve_transcription_model


LOCAL_MODEL_REQUIRED_FILES = {"config.json", "model.bin", "tokenizer.json", "vocabulary.txt"}
OPENROUTER_RETRY_STATUS_CODES = {502, 503, 504}
OPENROUTER_MAX_ATTEMPTS = 3


def normalize_transcript_text(text: str) -> str:
    return " ".join(text.split())


def resolve_whisper_model_source() -> str:
    configured_model = settings.whisper_model
    configured_path = Path(configured_model)
    if configured_path.exists():
        return str(configured_path)

    repo_name = configured_model
    if "/" not in repo_name:
        repo_name = f"Systran/faster-whisper-{configured_model}"

    cache_repo_dir = settings.model_cache_dir / "faster-whisper" / f"models--{repo_name.replace('/', '--')}"
    snapshots_dir = cache_repo_dir / "snapshots"
    if not snapshots_dir.exists():
        return configured_model

    snapshots = sorted(
        (path for path in snapshots_dir.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for snapshot in snapshots:
        if all((snapshot / filename).exists() for filename in LOCAL_MODEL_REQUIRED_FILES):
            print(f"Using cached faster-whisper model: {snapshot}")
            return str(snapshot)

    return configured_model


def transcribe(audio_file: Path, job_dir: Path, transcription_model_id: str | None = None) -> str:
    model_config = resolve_transcription_model(transcription_model_id)
    print(f"Transcribing with {model_config['id']}...")

    if model_config["provider"] == "openrouter":
        return transcribe_openrouter(audio_file, job_dir, model_config["model"])

    return transcribe_local(audio_file, job_dir)


def transcribe_local(audio_file: Path, job_dir: Path) -> str:
    settings.model_cache_dir.mkdir(parents=True, exist_ok=True)
    model = WhisperModel(
        resolve_whisper_model_source(),
        compute_type=settings.whisper_compute_type,
        download_root=str(settings.model_cache_dir / "faster-whisper"),
    )

    segments, _ = model.transcribe(
        str(audio_file),
        language=settings.transcription_language,
        task="transcribe",
    )
    transcript_path = job_dir / "transcript.txt"

    segment_texts = []
    with open(transcript_path, "w", encoding="utf-8") as transcript_file:
        for segment in segments:
            text = normalize_transcript_text(segment.text)
            if text:
                print(text, flush=True)
                if segment_texts:
                    transcript_file.write(" ")
                transcript_file.write(text)
                transcript_file.flush()
                segment_texts.append(text)

    transcript = normalize_transcript_text(" ".join(segment_texts))
    transcript_path.write_text(transcript, encoding="utf-8")

    return transcript


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
        "task": "transcribe",
    }
    if settings.transcription_language:
        payload["language"] = settings.transcription_language

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

    response_data = call_openrouter_transcription(request, model)

    text = response_data.get("text")
    if not isinstance(text, str):
        raise RuntimeError(f"OpenRouter transcription response does not contain text: {response_data}")
    text = normalize_transcript_text(text)

    transcript_path = job_dir / "transcript.txt"
    transcript_path.write_text(text, encoding="utf-8")
    print(text, flush=True)

    usage = response_data.get("usage")
    if usage:
        (job_dir / "transcription_usage.json").write_text(
            json.dumps(usage, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return text


def call_openrouter_transcription(request: Request, model: str) -> dict:
    last_error: RuntimeError | None = None
    for attempt in range(1, OPENROUTER_MAX_ATTEMPTS + 1):
        try:
            with urlopen(request, timeout=300) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"OpenRouter transcription failed: {exc.code} {error_body}")
            if exc.code not in OPENROUTER_RETRY_STATUS_CODES or attempt == OPENROUTER_MAX_ATTEMPTS:
                raise last_error from exc

            delay_seconds = attempt * 5
            print(
                f"OpenRouter returned {exc.code} for {model}; retrying "
                f"{attempt + 1}/{OPENROUTER_MAX_ATTEMPTS} in {delay_seconds}s...",
                flush=True,
            )
            time.sleep(delay_seconds)

    raise last_error or RuntimeError("OpenRouter transcription failed")
