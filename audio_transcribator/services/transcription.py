import base64
import json
import subprocess
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from audio_transcribator.config import settings
from audio_transcribator.services.transcription_models import resolve_transcription_model
from audio_transcribator.utils.files import write_text_atomic


LOCAL_MODEL_REQUIRED_FILES = {"config.json", "model.bin", "tokenizer.json", "vocabulary.txt"}
OPENROUTER_RETRY_STATUS_CODES = {502, 503, 504}
OPENROUTER_MAX_ATTEMPTS = 3


def resolve_auto_device(device: str) -> str:
    if device != "auto":
        return device

    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def format_timestamp(seconds: float) -> str:
    total_seconds = max(int(round(seconds)), 0)
    minutes, seconds_part = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds_part:02d}"
    return f"{minutes:02d}:{seconds_part:02d}"


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
        if settings.whisper_local_files_only:
            raise RuntimeError(
                "Local faster-whisper model was not found. Set WHISPER_MODEL to a local model directory "
                "or pre-download the model into data/model_cache/faster-whisper. "
                "Set WHISPER_LOCAL_FILES_ONLY=false only if Hugging Face downloads are allowed."
            )
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

    if settings.whisper_local_files_only:
        raise RuntimeError(
            "Local faster-whisper model was not found. Set WHISPER_MODEL to a local model directory "
            "or pre-download the model into data/model_cache/faster-whisper. "
            "Set WHISPER_LOCAL_FILES_ONLY=false only if Hugging Face downloads are allowed."
        )

    return configured_model


def transcribe(
    audio_file: Path,
    job_dir: Path,
    transcription_model_id: str | None = None,
    progress_callback=None,
) -> str:
    model_config = resolve_transcription_model(transcription_model_id)
    print(f"Transcribing with {model_config['id']}...")

    if model_config["provider"] == "openrouter":
        return transcribe_openrouter(audio_file, job_dir, model_config["model"], progress_callback=progress_callback)
    if model_config["provider"] == "whisperx":
        return transcribe_whisperx(
            audio_file,
            job_dir,
            model_config.get("model") or settings.whisperx_model,
            progress_callback=progress_callback,
        )

    return transcribe_local(audio_file, job_dir, progress_callback=progress_callback)


def _save_transcript_file(job_dir: Path, transcript: str) -> None:
    write_text_atomic(job_dir / "stenogramma.txt", transcript)


def _save_transcript_segments(job_dir: Path, segments: list[dict]) -> None:
    (job_dir / "transcript_segments.json").write_text(
        json.dumps(segments, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def transcribe_local(audio_file: Path, job_dir: Path, progress_callback=None) -> str:
    from faster_whisper import WhisperModel

    settings.model_cache_dir.mkdir(parents=True, exist_ok=True)
    model = WhisperModel(
        resolve_whisper_model_source(),
        compute_type=settings.whisper_compute_type,
        download_root=str(settings.model_cache_dir / "faster-whisper"),
        local_files_only=settings.whisper_local_files_only,
    )

    segments, info = model.transcribe(
        str(audio_file),
        language=settings.transcription_language,
        task="transcribe",
    )
    duration = float(getattr(info, "duration", 0) or 0)
    segment_texts = []
    transcript_segments = []
    for segment in segments:
        text = normalize_transcript_text(segment.text)
        if text:
            print(text, flush=True)
            segment_texts.append(text)
            transcript_segments.append(
                {
                    "start": float(segment.start),
                    "end": float(segment.end),
                    "text": text,
                }
            )
            _save_transcript_file(job_dir, normalize_transcript_text(" ".join(segment_texts)))
            _save_transcript_segments(job_dir, transcript_segments)
            if progress_callback and duration > 0:
                progress_callback(
                    min(float(segment.end) / duration * 100, 99),
                    f"Обработано {format_timestamp(float(segment.end))} из {format_timestamp(duration)}",
                )

    transcript = normalize_transcript_text(" ".join(segment_texts))
    _save_transcript_file(job_dir, transcript)
    _save_transcript_segments(job_dir, transcript_segments)
    if progress_callback:
        progress_callback(100)

    return transcript


def transcribe_whisperx(audio_file: Path, job_dir: Path, model_name: str, progress_callback=None) -> str:
    try:
        import whisperx
    except ImportError as exc:
        raise RuntimeError(
            "WhisperX is not installed. Install whisperx==3.1.1 in the Docker image or server environment "
            "before selecting the WhisperX large-v3 transcription model."
        ) from exc

    settings.model_cache_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_auto_device(settings.whisperx_device)
    print(f"Transcribing with WhisperX {model_name} on {device}...", flush=True)

    try:
        model = whisperx.load_model(
            model_name,
            device,
            compute_type=settings.whisper_compute_type,
            language=settings.transcription_language,
            download_root=str(settings.model_cache_dir / "whisperx"),
        )
    except TypeError:
        model = whisperx.load_model(
            model_name,
            device,
            compute_type=settings.whisper_compute_type,
            download_root=str(settings.model_cache_dir / "whisperx"),
        )

    result = model.transcribe(
        str(audio_file),
        batch_size=settings.whisperx_batch_size,
        language=settings.transcription_language,
    )
    raw_segments = result.get("segments") or []
    segment_texts = []
    transcript_segments = []
    duration = max((float(segment.get("end") or 0) for segment in raw_segments), default=0)
    for segment in raw_segments:
        text = normalize_transcript_text(str(segment.get("text") or ""))
        if not text:
            continue

        start = float(segment.get("start") or 0)
        end = float(segment.get("end") or start)
        print(text, flush=True)
        segment_texts.append(text)
        transcript_segments.append({"start": start, "end": end, "text": text})
        _save_transcript_file(job_dir, normalize_transcript_text(" ".join(segment_texts)))
        _save_transcript_segments(job_dir, transcript_segments)
        if progress_callback and duration > 0:
            progress_callback(
                min(end / duration * 100, 99),
                f"Обработано {format_timestamp(end)} из {format_timestamp(duration)}",
            )

    transcript = normalize_transcript_text(" ".join(segment_texts))
    _save_transcript_file(job_dir, transcript)
    _save_transcript_segments(job_dir, transcript_segments)
    if progress_callback:
        progress_callback(100)

    return transcript


def transcribe_openrouter(audio_file: Path, job_dir: Path, model: str, progress_callback=None) -> str:
    if not settings.openrouter_api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required for OpenRouter transcription models")

    if audio_file.stat().st_size > settings.openrouter_transcription_max_bytes:
        return transcribe_openrouter_chunked(audio_file, job_dir, model, progress_callback=progress_callback)

    response_data = transcribe_openrouter_file(audio_file, model)
    text = extract_openrouter_text(response_data)

    _save_transcript_file(job_dir, text)
    print(text, flush=True)

    usage = response_data.get("usage")
    if usage:
        (job_dir / "transcription_usage.json").write_text(
            json.dumps(usage, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if progress_callback:
        progress_callback(100)

    return text


def transcribe_openrouter_chunked(audio_file: Path, job_dir: Path, model: str, progress_callback=None) -> str:
    chunks_dir = job_dir / "openrouter_chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    chunk_pattern = chunks_dir / "chunk_%03d.mp3"
    print(
        f"Audio file is {audio_file.stat().st_size} bytes; splitting into "
        f"{settings.openrouter_transcription_chunk_seconds}s OpenRouter chunks...",
        flush=True,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(audio_file),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-b:a",
            "64k",
            "-f",
            "segment",
            "-segment_time",
            str(settings.openrouter_transcription_chunk_seconds),
            str(chunk_pattern),
        ],
        check=True,
    )

    chunks = sorted(chunks_dir.glob("chunk_*.mp3"))
    if not chunks:
        raise RuntimeError("OpenRouter chunking finished but no audio chunks were created")

    transcript_parts = []
    usage_parts = []
    for index, chunk in enumerate(chunks, start=1):
        print(f"Transcribing OpenRouter chunk {index}/{len(chunks)}: {chunk.name}", flush=True)
        response_data = transcribe_openrouter_file(chunk, model)
        text = extract_openrouter_text(response_data)
        if text:
            transcript_parts.append(text)
            current_transcript = normalize_transcript_text(" ".join(transcript_parts))
            _save_transcript_file(job_dir, current_transcript)
            print(text, flush=True)

        usage = response_data.get("usage")
        if usage:
            usage_parts.append({"chunk": chunk.name, "usage": usage})
        if progress_callback:
            progress_callback(index / len(chunks) * 100, f"Обработано частей: {index}/{len(chunks)}")

    transcript = normalize_transcript_text(" ".join(transcript_parts))
    _save_transcript_file(job_dir, transcript)
    if usage_parts:
        (job_dir / "transcription_usage.json").write_text(
            json.dumps(usage_parts, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return transcript


def transcribe_openrouter_file(audio_file: Path, model: str) -> dict:
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

    return call_openrouter_transcription(request, model)


def extract_openrouter_text(response_data: dict) -> str:
    text = response_data.get("text")
    if not isinstance(text, str):
        raise RuntimeError(f"OpenRouter transcription response does not contain text: {response_data}")
    return normalize_transcript_text(text)


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
