from audio_transcribator.config import settings


SUPPORTED_PROCESSING_DEVICES = {"auto", "cpu", "cuda"}


def normalize_processing_device(device: str | None) -> str:
    value = (device or "auto").strip().lower()
    if value == "gpu":
        value = "cuda"
    if value not in SUPPORTED_PROCESSING_DEVICES:
        raise ValueError("Processing device must be one of: auto, cpu, cuda")
    return value


def default_processing_device() -> str:
    return normalize_processing_device(settings.processing_device_default)


def resolve_processing_device(device: str | None, configured_device: str) -> str:
    selected = normalize_processing_device(device) if device else normalize_processing_device(configured_device)
    if selected != "auto":
        return selected

    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def whisper_compute_type_for_device(device: str) -> str:
    compute_type = settings.whisper_compute_type
    if device == "cpu" and compute_type in {"float16", "int8_float16"}:
        return settings.whisper_cpu_compute_type
    return compute_type
