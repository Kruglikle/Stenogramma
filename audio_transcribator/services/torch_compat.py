def ensure_torchaudio_backend_api() -> None:
    try:
        import torchaudio
    except Exception:
        return

    if not hasattr(torchaudio, "set_audio_backend"):
        torchaudio.set_audio_backend = lambda *args, **kwargs: None
    if not hasattr(torchaudio, "get_audio_backend"):
        torchaudio.get_audio_backend = lambda *args, **kwargs: None
