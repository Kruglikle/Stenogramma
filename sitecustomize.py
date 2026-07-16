try:
    from audio_transcribator.services.torch_compat import ensure_torchaudio_backend_api

    ensure_torchaudio_backend_api()
except Exception:
    pass
