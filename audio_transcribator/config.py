import os
from pathlib import Path


def load_env_file(env_file: Path) -> None:
    if not env_file.exists():
        return

    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key:
            os.environ.setdefault(key, value)


load_env_file(Path.cwd() / ".env")


class Settings:
    def __init__(self) -> None:
        self.base_dir = Path(os.getenv("BASE_DIR", Path.cwd())).resolve()
        self.upload_dir = Path(os.getenv("UPLOAD_DIR", self.base_dir / "data" / "uploads")).resolve()
        self.results_dir = Path(os.getenv("RESULTS_DIR", self.base_dir / "data" / "api_results")).resolve()
        self.model_cache_dir = Path(os.getenv("MODEL_CACHE_DIR", self.base_dir / "data" / "model_cache")).resolve()
        self.hf_home = Path(os.getenv("HF_HOME", self.model_cache_dir / "huggingface")).resolve()
        self.hf_hub_cache = Path(os.getenv("HF_HUB_CACHE", self.hf_home / "hub")).resolve()

        os.environ.setdefault("HF_HOME", str(self.hf_home))
        os.environ.setdefault("HF_HUB_CACHE", str(self.hf_hub_cache))
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

        self.api_token = os.getenv("API_TOKEN", "test-token")
        self.api_username = os.getenv("API_USERNAME", "admin")
        self.api_password = os.getenv("API_PASSWORD", "admin123")
        self.creator_username = os.getenv("CREATOR_USERNAME", "creator")
        self.creator_password = os.getenv("CREATOR_PASSWORD", "creator123")
        self.add_user_admin_token = os.getenv("ADD_USER_ADMIN_TOKEN", "change-me-add-user-token")

        self.database_url = os.getenv(
            "DATABASE_URL",
            "postgresql://audio_transcribator:audio_transcribator_password@localhost:5432/audio_transcribator",
        )
        self.database_connect_retries = int(os.getenv("DATABASE_CONNECT_RETRIES", "30"))
        self.database_connect_delay_seconds = float(os.getenv("DATABASE_CONNECT_DELAY_SECONDS", "1"))

        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        self.openrouter_base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.ollama_api_key = os.getenv("OLLAMA_API_KEY", "ollama")
        self.openrouter_transcription_max_bytes = int(
            os.getenv("OPENROUTER_TRANSCRIPTION_MAX_BYTES", str(18 * 1024 * 1024))
        )
        self.openrouter_transcription_chunk_seconds = int(
            os.getenv("OPENROUTER_TRANSCRIPTION_CHUNK_SECONDS", "600")
        )
        self.summary_model = os.getenv("SUMMARY_MODEL", "qwen/qwen3.5-35b-a3b")
        self.editor_model = os.getenv("EDITOR_MODEL", "qwen/qwen3.6-35b-a3b")
        self.editor_temperature = float(os.getenv("EDITOR_TEMPERATURE", "0.1"))
        self.transcription_models_file = Path(
            os.getenv("TRANSCRIPTION_MODELS_FILE", self.base_dir / "audio_transcribator" / "transcription_models.json")
        ).resolve()

        self.whisper_model = os.getenv("WHISPER_MODEL", "base")
        self.whisper_compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
        self.transcription_language = os.getenv("TRANSCRIPTION_LANGUAGE", "ru").strip() or None

        self.hf_token = os.getenv("HF_TOKEN")
        self.enable_diarization = os.getenv("ENABLE_DIARIZATION", "false").lower() in {"1", "true", "yes"}

    def ensure_dirs(self) -> None:
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        self.hf_home.mkdir(parents=True, exist_ok=True)
        self.hf_hub_cache.mkdir(parents=True, exist_ok=True)


settings = Settings()
