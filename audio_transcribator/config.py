import os
from pathlib import Path


class Settings:
    def __init__(self) -> None:
        self.base_dir = Path(os.getenv("BASE_DIR", Path.cwd())).resolve()
        self.upload_dir = Path(os.getenv("UPLOAD_DIR", self.base_dir / "data" / "uploads")).resolve()
        self.results_dir = Path(os.getenv("RESULTS_DIR", self.base_dir / "data" / "api_results")).resolve()

        self.api_token = os.getenv("API_TOKEN", "test-token")
        self.api_username = os.getenv("API_USERNAME", "admin")
        self.api_password = os.getenv("API_PASSWORD", "admin123")

        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        self.openrouter_base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        self.summary_model = os.getenv("SUMMARY_MODEL", "qwen/qwen3.5-35b-a3b")

        self.whisper_model = os.getenv("WHISPER_MODEL", "base")
        self.whisper_compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

        self.hf_token = os.getenv("HF_TOKEN")
        self.enable_diarization = os.getenv("ENABLE_DIARIZATION", "false").lower() in {"1", "true", "yes"}

    def ensure_dirs(self) -> None:
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()

