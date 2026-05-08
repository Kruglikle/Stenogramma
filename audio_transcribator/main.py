from fastapi import FastAPI

from audio_transcribator.api.routes import router
from audio_transcribator.config import settings


settings.ensure_dirs()

app = FastAPI(title="Audio/Video Processing API")
app.include_router(router)

