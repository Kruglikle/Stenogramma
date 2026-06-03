import asyncio
from contextlib import suppress
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from audio_transcribator.api.routes import router
from audio_transcribator.config import settings
from audio_transcribator.db import init_database
from audio_transcribator.services.jobs import cleanup_expired_jobs
from audio_transcribator.ui.routes import router as ui_router


settings.ensure_dirs()


async def cleanup_expired_jobs_loop() -> None:
    while True:
        await asyncio.sleep(settings.data_cleanup_interval_seconds)
        cleanup_expired_jobs()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_database()
    cleanup_expired_jobs()
    cleanup_task = asyncio.create_task(cleanup_expired_jobs_loop())
    try:
        yield
    finally:
        cleanup_task.cancel()
        with suppress(asyncio.CancelledError):
            await cleanup_task


app = FastAPI(title="Audio/Video Processing API", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(settings.base_dir / "audio_transcribator" / "static")), name="static")
app.include_router(router)
app.include_router(ui_router)
