import shutil
import subprocess
import sys
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse

from audio_transcribator.auth import check_auth, verify_credentials
from audio_transcribator.config import settings
from audio_transcribator.models import LoginRequest
from audio_transcribator.utils.files import ALLOWED_DOWNLOADS, tail


router = APIRouter()


@router.post("/login")
def login(data: LoginRequest):
    if not verify_credentials(data.username, data.password):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    return {"access_token": settings.api_token, "token_type": "bearer"}


@router.get("/")
def root():
    return {"status": "ok", "service": "audio-video-processing"}


@router.post("/process")
async def process_file(
    file: UploadFile = File(...),
    authorization: str | None = Header(default=None),
):
    check_auth(authorization)

    job_id = str(uuid4())
    job_dir = settings.results_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    safe_filename = file.filename or "upload"
    input_path = settings.upload_dir / f"{job_id}_{safe_filename}"

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    log_path = job_dir / "run.log"
    command = [sys.executable, "process_audio_fast.py", str(input_path), str(job_dir)]

    with open(log_path, "w", encoding="utf-8") as log_file:
        subprocess.Popen(
            command,
            cwd=str(settings.base_dir),
            stdout=log_file,
            stderr=log_file,
        )

    return {
        "status": "started",
        "job_id": job_id,
        "message": "File uploaded and processing started",
    }


@router.get("/result/{job_id}")
def get_result(job_id: str, authorization: str | None = Header(default=None)):
    check_auth(authorization)

    job_dir = settings.results_dir / job_id
    summary_file = job_dir / "summary.txt"
    transcript_file = job_dir / "transcript.txt"
    log_file = job_dir / "run.log"

    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="Job not found")

    result = {"job_id": job_id, "files": [p.name for p in job_dir.iterdir() if p.is_file()]}

    if transcript_file.exists():
        result["transcript"] = transcript_file.read_text(encoding="utf-8", errors="replace")

    if summary_file.exists():
        result["summary"] = summary_file.read_text(encoding="utf-8", errors="replace")

    if log_file.exists():
        result["log_tail"] = tail(log_file)

    return result


@router.get("/download/{job_id}/{filename}")
def download_file(
    job_id: str,
    filename: str,
    authorization: str | None = Header(default=None),
):
    check_auth(authorization)

    if filename not in ALLOWED_DOWNLOADS:
        raise HTTPException(status_code=403, detail="File is not allowed for download")

    job_dir = settings.results_dir / job_id
    file_path = job_dir / filename

    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="Job not found")

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(path=str(file_path), filename=filename, media_type="application/octet-stream")

