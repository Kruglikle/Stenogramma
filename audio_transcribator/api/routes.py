from fastapi import APIRouter, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse

from audio_transcribator.auth import check_auth, verify_credentials
from audio_transcribator.config import settings
from audio_transcribator.models import LoginRequest
from audio_transcribator.services.jobs import build_job_result, get_job_file, start_uploaded_file
from audio_transcribator.utils.files import ALLOWED_DOWNLOADS


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

    return start_uploaded_file(file)


@router.get("/result/{job_id}")
def get_result(job_id: str, authorization: str | None = Header(default=None)):
    check_auth(authorization)

    try:
        return build_job_result(job_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Job not found")


@router.get("/download/{job_id}/{filename}")
def download_file(
    job_id: str,
    filename: str,
    authorization: str | None = Header(default=None),
):
    check_auth(authorization)

    if filename not in ALLOWED_DOWNLOADS:
        raise HTTPException(status_code=403, detail="File is not allowed for download")

    try:
        file_path = get_job_file(job_id, filename)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(path=str(file_path), filename=filename, media_type="application/octet-stream")
