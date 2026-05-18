from fastapi import APIRouter, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from psycopg.errors import UniqueViolation

from audio_transcribator.auth import check_add_user_auth, check_auth, verify_credentials
from audio_transcribator.config import settings
from audio_transcribator.db import create_user
from audio_transcribator.models import AddUserRequest, LoginRequest, ProcessUrlRequest
from audio_transcribator.services.jobs import build_job_result, get_job_file, start_uploaded_file, start_url
from audio_transcribator.services.transcription_models import (
    DEFAULT_TRANSCRIPTION_MODEL_ID,
    TranscriptionModelError,
    list_transcription_models,
)
from audio_transcribator.utils.files import ALLOWED_DOWNLOADS


router = APIRouter()


@router.post("/login")
def login(data: LoginRequest):
    if not verify_credentials(data.username, data.password):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    return {"access_token": settings.api_token, "token_type": "bearer"}


@router.post("/add-user", status_code=201)
def add_user(data: AddUserRequest, authorization: str | None = Header(default=None)):
    check_add_user_auth(authorization)

    try:
        create_user(data.username, data.password)
    except ValueError:
        raise HTTPException(status_code=400, detail="Username and password are required")
    except UniqueViolation:
        raise HTTPException(status_code=409, detail="User already exists")

    return {"username": data.username, "created": True}


@router.get("/")
def root():
    return {"status": "ok", "service": "audio-video-processing"}


@router.post("/process")
async def process_file(
    file: UploadFile = File(...),
    transcription_model: str = Form(DEFAULT_TRANSCRIPTION_MODEL_ID),
    authorization: str | None = Header(default=None),
):
    check_auth(authorization)

    try:
        return start_uploaded_file(file, transcription_model_id=transcription_model)
    except TranscriptionModelError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/process-url")
def process_url(
    data: ProcessUrlRequest,
    authorization: str | None = Header(default=None),
):
    check_auth(authorization)

    try:
        return start_url(data.source_url, transcription_model_id=data.transcription_model)
    except (TranscriptionModelError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/transcription-models")
def get_transcription_models(authorization: str | None = Header(default=None)):
    check_auth(authorization)

    return {"models": list_transcription_models(), "default": DEFAULT_TRANSCRIPTION_MODEL_ID}


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
