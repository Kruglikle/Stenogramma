from fastapi import APIRouter, Cookie, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from audio_transcribator.auth import verify_credentials
from audio_transcribator.config import settings
from audio_transcribator.services.jobs import build_job_result, get_job_file, start_uploaded_file
from audio_transcribator.services.transcription_models import (
    DEFAULT_TRANSCRIPTION_MODEL_ID,
    TranscriptionModelError,
    list_transcription_models,
)
from audio_transcribator.utils.files import ALLOWED_DOWNLOADS


router = APIRouter(prefix="/ui", include_in_schema=False)
templates = Jinja2Templates(directory=str(settings.base_dir / "audio_transcribator" / "templates"))


def require_ui_auth(ui_token: str | None) -> None:
    if ui_token != settings.api_token:
        raise HTTPException(status_code=status.HTTP_303_SEE_OTHER, headers={"Location": "/ui/login"})


@router.get("", response_class=HTMLResponse)
def ui_root(ui_token: str | None = Cookie(default=None)):
    if ui_token == settings.api_token:
        return RedirectResponse(url="/ui/upload", status_code=status.HTTP_303_SEE_OTHER)
    return RedirectResponse(url="/ui/login", status_code=status.HTTP_303_SEE_OTHER)


@router.get("/login", response_class=HTMLResponse)
def login_page(request: Request, ui_token: str | None = Cookie(default=None)):
    if ui_token == settings.api_token:
        return RedirectResponse(url="/ui/upload", status_code=status.HTTP_303_SEE_OTHER)
    return templates.TemplateResponse(request, "login.html", {"error": None})


@router.post("/login")
def login(request: Request, username: str = Form(...), password: str = Form(...)):
    if not verify_credentials(username, password):
        return templates.TemplateResponse(
            request,
            "login.html",
            {"error": "Invalid username or password"},
            status_code=status.HTTP_401_UNAUTHORIZED,
        )

    response = RedirectResponse(url="/ui/upload", status_code=status.HTTP_303_SEE_OTHER)
    response.set_cookie("ui_token", settings.api_token, httponly=True, samesite="lax")
    return response


@router.post("/logout")
def logout():
    response = RedirectResponse(url="/ui/login", status_code=status.HTTP_303_SEE_OTHER)
    response.delete_cookie("ui_token")
    return response


@router.get("/upload", response_class=HTMLResponse)
def upload_page(request: Request, ui_token: str | None = Cookie(default=None)):
    require_ui_auth(ui_token)
    return templates.TemplateResponse(
        request,
        "upload.html",
        {
            "transcription_models": list_transcription_models(),
            "selected_transcription_model": DEFAULT_TRANSCRIPTION_MODEL_ID,
            "error": None,
        },
    )


@router.post("/upload")
def upload_file(
    request: Request,
    file: UploadFile = File(...),
    transcription_model: str = Form(DEFAULT_TRANSCRIPTION_MODEL_ID),
    ui_token: str | None = Cookie(default=None),
):
    require_ui_auth(ui_token)
    try:
        result = start_uploaded_file(file, transcription_model_id=transcription_model)
    except TranscriptionModelError as exc:
        return templates.TemplateResponse(
            request,
            "upload.html",
            {
                "transcription_models": list_transcription_models(),
                "selected_transcription_model": transcription_model,
                "error": str(exc),
            },
            status_code=status.HTTP_400_BAD_REQUEST,
        )
    return RedirectResponse(url=f"/ui/result/{result['job_id']}", status_code=status.HTTP_303_SEE_OTHER)


@router.get("/result/{job_id}", response_class=HTMLResponse)
def result_page(request: Request, job_id: str, ui_token: str | None = Cookie(default=None)):
    require_ui_auth(ui_token)
    try:
        result = build_job_result(job_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Job not found")

    downloads = [name for name in result["files"] if name in ALLOWED_DOWNLOADS]
    return templates.TemplateResponse(
        request,
        "result.html",
        {"result": result, "downloads": downloads},
    )


@router.get("/download/{job_id}/{filename}")
def download_file(job_id: str, filename: str, ui_token: str | None = Cookie(default=None)):
    require_ui_auth(ui_token)
    if filename not in ALLOWED_DOWNLOADS:
        raise HTTPException(status_code=403, detail="File is not allowed for download")

    try:
        file_path = get_job_file(job_id, filename)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(path=str(file_path), filename=filename, media_type="application/octet-stream")
