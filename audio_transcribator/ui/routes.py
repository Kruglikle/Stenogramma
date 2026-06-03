import hashlib
import hmac
import time

from fastapi import APIRouter, Cookie, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from audio_transcribator.auth import verify_credentials
from audio_transcribator.config import settings
from audio_transcribator.services.editor import edit_transcript
from audio_transcribator.services.editor_models import list_editor_model_groups
from audio_transcribator.services.jobs import (
    build_job_result,
    get_job_file,
    list_user_jobs,
    load_job_metadata,
    save_job_timing,
    start_uploaded_file,
    start_url,
    update_job_title,
)
from audio_transcribator.services.transcription_models import (
    DEFAULT_TRANSCRIPTION_MODEL_ID,
    TranscriptionModelError,
    list_transcription_models,
)
from audio_transcribator.utils.files import ALLOWED_DOWNLOADS


router = APIRouter(prefix="/ui", include_in_schema=False)
templates = Jinja2Templates(directory=str(settings.base_dir / "audio_transcribator" / "templates"))


def sign_ui_user(username: str) -> str:
    return hmac.new(settings.api_token.encode("utf-8"), username.encode("utf-8"), hashlib.sha256).hexdigest()


def require_ui_auth(ui_token: str | None, ui_user: str | None = None, ui_user_sig: str | None = None) -> str:
    if ui_token != settings.api_token or not ui_user or not ui_user_sig:
        raise HTTPException(status_code=status.HTTP_303_SEE_OTHER, headers={"Location": "/ui/login"})

    if not hmac.compare_digest(sign_ui_user(ui_user), ui_user_sig):
        raise HTTPException(status_code=status.HTTP_303_SEE_OTHER, headers={"Location": "/ui/login"})

    return ui_user


def build_cabinet_context(username: str) -> dict:
    return {"cabinet_jobs": list_user_jobs(username), "current_username": username}


def can_access_job(username: str, job_id: str) -> bool:
    metadata = load_job_metadata(settings.results_dir / job_id)
    owner = metadata.get("user_login")
    return owner in {None, username}


@router.get("", response_class=HTMLResponse)
def ui_root(
    ui_token: str | None = Cookie(default=None),
    ui_user: str | None = Cookie(default=None),
    ui_user_sig: str | None = Cookie(default=None),
):
    try:
        require_ui_auth(ui_token, ui_user, ui_user_sig)
        return RedirectResponse(url="/ui/upload", status_code=status.HTTP_303_SEE_OTHER)
    except HTTPException:
        return RedirectResponse(url="/ui/login", status_code=status.HTTP_303_SEE_OTHER)


@router.get("/login", response_class=HTMLResponse)
def login_page(
    request: Request,
    ui_token: str | None = Cookie(default=None),
    ui_user: str | None = Cookie(default=None),
    ui_user_sig: str | None = Cookie(default=None),
):
    try:
        require_ui_auth(ui_token, ui_user, ui_user_sig)
        return RedirectResponse(url="/ui/upload", status_code=status.HTTP_303_SEE_OTHER)
    except HTTPException:
        pass
    return templates.TemplateResponse(request, "login.html", {"error": None})


@router.post("/login")
def login(request: Request, username: str = Form(...), password: str = Form(...)):
    if not verify_credentials(username, password):
        return templates.TemplateResponse(
            request,
            "login.html",
            {"error": "Неверный логин или пароль"},
            status_code=status.HTTP_401_UNAUTHORIZED,
        )

    response = RedirectResponse(url="/ui/upload", status_code=status.HTTP_303_SEE_OTHER)
    response.set_cookie("ui_token", settings.api_token, httponly=True, samesite="lax")
    response.set_cookie("ui_user", username.strip(), httponly=True, samesite="lax")
    response.set_cookie("ui_user_sig", sign_ui_user(username.strip()), httponly=True, samesite="lax")
    return response


@router.post("/logout")
def logout():
    response = RedirectResponse(url="/ui/login", status_code=status.HTTP_303_SEE_OTHER)
    response.delete_cookie("ui_token")
    response.delete_cookie("ui_user")
    response.delete_cookie("ui_user_sig")
    return response


@router.get("/upload", response_class=HTMLResponse)
def upload_page(
    request: Request,
    ui_token: str | None = Cookie(default=None),
    ui_user: str | None = Cookie(default=None),
    ui_user_sig: str | None = Cookie(default=None),
):
    username = require_ui_auth(ui_token, ui_user, ui_user_sig)
    return templates.TemplateResponse(
        request,
        "upload.html",
        {
            "transcription_models": list_transcription_models(),
            "selected_transcription_model": DEFAULT_TRANSCRIPTION_MODEL_ID,
            "error": None,
            **build_cabinet_context(username),
        },
    )


@router.post("/upload")
def upload_file(
    request: Request,
    file: UploadFile | None = File(default=None),
    source_url: str = Form(default=""),
    title: str = Form(default=""),
    transcription_model: str = Form(DEFAULT_TRANSCRIPTION_MODEL_ID),
    ui_token: str | None = Cookie(default=None),
    ui_user: str | None = Cookie(default=None),
    ui_user_sig: str | None = Cookie(default=None),
):
    username = require_ui_auth(ui_token, ui_user, ui_user_sig)
    try:
        clean_source_url = source_url.strip()
        if file and file.filename:
            result = start_uploaded_file(
                file,
                transcription_model_id=transcription_model,
                user_login=username,
                title=title,
            )
        elif clean_source_url:
            result = start_url(
                clean_source_url,
                transcription_model_id=transcription_model,
                user_login=username,
                title=title,
            )
        else:
            raise ValueError("Загрузите файл или вставьте ссылку на медиа")
    except (TranscriptionModelError, ValueError) as exc:
        return templates.TemplateResponse(
            request,
            "upload.html",
            {
                "transcription_models": list_transcription_models(),
                "selected_transcription_model": transcription_model,
                "error": str(exc),
                **build_cabinet_context(username),
            },
            status_code=status.HTTP_400_BAD_REQUEST,
        )
    return RedirectResponse(url=f"/ui/result/{result['job_id']}", status_code=status.HTTP_303_SEE_OTHER)


@router.get("/result/{job_id}", response_class=HTMLResponse)
def result_page(
    request: Request,
    job_id: str,
    ui_token: str | None = Cookie(default=None),
    ui_user: str | None = Cookie(default=None),
    ui_user_sig: str | None = Cookie(default=None),
):
    username = require_ui_auth(ui_token, ui_user, ui_user_sig)
    try:
        result = build_job_result(job_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Job not found")
    if not can_access_job(username, job_id):
        raise HTTPException(status_code=404, detail="Job not found")

    downloads = [name for name in result["files"] if name in ALLOWED_DOWNLOADS]
    return templates.TemplateResponse(
        request,
        "result.html",
        {
            "result": result,
            "downloads": downloads,
            "editor_model": settings.editor_model,
            "editor_model_groups": list_editor_model_groups(),
            **build_cabinet_context(username),
        },
    )


@router.post("/result/{job_id}/title")
def rename_result(
    job_id: str,
    title: str = Form(default=""),
    ui_token: str | None = Cookie(default=None),
    ui_user: str | None = Cookie(default=None),
    ui_user_sig: str | None = Cookie(default=None),
):
    username = require_ui_auth(ui_token, ui_user, ui_user_sig)
    if not can_access_job(username, job_id):
        raise HTTPException(status_code=404, detail="Job not found")
    update_job_title(job_id, title)
    return RedirectResponse(url=f"/ui/result/{job_id}", status_code=status.HTTP_303_SEE_OTHER)


@router.post("/result/{job_id}/edit", response_class=HTMLResponse)
def edit_result_transcript(
    request: Request,
    job_id: str,
    editor_model: str = Form(default=""),
    ui_token: str | None = Cookie(default=None),
    ui_user: str | None = Cookie(default=None),
    ui_user_sig: str | None = Cookie(default=None),
):
    username = require_ui_auth(ui_token, ui_user, ui_user_sig)
    try:
        result = build_job_result(job_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Job not found")
    if not can_access_job(username, job_id):
        raise HTTPException(status_code=404, detail="Job not found")

    transcript = result.get("transcript")
    if not transcript:
        raise HTTPException(status_code=400, detail="Transcript is not ready")

    try:
        job_dir = settings.results_dir / job_id
        started = time.perf_counter()
        edit_transcript(transcript, job_dir, model=editor_model)
        save_job_timing(job_dir, "editing", time.perf_counter() - started)
        return RedirectResponse(url=f"/ui/result/{job_id}", status_code=status.HTTP_303_SEE_OTHER)
    except (ValueError, RuntimeError) as exc:
        save_job_timing(settings.results_dir / job_id, "editing", time.perf_counter() - started, status="failed")
        result = build_job_result(job_id)
        downloads = [name for name in result["files"] if name in ALLOWED_DOWNLOADS]
        return templates.TemplateResponse(
            request,
            "result.html",
            {
                "result": result,
                "downloads": downloads,
                "editor_model": settings.editor_model,
                "editor_model_groups": list_editor_model_groups(),
                "editor_error": str(exc),
                **build_cabinet_context(username),
            },
            status_code=status.HTTP_400_BAD_REQUEST,
        )
    except Exception as exc:
        save_job_timing(settings.results_dir / job_id, "editing", time.perf_counter() - started, status="failed")
        result = build_job_result(job_id)
        downloads = [name for name in result["files"] if name in ALLOWED_DOWNLOADS]
        return templates.TemplateResponse(
            request,
            "result.html",
            {
                "result": result,
                "downloads": downloads,
                "editor_model": settings.editor_model,
                "editor_model_groups": list_editor_model_groups(),
                "editor_error": str(exc),
                **build_cabinet_context(username),
            },
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@router.get("/download/{job_id}/{filename}")
def download_file(
    job_id: str,
    filename: str,
    ui_token: str | None = Cookie(default=None),
    ui_user: str | None = Cookie(default=None),
    ui_user_sig: str | None = Cookie(default=None),
):
    username = require_ui_auth(ui_token, ui_user, ui_user_sig)
    if not can_access_job(username, job_id):
        raise HTTPException(status_code=404, detail="File not found")
    if filename not in ALLOWED_DOWNLOADS:
        raise HTTPException(status_code=403, detail="File is not allowed for download")

    try:
        file_path = get_job_file(job_id, filename)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(path=str(file_path), filename=filename, media_type="application/octet-stream")
