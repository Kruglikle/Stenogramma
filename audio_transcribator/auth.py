from fastapi import Header, HTTPException

from audio_transcribator.config import settings


def check_auth(authorization: str | None = Header(default=None)) -> None:
    if authorization != f"Bearer {settings.api_token}":
        raise HTTPException(status_code=401, detail="Unauthorized")


def verify_credentials(username: str, password: str) -> bool:
    return username == settings.api_username and password == settings.api_password

