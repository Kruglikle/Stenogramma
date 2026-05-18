from pydantic import BaseModel


class LoginRequest(BaseModel):
    username: str
    password: str


class AddUserRequest(BaseModel):
    username: str
    password: str


class ProcessUrlRequest(BaseModel):
    source_url: str
    transcription_model: str | None = None
