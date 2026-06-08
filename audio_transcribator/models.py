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
    enable_summary: bool = True
    enable_diarization: bool = False
    diarization_speakers: int = 0
