import json
from pathlib import Path

from openai import OpenAI

from audio_transcribator.config import settings
from audio_transcribator.services.editor_models import resolve_editor_model


EDITOR_PROMPT = """Ты профессиональный русскоязычный редактор стенограмм.

Исправь орфографические, пунктуационные и очевидные грамматические ошибки.
Приведи текст в аккуратный, читаемый вид: расставь абзацы, восстанови знаки препинания,
убери повторы, слова-паразиты и сбои распознавания речи, если они очевидны из контекста.

Не добавляй новых фактов, не сокращай смысл и не меняй порядок высказываний.
Если в тексте есть имена спикеров, таймкоды, списки или технические термины, сохрани их.
Верни только отредактированный текст без комментариев и пояснений.

Стенограмма:
{transcript}
"""


def edit_transcript(transcript: str, job_dir: Path, model: str | None = None) -> str:
    selected_model = resolve_editor_model(model, settings.editor_model)

    if selected_model["provider"] == "openrouter" and not settings.openrouter_api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required for OpenRouter AI editing")

    if selected_model["provider"] == "ollama":
        base_url = settings.ollama_base_url.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"
        client = OpenAI(api_key=settings.ollama_api_key, base_url=base_url)
    else:
        client = OpenAI(api_key=settings.openrouter_api_key, base_url=settings.openrouter_base_url)

    print(f"Editing transcript with {selected_model['model']}...")
    response = client.chat.completions.create(
        model=selected_model["model"],
        temperature=settings.editor_temperature,
        messages=[{"role": "user", "content": EDITOR_PROMPT.format(transcript=transcript)}],
    )

    result = response.choices[0].message.content or ""
    (job_dir / "edited_transcript.txt").write_text(result, encoding="utf-8")

    usage = getattr(response, "usage", None)
    if usage:
        usage_data = usage.model_dump() if hasattr(usage, "model_dump") else dict(usage)
        (job_dir / "editing_usage.json").write_text(
            json.dumps(
                {
                    "provider": selected_model["provider"],
                    "model": selected_model["model"],
                    "usage": usage_data,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    print("Edited transcript saved.")
    return result
