import json
from pathlib import Path

from openai import OpenAI

from audio_transcribator.config import settings


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
    if not settings.openrouter_api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required for AI editing")

    selected_model = model or settings.editor_model
    print(f"Editing transcript with {selected_model}...")

    client = OpenAI(api_key=settings.openrouter_api_key, base_url=settings.openrouter_base_url)
    response = client.chat.completions.create(
        model=selected_model,
        temperature=settings.editor_temperature,
        messages=[{"role": "user", "content": EDITOR_PROMPT.format(transcript=transcript)}],
    )

    result = response.choices[0].message.content or ""
    (job_dir / "edited_transcript.txt").write_text(result, encoding="utf-8")

    usage = getattr(response, "usage", None)
    if usage:
        usage_data = usage.model_dump() if hasattr(usage, "model_dump") else dict(usage)
        (job_dir / "editing_usage.json").write_text(
            json.dumps({"model": selected_model, "usage": usage_data}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print("Edited transcript saved.")
    return result
