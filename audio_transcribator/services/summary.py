import json
from pathlib import Path

from openai import OpenAI

from audio_transcribator.config import settings
from audio_transcribator.utils.files import write_text_atomic


SUMMARY_PROMPT = """Сделай анализ расшифровки.

Выведи:
1. Краткое резюме
2. Основные темы
3. Ключевые слова
4. Возможное назначение материала

Расшифровка:
{transcript}
"""


def summarize(transcript: str, job_dir: Path) -> str | None:
    if not transcript.strip():
        print("Transcript is empty, skipping summary")
        return None

    base_url = settings.ollama_base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"

    print(f"Summarizing with local model {settings.summary_model}...")
    client = OpenAI(api_key=settings.ollama_api_key, base_url=base_url)

    response = client.chat.completions.create(
        model=settings.summary_model,
        messages=[{"role": "user", "content": SUMMARY_PROMPT.format(transcript=transcript)}],
    )

    result = response.choices[0].message.content or ""
    write_text_atomic(job_dir / "summary.txt", result)

    usage = getattr(response, "usage", None)
    if usage:
        usage_data = usage.model_dump() if hasattr(usage, "model_dump") else dict(usage)
        (job_dir / "summary_usage.json").write_text(
            json.dumps(
                {
                    "provider": "ollama",
                    "model": settings.summary_model,
                    "usage": usage_data,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    print("Summary saved.")
    return result
