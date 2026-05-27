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
    if not settings.openrouter_api_key:
        print("OPENROUTER_API_KEY is not set, skipping summary")
        return None

    print("Summarizing...")
    client = OpenAI(api_key=settings.openrouter_api_key, base_url=settings.openrouter_base_url)

    response = client.chat.completions.create(
        model=settings.summary_model,
        messages=[{"role": "user", "content": SUMMARY_PROMPT.format(transcript=transcript)}],
    )

    result = response.choices[0].message.content or ""
    write_text_atomic(job_dir / "summary.txt", result)

    print("Summary saved.")
    return result
