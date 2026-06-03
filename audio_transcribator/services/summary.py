import json
from pathlib import Path

from openai import OpenAI

from audio_transcribator.config import settings
from audio_transcribator.utils.files import write_text_atomic


SUMMARY_SYSTEM_PROMPT = """Ты профессиональный русскоязычный специалист по сжатию текста, редактор аналитических резюме и стенограмм.

Твоя задача - превратить расшифровку аудио или видео в короткое, точное и полезное резюме.
Сохраняй факты, имена, цифры, договоренности и причинно-следственные связи.
Не добавляй фактов, которых нет в стенограмме. Не цитируй длинные фрагменты. Не пиши рассуждения о своей работе.
Если в тексте есть ошибки распознавания речи, исправляй только очевидные ошибки по контексту.
Пиши по-русски, деловым и понятным стилем."""

SUMMARY_USER_PROMPT = """Сделай профессиональное сжатое резюме этой расшифровки.

Формат ответа:

## Краткое резюме
3-6 предложений: о чем материал и главный смысл.

## Ключевые темы
5-10 пунктов.

## Важные детали
Факты, цифры, имена, решения, договоренности, сроки. Если таких деталей нет, напиши "Не указано".

## Вывод
1-3 предложения: зачем этот материал может быть полезен.

Расшифровка:
{transcript}
"""


def strip_model_thinking(text: str) -> str:
    result = text.strip()
    while "<think>" in result and "</think>" in result:
        start = result.find("<think>")
        end = result.find("</think>", start) + len("</think>")
        result = (result[:start] + result[end:]).strip()
    return result


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
        temperature=0.2,
        messages=[
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": SUMMARY_USER_PROMPT.format(transcript=transcript)},
        ],
    )

    result = strip_model_thinking(response.choices[0].message.content or "")
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
