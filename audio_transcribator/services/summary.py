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

PART_SUMMARY_PROMPT = """Сожми эту часть расшифровки в рабочее резюме для последующей финальной сборки.

Верни только:
- главные тезисы;
- факты, имена, цифры, решения, договоренности, сроки;
- важные контекстные детали.

Не добавляй вступление и не пиши рассуждения.

Часть {part_number} из {parts_count}:
{transcript_part}
"""

FINAL_SUMMARY_PROMPT = """Сделай профессиональное финальное резюме по рабочим резюме частей расшифровки.

Формат ответа:

## Краткое резюме
3-6 предложений: о чем материал и главный смысл.

## Ключевые темы
5-10 пунктов.

## Важные детали
Факты, цифры, имена, решения, договоренности, сроки. Если таких деталей нет, напиши "Не указано".

## Вывод
1-3 предложения: зачем этот материал может быть полезен.

Рабочие резюме частей:
{partial_summaries}
"""


def strip_model_thinking(text: str) -> str:
    result = text.strip()
    while "<think>" in result and "</think>" in result:
        start = result.find("<think>")
        end = result.find("</think>", start) + len("</think>")
        result = (result[:start] + result[end:]).strip()
    return result


def split_transcript(text: str, chunk_chars: int) -> list[str]:
    normalized = text.strip()
    if len(normalized) <= chunk_chars:
        return [normalized]

    chunks = []
    current = []
    current_len = 0
    for paragraph in normalized.splitlines():
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        paragraph_len = len(paragraph)
        if current and current_len + paragraph_len + 1 > chunk_chars:
            chunks.append("\n".join(current))
            current = []
            current_len = 0

        if paragraph_len > chunk_chars:
            for start in range(0, paragraph_len, chunk_chars):
                part = paragraph[start : start + chunk_chars].strip()
                if part:
                    chunks.append(part)
            continue

        current.append(paragraph)
        current_len += paragraph_len + 1

    if current:
        chunks.append("\n".join(current))
    return chunks


def build_ollama_client() -> OpenAI:
    base_url = settings.ollama_base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    return OpenAI(
        api_key=settings.ollama_api_key,
        base_url=base_url,
        timeout=settings.ollama_request_timeout_seconds,
    )


def call_summary_model(client: OpenAI, prompt: str) -> str:
    response = client.chat.completions.create(
        model=settings.summary_model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return strip_model_thinking(response.choices[0].message.content or "")


def write_summary_usage(job_dir: Path, usage_items: list[dict]) -> None:
    if not usage_items:
        return

    (job_dir / "summary_usage.json").write_text(
        json.dumps(
            {
                "provider": "ollama",
                "model": settings.summary_model,
                "usage": usage_items,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def summarize(transcript: str, job_dir: Path, progress_callback=None) -> str | None:
    if not transcript.strip():
        print("Transcript is empty, skipping summary")
        return None

    print(f"Summarizing with local model {settings.summary_model}...")
    client = build_ollama_client()
    chunks = split_transcript(transcript, max(settings.summary_chunk_chars, 2000))
    partial_summaries = []
    usage_items = []

    for index, chunk in enumerate(chunks, start=1):
        print(f"Summarizing chunk {index}/{len(chunks)}...", flush=True)
        prompt = PART_SUMMARY_PROMPT.format(
            part_number=index,
            parts_count=len(chunks),
            transcript_part=chunk,
        )
        partial = call_summary_model(client, prompt)
        if partial:
            partial_summaries.append(f"### Часть {index}\n{partial}")
            write_text_atomic(
                job_dir / "summary.txt",
                "Резюме готовится. Уже обработано частей: "
                f"{index}/{len(chunks)}.\n\n" + "\n\n".join(partial_summaries),
            )
        usage_items.append({"stage": "chunk", "chunk": index})
        if progress_callback:
            progress_callback(index / (len(chunks) + 1) * 100, f"Обработано частей: {index}/{len(chunks)}")

    partial_text = "\n\n".join(partial_summaries)
    final_prompt = FINAL_SUMMARY_PROMPT.format(partial_summaries=partial_text)
    if progress_callback:
        progress_callback(len(chunks) / (len(chunks) + 1) * 100, "Финальная сборка резюме")
    result = call_summary_model(client, final_prompt)
    write_text_atomic(job_dir / "summary.txt", result)
    usage_items.append({"stage": "final", "chunks": len(chunks)})
    write_summary_usage(job_dir, usage_items)

    print("Summary saved.")
    if progress_callback:
        progress_callback(100)
    return result
