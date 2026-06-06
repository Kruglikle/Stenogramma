import json
import os
import re
from pathlib import Path

from audio_transcribator.config import settings
from audio_transcribator.utils.files import write_text_atomic


LOCAL_PIPELINE_TEMPLATE = """version: 3.1.0
pipeline:
  name: pyannote.audio.pipelines.SpeakerDiarization
  params:
    clustering: AgglomerativeClustering
    embedding: {embedding_model}
    embedding_batch_size: 32
    embedding_exclude_overlap: true
    segmentation: {segmentation_model}
    segmentation_batch_size: 32
params:
  clustering:
    method: centroid
    min_cluster_size: 12
    threshold: 0.7045654963945799
  segmentation:
    min_duration_off: 0.0
"""


def format_timestamp(seconds: float) -> str:
    total_ms = max(int(round(seconds * 1000)), 0)
    ms = total_ms % 1000
    total_seconds = total_ms // 1000
    seconds_part = total_seconds % 60
    total_minutes = total_seconds // 60
    minutes = total_minutes % 60
    hours = total_minutes // 60
    return f"{hours:02d}:{minutes:02d}:{seconds_part:02d}.{ms:03d}"


def yaml_path(path: Path) -> str:
    return json.dumps(str(path).replace("\\", "/"), ensure_ascii=False)


def ensure_local_pipeline_config() -> Path:
    config_path = settings.pyannote_pipeline_config
    if os.getenv("PYANNOTE_PIPELINE_CONFIG") and config_path.exists():
        return config_path

    if not settings.pyannote_segmentation_model.exists():
        raise RuntimeError(
            "Local pyannote segmentation model was not found. "
            f"Expected: {settings.pyannote_segmentation_model}. "
            "Download models once before enabling diarization."
        )
    if not settings.pyannote_embedding_model.exists():
        raise RuntimeError(
            "Local pyannote embedding model was not found. "
            f"Expected: {settings.pyannote_embedding_model}. "
            "Download models once before enabling diarization."
        )

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        LOCAL_PIPELINE_TEMPLATE.format(
            segmentation_model=yaml_path(settings.pyannote_segmentation_model),
            embedding_model=yaml_path(settings.pyannote_embedding_model),
        ),
        encoding="utf-8",
    )
    return config_path


def load_pipeline():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        import torch
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise RuntimeError("Install pyannote.audio==3.1.1 to enable pyannote diarization") from exc

    config_path = ensure_local_pipeline_config()
    print(f"Loading local pyannote pipeline: {config_path}", flush=True)
    pipeline = Pipeline.from_pretrained(str(config_path))

    device_name = settings.pyannote_device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name:
        pipeline.to(torch.device(device_name))
        print(f"Pyannote diarization device: {device_name}", flush=True)
    return pipeline


def build_pipeline_kwargs() -> dict:
    kwargs = {}
    if settings.diarization_speakers > 0:
        kwargs["num_speakers"] = settings.diarization_speakers
        return kwargs
    if settings.diarization_min_speakers > 0:
        kwargs["min_speakers"] = settings.diarization_min_speakers
    if settings.diarization_max_speakers > 0:
        kwargs["max_speakers"] = settings.diarization_max_speakers
    return kwargs


def speaker_sort_key(speaker: str) -> tuple[str, int]:
    match = re.search(r"(\d+)$", speaker)
    return (speaker[: match.start()] if match else speaker, int(match.group(1)) if match else 0)


def normalize_speaker_labels(turns: list[dict]) -> list[dict]:
    ordered_speakers = []
    for turn in turns:
        speaker = turn["speaker"]
        if speaker not in ordered_speakers:
            ordered_speakers.append(speaker)
    speaker_map = {
        speaker: f"Спикер {index}"
        for index, speaker in enumerate(sorted(ordered_speakers, key=speaker_sort_key), start=1)
    }
    return [
        {
            "speaker": speaker_map[turn["speaker"]],
            "raw_speaker": turn["speaker"],
            "start": turn["start"],
            "end": turn["end"],
        }
        for turn in turns
    ]


def annotation_to_turns(annotation) -> list[dict]:
    turns = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        turns.append(
            {
                "speaker": str(speaker),
                "start": float(turn.start),
                "end": float(turn.end),
            }
        )
    turns.sort(key=lambda item: (item["start"], item["end"]))
    return normalize_speaker_labels(turns)


def write_diarization_outputs(job_dir: Path, annotation, turns: list[dict]) -> None:
    lines = [
        f"{format_timestamp(item['start'])} - {format_timestamp(item['end'])}: {item['speaker']}"
        for item in turns
    ]
    write_text_atomic(job_dir / "diarization.txt", "\n".join(lines))
    (job_dir / "diarization.json").write_text(
        json.dumps(turns, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with open(job_dir / "diarization.rttm", "w", encoding="utf-8") as rttm:
        annotation.write_rttm(rttm)


def load_transcript_segments(job_dir: Path) -> list[dict]:
    segments_path = job_dir / "transcript_segments.json"
    if not segments_path.exists():
        return []
    try:
        data = json.loads(segments_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    return [
        {
            "start": float(item.get("start", 0)),
            "end": float(item.get("end", 0)),
            "text": str(item.get("text", "")).strip(),
        }
        for item in data
        if isinstance(item, dict) and str(item.get("text", "")).strip()
    ]


def overlap_seconds(left: dict, right: dict) -> float:
    return max(min(left["end"], right["end"]) - max(left["start"], right["start"]), 0.0)


def find_segment_speaker(segment: dict, turns: list[dict]) -> str:
    best_turn = None
    best_overlap = 0.0
    for turn in turns:
        overlap = overlap_seconds(segment, turn)
        if overlap > best_overlap:
            best_overlap = overlap
            best_turn = turn
    if best_turn and best_overlap > 0:
        return best_turn["speaker"]
    midpoint = (segment["start"] + segment["end"]) / 2
    for turn in turns:
        if turn["start"] <= midpoint <= turn["end"]:
            return turn["speaker"]
    return "Спикер ?"


def write_diarized_transcript(job_dir: Path, turns: list[dict]) -> None:
    segments = load_transcript_segments(job_dir)
    if not segments:
        return

    lines = []
    for segment in segments:
        speaker = find_segment_speaker(segment, turns)
        lines.append(
            f"{format_timestamp(segment['start'])} - {format_timestamp(segment['end'])} | "
            f"{speaker}: {segment['text']}"
        )
    write_text_atomic(job_dir / "diarized_transcript.txt", "\n".join(lines))


def diarize(audio_file: Path, job_dir: Path) -> list[dict]:
    print("Running local pyannote speaker diarization 3.1...", flush=True)
    pipeline = load_pipeline()
    diarization = pipeline(str(audio_file), **build_pipeline_kwargs())
    turns = annotation_to_turns(diarization)
    write_diarization_outputs(job_dir, diarization, turns)
    write_diarized_transcript(job_dir, turns)
    print(f"Diarization completed: {len(turns)} speaker turns.", flush=True)
    return turns
