import json
import os
import re
import shutil
from pathlib import Path

from audio_transcribator.config import settings
from audio_transcribator.utils.files import write_text_atomic


LOCAL_PIPELINE_TEMPLATE = """version: 3.1.0
pipeline:
  name: pyannote.audio.pipelines.SpeakerDiarization
  params:
    clustering: AgglomerativeClustering
    embedding: {embedding_model_dir}
    embedding_batch_size: 32
    embedding_exclude_overlap: true
    segmentation:
      checkpoint: {segmentation_checkpoint}
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


def model_checkpoint(model_dir: Path) -> Path:
    checkpoint = model_dir / "pytorch_model.bin"
    if checkpoint.exists():
        return checkpoint

    candidates = sorted(model_dir.rglob("pytorch_model.bin"))
    if candidates:
        return candidates[0]

    raise RuntimeError(f"Local pyannote model checkpoint was not found in {model_dir}")


def embedding_source_checkpoint(model_dir: Path) -> Path:
    checkpoint = model_dir / "speaker-embedding.onnx"
    if checkpoint.exists():
        return checkpoint

    candidates = sorted(model_dir.rglob("*.onnx"))
    if candidates:
        return candidates[0]

    raise RuntimeError(
        f"Local WeSpeaker ONNX checkpoint was not found in {model_dir}. "
        "Download hbredin/wespeaker-voxceleb-resnet34-LM for pyannote 3.1."
    )


def embedding_checkpoint(model_dir: Path) -> Path:
    source = embedding_source_checkpoint(model_dir)
    if "pyannote" not in str(source).lower() and "wespeaker" in str(source).lower():
        return source

    alias = settings.pyannote_model_dir.parent / "wespeaker-voxceleb-resnet34-LM.onnx"
    alias.parent.mkdir(parents=True, exist_ok=True)
    if not alias.exists() or alias.stat().st_size != source.stat().st_size:
        shutil.copy2(source, alias)
    return alias


def ensure_local_pipeline_config() -> Path:
    config_path = settings.pyannote_pipeline_config

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
            segmentation_checkpoint=yaml_path(model_checkpoint(settings.pyannote_segmentation_model)),
            embedding_model_dir=yaml_path(embedding_checkpoint(settings.pyannote_embedding_model)),
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


def build_pipeline_kwargs(diarization_speakers: int | None = None) -> dict:
    kwargs = {}
    requested_speakers = diarization_speakers or settings.diarization_speakers
    if requested_speakers > 0:
        kwargs["num_speakers"] = requested_speakers
        return kwargs
    if settings.diarization_min_speakers > 0:
        kwargs["min_speakers"] = settings.diarization_min_speakers
    if settings.diarization_max_speakers > 0:
        kwargs["max_speakers"] = settings.diarization_max_speakers
    return kwargs


class DiarizationProgressHook:
    def __init__(self, progress_callback):
        self.progress_callback = progress_callback
        self.steps = []

    def __call__(self, step_name, step_artifact, file=None, total=None, completed=None):
        if not self.progress_callback:
            return
        if step_name not in self.steps:
            self.steps.append(step_name)
        step_index = self.steps.index(step_name)
        if completed is None or total in {None, 0}:
            fraction = 0.0
        else:
            fraction = max(0.0, min(float(completed) / float(total), 1.0))

        expected_steps = max(4, len(self.steps))
        percent = min(((step_index + fraction) / expected_steps) * 100, 99)
        self.progress_callback(percent, f"Этап pyannote: {step_name}")


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


def diarize(
    audio_file: Path,
    job_dir: Path,
    diarization_speakers: int | None = None,
    progress_callback=None,
) -> list[dict]:
    print("Running local pyannote speaker diarization 3.1...", flush=True)
    if progress_callback:
        progress_callback(1, "Загрузка локального pipeline pyannote")
    pipeline = load_pipeline()
    if progress_callback:
        progress_callback(5, "Запуск pyannote")
    kwargs = build_pipeline_kwargs(diarization_speakers)
    if progress_callback:
        diarization = pipeline(str(audio_file), hook=DiarizationProgressHook(progress_callback), **kwargs)
    else:
        diarization = pipeline(str(audio_file), **kwargs)
    if progress_callback:
        progress_callback(99, "Сборка стенограммы со спикерами")
    turns = annotation_to_turns(diarization)
    write_diarization_outputs(job_dir, diarization, turns)
    write_diarized_transcript(job_dir, turns)
    print(f"Diarization completed: {len(turns)} speaker turns.", flush=True)
    if progress_callback:
        progress_callback(100)
    return turns
