import argparse
import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download


PIPELINE_TEMPLATE = """version: 3.1.0
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


def yaml_path(path: Path) -> str:
    return json.dumps(str(path.resolve()).replace("\\", "/"), ensure_ascii=False)


def download_repo(repo_id: str, target_dir: Path, token: str | None) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {repo_id} -> {target_dir}")
    snapshot_download(
        repo_id,
        local_dir=str(target_dir),
        token=token,
        local_dir_use_symlinks=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download pyannote diarization models for offline runtime.")
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=Path(os.getenv("PYANNOTE_MODEL_DIR", "data/model_cache/pyannote")),
    )
    parser.add_argument("--token", default=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN"))
    args = parser.parse_args()

    if not args.token:
        raise SystemExit("Set HF_TOKEN/HUGGINGFACE_TOKEN or pass --token for the one-time model download.")

    target_dir = args.target_dir.resolve()
    pipeline_dir = target_dir / "speaker-diarization-3.1"
    segmentation_dir = target_dir / "segmentation-3.0"
    embedding_dir = target_dir / "wespeaker-voxceleb-resnet34-LM"

    download_repo("pyannote/speaker-diarization-3.1", pipeline_dir, args.token)
    download_repo("pyannote/segmentation-3.0", segmentation_dir, args.token)
    download_repo("pyannote/wespeaker-voxceleb-resnet34-LM", embedding_dir, args.token)

    config_path = pipeline_dir / "config.yaml"
    config_path.write_text(
        PIPELINE_TEMPLATE.format(
            segmentation_model=yaml_path(segmentation_dir),
            embedding_model=yaml_path(embedding_dir),
        ),
        encoding="utf-8",
    )
    print(f"Wrote offline pipeline config: {config_path}")


if __name__ == "__main__":
    main()
