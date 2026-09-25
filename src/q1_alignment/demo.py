from __future__ import annotations

from pathlib import Path

import numpy as np

from .common import read_jsonl


def make_demo_features(
    manifest_path: str | Path,
    timeline_path: str | Path,
    output_dir: str | Path,
    limit: int = 3,
) -> list[Path]:
    manifest = read_jsonl(manifest_path)
    timelines = {row["id"]: row for row in read_jsonl(timeline_path)}
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    for sample_index, sample in enumerate(manifest[:limit]):
        timeline = timelines[sample["id"]]
        word_count = len(timeline["words"])
        duration = float(sample["duration_sec"])
        rng = np.random.default_rng(20260923 + sample_index)

        audio_count = max(1, int(np.ceil(duration * 100)))
        audio_centers = (np.arange(audio_count) + 0.5) * duration / audio_count
        audio_step = duration / audio_count
        audio_times = np.column_stack(
            [np.maximum(0, audio_centers - audio_step / 2), np.minimum(duration, audio_centers + audio_step / 2)]
        )

        vision_count = max(1, int(sample["frame_count"]))
        vision_centers = (np.arange(vision_count) + 0.5) * duration / vision_count
        vision_step = duration / vision_count
        vision_times = np.column_stack(
            [np.maximum(0, vision_centers - vision_step / 2), np.minimum(duration, vision_centers + vision_step / 2)]
        )

        output = output_root / f"{sample['safe_id']}.npz"
        np.savez_compressed(
            output,
            text_features=rng.normal(size=(word_count, 16)).astype(np.float32),
            audio_features=rng.normal(size=(audio_count, 8)).astype(np.float32),
            audio_times=audio_times.astype(np.float64),
            audio_valid=np.ones(audio_count, dtype=np.uint8),
            vision_features=rng.normal(size=(vision_count, 6)).astype(np.float32),
            vision_times=vision_times.astype(np.float64),
            vision_valid=np.ones(vision_count, dtype=np.uint8),
        )
        outputs.append(output)
    return outputs

