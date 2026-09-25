from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .alignment import align_sample
from .common import read_jsonl
from .timeline import validate_timeline


REQUIRED_BUNDLE_KEYS = {
    "audio_features",
    "audio_times",
    "vision_features",
    "vision_times",
    "text_features",
}


def load_bundle(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        missing = REQUIRED_BUNDLE_KEYS - set(data.files)
        if missing:
            raise ValueError(f"{path}: missing keys {sorted(missing)}")
        return {key: data[key] for key in data.files}


def run_alignment(
    manifest_path: str | Path,
    timeline_path: str | Path,
    feature_dir: str | Path,
    output_dir: str | Path,
    max_positions: int = 50,
    allow_placeholder: bool = False,
    allow_partial: bool = False,
) -> list[dict[str, Any]]:
    manifest = read_jsonl(manifest_path)
    timelines = {row["id"]: row for row in read_jsonl(timeline_path)}
    feature_root = Path(feature_dir)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    report: list[dict[str, Any]] = []

    for sample in manifest:
        sample_id = sample["id"]
        if sample_id not in timelines:
            raise ValueError(f"missing timeline: {sample_id}")
        timeline = timelines[sample_id]
        # A digitally silent clip is *expected* to have final_usable=False (R8:
        # no word time exists).  Treating that as a hard error would abort the
        # whole 100-sample run the moment real features arrive, which is exactly
        # what the interface dry-run caught on 2026-09-25.  It gets a supported
        # path instead: text stays real, audio/vision are masked out everywhere.
        audio_absent = not bool(
            (timeline.get("diagnostics") or {}).get("audio_present", True)
        )
        if not allow_placeholder and not timeline.get("final_usable", False) and not audio_absent:
            raise ValueError(f"timeline is not marked final_usable: {sample_id}")
        if not audio_absent:
            # Skipped for the sentinel-only timeline on purpose: it would report
            # "time outside clip / start >= end" for every word, which is true
            # but meaningless -- there is no time axis to validate.
            issues = validate_timeline(timeline, float(sample["duration_sec"]))
            if issues:
                raise ValueError(f"invalid timeline {sample_id}: {'; '.join(issues)}")

        bundle_path = feature_root / f"{sample['safe_id']}.npz"
        if not bundle_path.exists():
            if allow_partial:
                continue
            raise FileNotFoundError(f"missing feature bundle: {sample_id}: {bundle_path}")
        arrays, provenance = align_sample(
            sample, timeline, load_bundle(bundle_path), max_positions=max_positions
        )
        output_npz = output_root / f"{sample['safe_id']}.npz"
        output_json = output_root / f"{sample['safe_id']}.json"
        np.savez_compressed(output_npz, **arrays)
        output_json.write_text(
            json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        report.append(
            {
                "id": sample_id,
                "npz": str(output_npz),
                "json": str(output_json),
                "effective_length": int(arrays["effective_length"]),
                "audio_absent": audio_absent,
            }
        )
        coverage = provenance.get("text_word_coverage") or {}
        if coverage.get("first_uncovered") is not None:
            note = (
                f"text features cover {coverage['covered']}/{coverage['total']} "
                f"timeline words (first uncovered #{coverage['first_uncovered']}); "
                "those words get text_mask=0 — check the word axis against "
                "handoff/q1_expected_words.csv"
            )
            report[-1]["text_word_coverage"] = coverage
            report[-1]["issue"] = note
            print(f"  [warn] {sample_id}: {note}")
    return report
