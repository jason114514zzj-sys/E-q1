from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .common import read_jsonl


def audit_aligned(
    manifest_path: str | Path,
    aligned_dir: str | Path,
    allow_partial: bool = False,
) -> dict[str, Any]:
    manifest = read_jsonl(manifest_path)
    root = Path(aligned_dir)
    issues: list[str] = []
    checked = 0
    for sample in manifest:
        npz_path = root / f"{sample['safe_id']}.npz"
        json_path = root / f"{sample['safe_id']}.json"
        if not npz_path.exists() or not json_path.exists():
            if not allow_partial:
                issues.append(f"missing aligned output: {sample['id']}")
            continue
        checked += 1
        with np.load(npz_path, allow_pickle=False) as arrays:
            lengths = {arrays[name].shape[0] for name in ("text", "audio", "vision")}
            if len(lengths) != 1:
                issues.append(f"{sample['id']}: modality sequence lengths differ")
            for name in ("text", "audio", "vision"):
                if not np.isfinite(arrays[name]).all():
                    issues.append(f"{sample['id']}: {name} contains NaN/Inf")
            for name in ("attention_mask", "text_mask", "audio_mask", "vision_mask"):
                if not np.isin(arrays[name], [0, 1]).all():
                    issues.append(f"{sample['id']}: {name} is not binary")
            for feature_name, mask_name in (
                ("text", "text_mask"),
                ("audio", "audio_mask"),
                ("vision", "vision_mask"),
            ):
                padded = arrays[mask_name] == 0
                if not np.all(arrays[feature_name][padded] == 0):
                    issues.append(f"{sample['id']}: {feature_name} has nonzero masked rows")
        provenance = json.loads(json_path.read_text(encoding="utf-8"))
        if provenance["id"] != sample["id"]:
            issues.append(f"{sample['id']}: provenance id mismatch")

    return {
        "manifest_count": len(manifest),
        "checked_count": checked,
        "issue_count": len(issues),
        "issues": issues,
    }

