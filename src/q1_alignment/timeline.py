from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .common import read_jsonl, safe_id, write_jsonl


WORD_PATTERN = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z]+)?")


def tokenize_transcript(text: str) -> list[str]:
    return WORD_PATTERN.findall(text)


def make_placeholder_timelines(
    manifest_path: str | Path, output: str | Path
) -> list[dict[str, Any]]:
    """Create development-only timings covering the entire clip.

    Durations are allocated in proportion to token character counts. These timings
    deliberately carry final_usable=False and are rejected by the normal align CLI.
    """

    records: list[dict[str, Any]] = []
    for sample in read_jsonl(manifest_path):
        tokens = tokenize_transcript(sample["text"])
        if not tokens:
            raise ValueError(f"no tokens in transcript: {sample['id']}")
        duration = float(sample["duration_sec"])
        weights = [max(1, len(token.replace("'", ""))) for token in tokens]
        total = float(sum(weights))
        cursor = 0.0
        words = []
        for index, (token, weight) in enumerate(zip(tokens, weights)):
            end = duration if index == len(tokens) - 1 else cursor + duration * weight / total
            words.append(
                {
                    "word": token,
                    "start": cursor,
                    "end": end,
                    "confidence": None,
                }
            )
            cursor = end
        records.append(
            {
                "id": sample["id"],
                "method": "uniform_character_weight_placeholder",
                "final_usable": False,
                "words": words,
            }
        )
    write_jsonl(output, records)
    return records


def validate_timeline(
    timeline: dict[str, Any], duration: float, tolerance: float = 0.05
) -> list[str]:
    issues: list[str] = []
    words = timeline.get("words")
    if not isinstance(words, list) or not words:
        return ["timeline contains no words"]
    previous_end = 0.0
    for index, word in enumerate(words):
        try:
            start = float(word["start"])
            end = float(word["end"])
        except (KeyError, TypeError, ValueError):
            issues.append(f"word {index}: invalid start/end")
            continue
        if start < -tolerance or end > duration + tolerance:
            issues.append(f"word {index}: time outside clip: {start:.3f}-{end:.3f}")
        if start >= end:
            issues.append(f"word {index}: start >= end")
        if start < previous_end - 1e-9:
            issues.append(f"word {index}: overlaps or is non-monotonic")
        previous_end = max(previous_end, end)
    return issues


def import_ctm(
    ctm_path: str | Path, manifest_path: str | Path, output: str | Path
) -> list[dict[str, Any]]:
    manifest = read_jsonl(manifest_path)
    id_map: dict[str, str] = {}
    for sample in manifest:
        id_map[sample["id"]] = sample["id"]
        id_map[safe_id(sample["id"])] = sample["id"]

    grouped: dict[str, list[dict[str, Any]]] = {sample["id"]: [] for sample in manifest}
    with Path(ctm_path).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 5:
                raise ValueError(f"{ctm_path}:{line_number}: expected at least 5 columns")
            utterance, _channel, start_text, duration_text, word = fields[:5]
            if utterance not in id_map:
                raise ValueError(f"{ctm_path}:{line_number}: unknown id {utterance}")
            start = float(start_text)
            word_duration = float(duration_text)
            confidence = float(fields[5]) if len(fields) >= 6 else None
            grouped[id_map[utterance]].append(
                {
                    "word": word,
                    "start": start,
                    "end": start + word_duration,
                    "confidence": confidence,
                }
            )

    records = []
    for sample in manifest:
        words = sorted(grouped[sample["id"]], key=lambda item: (item["start"], item["end"]))
        if not words:
            raise ValueError(f"CTM contains no words for {sample['id']}")
        records.append(
            {"id": sample["id"], "method": "ctm_import", "final_usable": True, "words": words}
        )
    write_jsonl(output, records)
    return records
