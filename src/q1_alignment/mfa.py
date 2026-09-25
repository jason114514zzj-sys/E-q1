from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from .common import read_jsonl, safe_id, write_jsonl
from .timeline import tokenize_transcript, validate_timeline


def prepare_mfa_corpus(
    manifest_path: str | Path,
    data_root: str | Path,
    output_dir: str | Path,
    ffmpeg_exe: str | Path,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Create same-name 16 kHz mono WAV/LAB pairs for MFA.

    The transcript is copied from the competition label sheet through manifest.jsonl.
    Audio is decoded from the original competition MP4 without modifying the source.
    """

    manifest = read_jsonl(manifest_path)
    if limit is not None:
        manifest = manifest[:limit]
    source_root = Path(data_root).resolve()
    corpus_root = Path(output_dir).resolve()
    corpus_root.mkdir(parents=True, exist_ok=True)
    ffmpeg = Path(ffmpeg_exe).resolve()
    if not ffmpeg.exists():
        raise FileNotFoundError(ffmpeg)

    report: list[dict[str, Any]] = []
    for sample in manifest:
        stem = sample["safe_id"]
        video_path = source_root / Path(sample["video_relpath"])
        wav_path = corpus_root / f"{stem}.wav"
        lab_path = corpus_root / f"{stem}.lab"
        if not video_path.exists():
            raise FileNotFoundError(video_path)

        lab_path.write_text(sample["text"].strip() + "\n", encoding="utf-8")
        command = [
            str(ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-sample_fmt",
            "s16",
            str(wav_path),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed for {sample['id']}: {completed.stderr.strip()}"
            )
        if not wav_path.exists() or wav_path.stat().st_size <= 44:
            raise RuntimeError(f"empty WAV output: {sample['id']}")
        report.append(
            {
                "id": sample["id"],
                "wav": str(wav_path),
                "lab": str(lab_path),
                "transcript_word_count": len(tokenize_transcript(sample["text"])),
            }
        )
    return report


def _word_entries(payload: dict[str, Any], path: Path) -> list[list[Any]]:
    tiers = payload.get("tiers")
    if not isinstance(tiers, dict):
        raise ValueError(f"{path}: missing MFA tiers object")
    candidates = []
    for name, tier in tiers.items():
        normalized = str(name).strip().lower()
        if normalized == "words" or normalized.endswith(" - words"):
            entries = tier.get("entries") if isinstance(tier, dict) else None
            if isinstance(entries, list):
                candidates.append(entries)
    if not candidates:
        raise ValueError(f"{path}: no words tier")
    if len(candidates) > 1:
        raise ValueError(f"{path}: multiple speaker word tiers are not supported")
    return candidates[0]


def import_mfa_json(
    input_dir: str | Path,
    manifest_path: str | Path,
    output: str | Path,
) -> list[dict[str, Any]]:
    manifest = read_jsonl(manifest_path)
    by_safe_id = {sample["safe_id"]: sample for sample in manifest}
    json_files = list(Path(input_dir).rglob("*.json"))
    file_by_stem = {path.stem: path for path in json_files}
    records: list[dict[str, Any]] = []
    silence_labels = {"", "<eps>", "<sil>", "sil", "sp"}

    for safe_name, sample in by_safe_id.items():
        if safe_name not in file_by_stem:
            raise FileNotFoundError(f"missing MFA JSON for {sample['id']}")
        path = file_by_stem[safe_name]
        payload = json.loads(path.read_text(encoding="utf-8"))
        words = []
        for entry in _word_entries(payload, path):
            if not isinstance(entry, list) or len(entry) < 3:
                raise ValueError(f"{path}: invalid word entry {entry!r}")
            start, end, label = float(entry[0]), float(entry[1]), str(entry[2]).strip()
            if label.lower() in silence_labels:
                continue
            words.append(
                {"word": label, "start": start, "end": end, "confidence": None}
            )
        record = {
            "id": sample["id"],
            "method": "montreal_forced_aligner_json",
            "final_usable": True,
            "source_file": path.name,
            "transcript_word_count": len(tokenize_transcript(sample["text"])),
            "aligned_word_count": len(words),
            "words": words,
        }
        issues = validate_timeline(record, float(sample["duration_sec"]))
        if issues:
            raise ValueError(f"invalid MFA result {sample['id']}: {'; '.join(issues)}")
        records.append(record)

    write_jsonl(output, records)
    return records

