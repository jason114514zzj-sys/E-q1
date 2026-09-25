"""Drivers that turn the manifest into real word-level timelines.

Kept separate from :mod:`q1_alignment.cli` so the same logic can be reused by
notebooks or batch scripts without going through argparse.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

from .common import read_jsonl, write_jsonl
from .forced_align import SILENCE_PEAK_FLOOR, align_transcript, load_ctc_bundle
from .timeline import validate_timeline


def build_word_timelines(
    manifest_path: str | Path,
    data_root: str | Path,
    output: str | Path,
    method: str = "energy",
    ffmpeg_exe: str = "ffmpeg",
    sample_rate: int = 16000,
    limit: int | None = None,
    report_path: str | Path | None = None,
    device: str = "cuda",
) -> list[dict[str, Any]]:
    """Align every manifest sample and write ``word_timelines.jsonl``.

    A failure on one sample does not abort the run: the sample is recorded with
    ``final_usable=False`` plus the reason, so nothing disappears silently.
    """

    manifest = read_jsonl(manifest_path)
    if limit is not None:
        manifest = manifest[:limit]
    root = Path(data_root).resolve()
    output_path = Path(output)

    bundle = None
    if method == "ctc":
        try:
            bundle = load_ctc_bundle(device=device)
        except Exception as exc:
            raise RuntimeError(
                f"CTC back end unavailable ({exc}); use --method energy instead"
            ) from exc

    records: list[dict[str, Any]] = []
    report: list[dict[str, Any]] = []

    for sample in manifest:
        video_path = root / Path(sample["video_relpath"])
        duration = float(sample["duration_sec"])
        started = time.perf_counter()
        entry: dict[str, Any] = {
            "id": sample["id"],
            "method": method,
            "final_usable": False,
            "words": [],
        }
        row: dict[str, Any] = {
            "id": sample["id"],
            "safe_id": sample["safe_id"],
            "method": method,
            "duration_sec": duration,
            "transcript_words": None,
            "aligned_words": 0,
            "status": "error",
            "issue": "",
            "elapsed_sec": 0.0,
            # C1: audio-presence evidence, so a digitally silent track can never
            # be mistaken for a successful alignment.
            "audio_present": None,
            "energy_peak": None,
            "speech_active_ratio": None,
            # C2/C3: container- and frame-axis evidence carried from the manifest.
            "has_edit_list": sample.get("has_edit_list"),
            "edit_list_count": sample.get("edit_list_count"),
            "first_frame_pts": sample.get("first_frame_pts"),
            "head_gap_sec": sample.get("head_gap_sec"),
            "tail_gap_sec": sample.get("tail_gap_sec"),
        }
        try:
            result = align_transcript(
                video_path,
                sample["text"],
                duration,
                method=method,
                bundle=bundle,
                sample_rate=sample_rate,
                ffmpeg_exe=ffmpeg_exe,
                device=device,
            )
            entry["words"] = result.words
            entry["final_usable"] = bool(result.final_usable)
            entry["diagnostics"] = result.diagnostics
            # ``method`` keeps the *requested* back end (the CLI argument, written
            # at row creation) because downstream readers select on it; the back
            # end that actually ran is recorded separately.  Without this the two
            # digitally silent clips looked like normal "monotonic" runs in
            # word_timelines.jsonl even though the aligner had bailed out.
            entry["alignment_method"] = result.method
            if result.warnings:
                entry["warnings"] = result.warnings

            diag = result.diagnostics or {}
            audio_present = diag.get("audio_present", True)
            row["audio_present"] = bool(audio_present)
            for key in ("energy_peak", "speech_active_ratio"):
                if key in diag:
                    row[key] = diag[key]

            if not audio_present:
                # C1: there is no speech signal to align against.  Every word
                # carries the -1.0 sentinel, so timeline validation would only
                # produce meaningless "outside clip / start >= end" complaints;
                # the sample is reported as its own status instead.
                entry["final_usable"] = False
                row["status"] = "audio_absent"
                row["issue"] = (
                    f"digital silence (peak RMS {diag.get('energy_peak', 0.0):.3e} "
                    f"<= {SILENCE_PEAK_FLOOR:g}); no word time emitted"
                )
                row["aligned_words"] = len(result.words)
                row["transcript_words"] = len(result.words)
                row["elapsed_sec"] = round(time.perf_counter() - started, 3)
                records.append(entry)
                report.append(row)
                continue

            issues = validate_timeline(
                entry, duration, tolerance=max(0.05, 0.02 * duration)
            )
            if issues:
                entry["final_usable"] = False
                row["issue"] = "; ".join(issues[:3])
            row["status"] = "ok" if entry["final_usable"] else "suspect"
            row["aligned_words"] = len(result.words)
            row["transcript_words"] = len(result.words)
            for key in ("voiced_span_count", "voiced_ratio"):
                if key in result.diagnostics:
                    row[key] = result.diagnostics[key]
        except Exception as exc:
            row["status"] = "error"
            row["issue"] = f"{type(exc).__name__}: {exc}"
            entry["error"] = row["issue"]
        row["elapsed_sec"] = round(time.perf_counter() - started, 3)
        records.append(entry)
        report.append(row)

    write_jsonl(output_path, records)
    if report_path is not None:
        write_report(report_path, report)
    return records


def write_report(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Write the per-sample diagnostics table as CSV."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
