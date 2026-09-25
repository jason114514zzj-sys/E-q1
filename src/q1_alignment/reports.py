"""CSV deliverables required by the problem-1 collaboration document.

Two tables are produced:

``alignment_trace.csv``
    One row per aligned position per sample: the time interval, the words it
    covers, the source audio/vision frame ranges and the vision validity ratio.
    This is the evidence trail that lets a reader walk any position back to the
    original footage, and it is what problem 3 will reuse later.

``qc_report.csv``
    One row per sample with pass/fail flags for the acceptance rules in the
    collaboration document (shape, time, finiteness, mask consistency).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .common import read_jsonl


def _write_csv(path: str | Path, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        output.write_text("", encoding="utf-8")
        return
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _format_range(indices: list[int] | None) -> str:
    """Collapse an index list into ``start-end`` or ``start,end,...`` form."""

    if not indices:
        return ""
    values = sorted(int(v) for v in indices)
    if len(values) == 1:
        return str(values[0])
    contiguous = all(b - a == 1 for a, b in zip(values, values[1:]))
    if contiguous:
        return f"{values[0]}-{values[-1]}"
    return ",".join(str(v) for v in values)


def build_alignment_trace(
    manifest_path: str | Path,
    aligned_dir: str | Path,
    output_csv: str | Path,
) -> int:
    """Flatten every sample's provenance JSON into one long CSV."""

    manifest = read_jsonl(manifest_path)
    root = Path(aligned_dir)
    rows: list[dict[str, Any]] = []
    written = 0

    for sample in manifest:
        json_path = root / f"{sample['safe_id']}.json"
        npz_path = root / f"{sample['safe_id']}.npz"
        if not json_path.exists():
            continue
        provenance = json.loads(json_path.read_text(encoding="utf-8"))
        vision_ratio: dict[int, float] = {}
        if npz_path.exists():
            with np.load(npz_path, allow_pickle=False) as arrays:
                valid = arrays["vision_valid"] if "vision_valid" in arrays.files else None
                ratio = arrays["vision_valid_ratio"] if "vision_valid_ratio" in arrays.files else None
                if ratio is not None:
                    vision_ratio = {i: float(v) for i, v in enumerate(ratio)}

        for bucket in provenance.get("bins", []):
            position = int(bucket["position"])
            text_words = bucket.get("text", "")
            rows.append({
                "sample_id": sample["id"],
                "safe_id": sample["safe_id"],
                "position": position,
                "word_start_index": bucket.get("word_start_index"),
                "word_end_index": bucket.get("word_end_index"),
                "words": text_words,
                "word_count": len(text_words.split()) if text_words else 0,
                "start_sec": round(float(bucket.get("start", 0.0)), 6),
                "end_sec": round(float(bucket.get("end", 0.0)), 6),
                "duration_sec": round(
                    float(bucket.get("end", 0.0)) - float(bucket.get("start", 0.0)), 6
                ),
                "audio_frame_range": _format_range(bucket.get("audio_feature_indices")),
                "audio_frame_count": len(bucket.get("audio_feature_indices") or []),
                "vision_frame_range": _format_range(bucket.get("vision_feature_indices")),
                "vision_frame_count": len(bucket.get("vision_feature_indices") or []),
                "text_feature_count": len(bucket.get("text_feature_indices") or []),
                "vision_valid_ratio": round(vision_ratio.get(position, 0.0), 4),
            })
            written += 1

    _write_csv(output_csv, rows)
    return written


def build_qc_report(
    manifest_path: str | Path,
    aligned_dir: str | Path,
    output_csv: str | Path,
    allow_partial: bool = False,
    timeline_path: str | Path | None = None,
) -> dict[str, Any]:
    """Per-sample acceptance report following the collaboration checklist.

    The report covers **two independent products**:

    1. the word-level timeline (``work/word_timelines.jsonl``, step 2/8), which
       is produced for all 100 clips without needing any teammate features; and
    2. the three-modality aligned feature arrays (``aligned_real/*.npz``,
       step 5/8), which *do* need the teammate handoff.

    Before 2026-09-24 these shared a single ``status`` column, so a run in which
    all 100 timelines existed but only 3 feature arrays had been produced
    reported "3 pass / 97 skipped".  That tally is indistinguishable from a run
    that silently dropped 97 samples, and quoting it in the paper would have
    understated our own coverage by 97%.

    Passing ``timeline_path`` makes the two products separately visible: a
    sample with a good timeline but no feature array is reported as
    ``awaiting_features`` rather than ``skipped``.

    A third distinction was added after measuring the source clips directly:
    two of the 100 video files carry a **digitally silent** audio track (peak
    RMS exactly 0).  For those clips an "aligned" timeline can only be
    fabricated, so they are reported as ``audio_absent`` with the peak RMS
    quoted, and their word times are withheld rather than invented.
    """

    manifest = read_jsonl(manifest_path)
    root = Path(aligned_dir)
    rows: list[dict[str, Any]] = []

    have_timeline: dict[str, bool] = {}
    silent_tracks: dict[str, bool] = {}
    timeline_evidence: dict[str, dict[str, Any]] = {}
    if timeline_path is not None:
        tp = Path(timeline_path)
        if tp.exists():
            for rec in read_jsonl(tp):
                ws = rec.get("words") or []
                diag = rec.get("diagnostics") or {}
                audio_present = bool(diag.get("audio_present", True))
                silent_tracks[rec["id"]] = not audio_present
                # A timeline that exists but whose every boundary is the -1.0
                # sentinel is NOT a usable timeline.  Counting it as ready was
                # the defect this column closes.
                have_timeline[rec["id"]] = bool(ws) and all(
                    float(w.get("start", -1.0)) >= 0.0 for w in ws
                )
                timeline_evidence[rec["id"]] = {
                    "audio_present": audio_present,
                    "energy_peak": diag.get("energy_peak"),
                    "speech_active_ratio": diag.get("speech_active_ratio"),
                }

    for sample in manifest:
        npz_path = root / f"{sample['safe_id']}.npz"
        json_path = root / f"{sample['safe_id']}.json"
        sid = sample["id"]
        timeline_ok = have_timeline.get(sid) if timeline_path is not None else None
        silent = silent_tracks.get(sid, False) if timeline_path is not None else None
        evidence = timeline_evidence.get(sid, {})
        row: dict[str, Any] = {
            "sample_id": sid,
            "safe_id": sample["safe_id"],
            "files_present": npz_path.exists() and json_path.exists(),
            "timeline_ok": timeline_ok,
            # ---- C1/C2/C3 evidence columns --------------------------------
            # Each of these records one measured fact about the source clip, so
            # that a later reader can re-derive why a sample was treated the way
            # it was without re-running ffprobe or the aligner.
            "audio_present": evidence.get("audio_present", True)
            if evidence
            else (None if silent is None else not silent),
            "energy_peak": evidence.get("energy_peak"),
            "speech_active_ratio": evidence.get("speech_active_ratio"),
            "has_edit_list": sample.get("has_edit_list"),
            "edit_list_count": sample.get("edit_list_count"),
            "first_frame_pts": sample.get("first_frame_pts"),
            "head_gap_sec": sample.get("head_gap_sec"),
            "tail_gap_sec": sample.get("tail_gap_sec"),
            "shape_ok": False,
            "time_monotonic": False,
            "time_within_clip": False,
            "finite_ok": False,
            "masks_binary": False,
            "masked_rows_zero": False,
            "id_ok": False,
            "issues": "",
            "status": "missing",
        }
        if silent:
            # The audio track carries no signal at all, so no word time and no
            # audio feature can exist for this clip.  That is a property of the
            # source material, verified by the peak RMS reading, not an
            # omission by us -- so it gets its own status instead of being
            # folded into "missing" (which would look like a bug) or into
            # "awaiting_features" (which would make it look recoverable).
            row["issues"] = (
                "audio track digitally silent"
                + (
                    f" (peak RMS {float(evidence['energy_peak']):.3e})"
                    if evidence.get("energy_peak") is not None
                    else ""
                )
                + "; word times withheld rather than fabricated"
            )
            row["status"] = "audio_absent"
            rows.append(row)
            continue
        if not row["files_present"]:
            if timeline_ok:
                # The word-level deliverable for this sample is complete; only
                # the feature-dependent step is outstanding.  Saying "skipped"
                # here would hide real work that was actually done.
                row["issues"] = "word timeline ready; awaiting teammate features"
                row["status"] = "awaiting_features"
            elif not allow_partial:
                row["issues"] = "aligned output missing"
                row["status"] = "missing"
            else:
                # Nothing at all was produced for this sample in a pilot run.
                row["issues"] = "not processed in this run"
                row["status"] = "skipped"
            rows.append(row)
            continue

        issues: list[str] = []
        with np.load(npz_path, allow_pickle=False) as arrays:
            names = set(arrays.files)
            lengths = {arrays[name].shape[0] for name in ("text", "audio", "vision") if name in names}
            row["shape_ok"] = len(lengths) == 1
            if not row["shape_ok"]:
                issues.append("modality lengths differ")

            mask_names = [n for n in ("sequence_mask", "text_mask", "audio_mask", "vision_mask") if n in names]
            if len(mask_names) == 4:
                row["masks_binary"] = all(
                    bool(np.isin(arrays[n], [0, 1]).all()) for n in mask_names
                )
                if not row["masks_binary"]:
                    issues.append("mask not binary")

            row["finite_ok"] = all(
                bool(np.isfinite(arrays[n]).all()) for n in ("text", "audio", "vision") if n in names
            )
            if not row["finite_ok"]:
                issues.append("NaN/Inf present")

            ok_zero = True
            for feature, mask in (("text", "text_mask"), ("audio", "audio_mask"), ("vision", "vision_mask")):
                if feature in names and mask in names:
                    padded = arrays[mask] == 0
                    if not np.all(arrays[feature][padded] == 0):
                        ok_zero = False
            row["masked_rows_zero"] = ok_zero
            if not ok_zero:
                issues.append("masked rows are nonzero")

            if "interval_start" in names and "interval_end" in names:
                starts = arrays["interval_start"]
                ends = arrays["interval_end"]
                # Padding slots carry a -1 sentinel. CLS and SEP are real
                # sequence positions but have no interval either, so select on
                # non-negative start values rather than on sequence_mask.
                real = (starts >= 0) & (ends >= 0)
                s = starts[real]
                e = ends[real]
                row["time_monotonic"] = bool(np.all(e >= s)) if s.size else False
                if not row["time_monotonic"]:
                    issues.append("interval end < start")
                duration = float(sample.get("duration_sec", 0.0))
                tol = max(0.05, 0.02 * duration)
                row["time_within_clip"] = bool(
                    np.all(s >= -tol) and np.all(e <= duration + tol)
                ) if s.size else False
                if not row["time_within_clip"]:
                    issues.append("interval outside clip duration")

        provenance = json.loads(json_path.read_text(encoding="utf-8"))
        row["id_ok"] = provenance.get("id") == sample["id"]
        if not row["id_ok"]:
            issues.append("provenance id mismatch")

        # ---- coverage and speaking-rate checks ----------------------------
        # These catch the defect that structural checks missed entirely: an
        # alignment can be monotone, finite and in-range while covering only a
        # fraction of the clip and implying an impossible speech rate.
        bins = provenance.get("bins", [])
        if bins:
            span_start = min(float(b.get("start", 0.0)) for b in bins)
            span_end = max(float(b.get("end", 0.0)) for b in bins)
            duration = float(sample.get("duration_sec", 0.0))
            coverage = (span_end - span_start) / duration if duration > 0 else 0.0
            row["coverage"] = round(coverage, 4)
            word_total = sum(len(str(b.get("text", "")).split()) for b in bins)
            row["words_per_second"] = round(word_total / duration, 4) if duration > 0 else 0.0
            if coverage < 0.9:
                issues.append(f"coverage {coverage:.1%} below 90%")
            if row["words_per_second"] > 6.0:
                issues.append(f"implausible rate {row['words_per_second']:.1f} words/s")

        row["issues"] = "; ".join(issues)
        row["status"] = "pass" if not issues else "fail"
        rows.append(row)

    _write_csv(output_csv, rows)
    passed = sum(1 for r in rows if r["status"] == "pass")
    failed = sum(1 for r in rows if r["status"] == "fail")
    missing = sum(1 for r in rows if r["status"] == "missing")
    skipped = sum(1 for r in rows if r["status"] == "skipped")
    awaiting = sum(1 for r in rows if r["status"] == "awaiting_features")
    absent = sum(1 for r in rows if r["status"] == "audio_absent")

    # A "skipped" row is only benign during a pilot run.  If the caller asked
    # for a full run (allow_partial=False) then every sample must have produced
    # output, and any missing row is a genuine gap -- not a benign skip.
    #
    # This distinction matters because the summary counters alone are
    # ambiguous: a 3-sample pilot and a run that silently dropped 97 samples
    # both report "3 pass / 97 skipped".  A reader (or the paper) cannot tell
    # which happened, so the report must say so explicitly.
    unprocessed = sorted(
        r["sample_id"] for r in rows if r["status"] in ("skipped", "missing")
    )
    incomplete = bool(unprocessed)

    # "complete" must not be read as "finished".  A run where every sample is
    # merely awaiting teammate features has complete=True but zero verified
    # samples -- reporting that as success would be exactly the kind of
    # overstatement this report exists to prevent.  So expose the two ideas
    # separately.
    verified = passed + failed
    # Samples whose audio track is digitally silent cannot yield word times or
    # audio features at all.  That is a *measured* property of the source clip,
    # not a gap in our run, so it is counted in its own bucket: excluding it
    # from ``unprocessed`` keeps ``complete`` meaningful, and reporting it
    # separately keeps it from being silently absorbed into
    # ``awaiting_features`` (which would imply the features are still coming).
    feature_step_done = verified + absent == len(rows) and not incomplete

    # The word-level timeline is a deliverable in its own right: it covers all
    # 100 clips without any teammate input.  Report its coverage separately so
    # a missing feature handoff can never be mistaken for missing alignment.
    if timeline_path is not None:
        n_timeline = sum(1 for r in rows if r.get("timeline_ok"))
        timeline_coverage = n_timeline / len(rows) if rows else 0.0
    else:
        n_timeline = None
        timeline_coverage = None

    return {
        "samples": len(rows),
        "passed": passed,
        "failed": failed,
        "missing": missing,
        "skipped": skipped,
        "awaiting_features": awaiting,
        "audio_absent": absent,
        "verified": verified,
        "complete": not incomplete,
        "feature_step_done": feature_step_done,
        "unprocessed_count": len(unprocessed),
        "unprocessed_ids": unprocessed[:20],
        "run_mode": "partial" if allow_partial else "full",
        "timeline_ready": n_timeline,
        "timeline_coverage": timeline_coverage,
        "output": str(output_csv),
    }


