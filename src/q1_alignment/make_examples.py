"""Generate the typical-sample alignment evidence required by section 四.2.(3).

The problem statement asks for at least one worked example showing how a text
segment, its speech time span, its video frame span and the three feature
groups line up on one timeline.

This script picks representative samples (shortest, longest, hardest, plus the
three used in the pilot run), extracts per-word timing from the generated
timeline, resolves the audio and vision frame ranges that fall inside each word
interval, and writes human-readable evidence.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .common import read_jsonl
from .forced_align import _frame_energy, load_audio_mono


def _frame_centers(times: np.ndarray) -> np.ndarray:
    times = np.asarray(times, dtype=np.float64)
    if times.ndim == 1:
        return times
    return times.mean(axis=1)


def pick_examples(
    manifest_path: str | Path,
    timelines_path: str | Path,
    limit: int = 6,
) -> list[dict[str, Any]]:
    """Choose representative samples: shortest, longest, densest, hardest."""

    manifest = read_jsonl(manifest_path)
    timelines = {row["id"]: row for row in read_jsonl(timelines_path)}

    by_duration = sorted(manifest, key=lambda r: r["duration_sec"])
    by_words = sorted(manifest, key=lambda r: -len(timelines.get(r["id"], {}).get("words", [])))
    by_frames = sorted(manifest, key=lambda r: -r["frame_count"])

    chosen: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(record: dict[str, Any], reason: str) -> None:
        if record["id"] in seen:
            return
        seen.add(record["id"])
        chosen.append({"record": record, "reason": reason})

    add(by_duration[0], "最短样本")
    add(by_duration[-1], "最长样本")
    add(by_words[0], "词数最多（触发48位合并）")
    add(by_frames[0], "视觉帧数最多")
    add(by_duration[len(by_duration) // 2], "时长中位数样本")
    add(by_words[len(by_words) // 2], "词数中位数样本")

    return chosen[:limit]


def build_example(
    record: dict[str, Any],
    timeline: dict[str, Any],
    data_root: Path,
    ffmpeg_exe: str,
    out_dir: Path,
    aligned_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Write the per-word evidence table and a summary for one sample."""

    words = timeline.get("words", [])
    video_path = data_root / record["video_relpath"]
    duration = float(record["duration_sec"])
    diagnostics = timeline.get("diagnostics") or {}
    audio_present = bool(diagnostics.get("audio_present", True))

    # audio frame grid identical to the aligner's configuration
    waveform, rate = load_audio_mono(video_path, 16000, ffmpeg_exe)
    rms, audio_centers = _frame_energy(waveform, rate)

    # vision frame grid from decoded timestamps
    from .ffprobe_utils import read_timing

    timing = read_timing(video_path, ffmpeg_exe.replace("ffmpeg", "ffprobe")
                         if ffmpeg_exe.endswith("ffmpeg") else None)
    vision_centers = np.asarray(timing.frame_times, dtype=np.float64)

    rows: list[dict[str, Any]] = []
    for index, word in enumerate(words):
        start = float(word["start"])
        end = float(word["end"])
        if not audio_present:
            # Every boundary is the -1.0 sentinel, so intersecting it with the
            # frame grids would produce empty sets that look like a real
            # "this word has no frames" finding.  Say what actually happened.
            rows.append({
                "word_index": index,
                "word": word["word"],
                "start_sec": None,
                "end_sec": None,
                "duration_sec": None,
                "audio_frame_count": None,
                "audio_frame_range": "",
                "vision_frame_count": None,
                "vision_frame_range": "",
            })
            continue
        a_idx = np.flatnonzero((audio_centers >= start) & (audio_centers < end))
        v_idx = np.flatnonzero((vision_centers >= start) & (vision_centers < end))
        rows.append({
            "word_index": index,
            "word": word["word"],
            "start_sec": round(start, 4),
            "end_sec": round(end, 4),
            "duration_sec": round(end - start, 4),
            "audio_frame_count": int(a_idx.size),
            "audio_frame_range": f"{int(a_idx[0])}-{int(a_idx[-1])}" if a_idx.size else "",
            "vision_frame_count": int(v_idx.size),
            "vision_frame_range": f"{int(v_idx[0])}-{int(v_idx[-1])}" if v_idx.size else "",
        })

    safe = record["safe_id"]
    sample_dir = out_dir / safe
    sample_dir.mkdir(parents=True, exist_ok=True)

    with (sample_dir / "word_alignment_evidence.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # alignment overview for plotting and for the paper
    _write_timeline_overview(sample_dir / "timeline_overview.csv", words)

    summary = {
        "sample_id": record["id"],
        "safe_id": safe,
        "video_relpath": record["video_relpath"],
        "real_duration_sec": round(duration, 4),
        "declared_duration_sec": round(float(record.get("declared_duration_sec", 0)), 4),
        "declared_frame_count": int(record.get("declared_frame_count", 0)),
        "real_frame_count": int(record["frame_count"]),
        "transcript": record["text"],
        "word_count": len(words),
        "audio_frame_count": int(rms.size),
        "audio_present": audio_present,
        "energy_peak": diagnostics.get("energy_peak"),
        "speech_active_ratio": diagnostics.get("speech_active_ratio"),
        "has_edit_list": record.get("has_edit_list"),
        "first_frame_pts": record.get("first_frame_pts"),
        "tail_gap_sec": record.get("tail_gap_sec"),
        "first_word_start": round(float(words[0]["start"]), 4) if words else None,
        "last_word_end": round(float(words[-1]["end"]), 4) if words else None,
        "label": float(record["label"]),
        "annotation": record["annotation"],
        "evidence_grade": (
            "measured" if audio_present else
            "audio_absent: no word time exists for this clip"
        ),
    }
    # ---- 三类特征的对应关系（题目四.2.(3) 明文要求） ---------------------
    # The problem asks the worked example to show "the text segment, its speech
    # span, its video frame span AND the correspondence of the three feature
    # groups".  The per-word frame table above only covers the first three; this
    # table adds the aligned 50-position feature arrays themselves.
    if aligned_dir is not None:
        npz_path = Path(aligned_dir) / f"{safe}.npz"
        json_path = Path(aligned_dir) / f"{safe}.json"
        if npz_path.exists():
            correspondence = _write_feature_correspondence(
                sample_dir / "feature_correspondence.csv", npz_path, json_path
            )
            summary["feature_correspondence_rows"] = correspondence["rows"]
            summary["feature_correspondence_masks"] = correspondence["masks"]

    (sample_dir / "sample_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def _write_feature_correspondence(
    path: Path, npz_path: Path, json_path: Path
) -> dict[str, Any]:
    """One row per aligned position: words, seconds, and the three feature groups.

    This is the artefact the problem statement's 四.2.(3) asks for: a reader can
    start from a text segment, read off its speech span and frame span, and see
    the exact feature row each modality contributed -- including whether that
    modality was usable there (mask) or absent (sentinel -1.0).
    """

    with np.load(npz_path, allow_pickle=False) as arrays:
        data = {name: arrays[name] for name in arrays.files}
    provenance = (
        json.loads(json_path.read_text(encoding="utf-8")) if json_path.exists() else {}
    )
    bins = {int(b["position"]): b for b in provenance.get("bins", [])}

    positions = range(1, int(data["effective_length"]) - 1)
    rows = []
    mask_totals = {"text": 0, "audio": 0, "vision": 0}
    for position in positions:
        bucket = bins.get(position, {})
        row = {
            "position": position,
            "text_segment": bucket.get("text", ""),
            "start_sec": bucket.get("start"),
            "end_sec": bucket.get("end"),
        }
        for modality in ("text", "audio", "vision"):
            flag = int(data[f"{modality}_mask"][position])
            vector = np.asarray(data[modality][position], dtype=np.float64)
            row[f"{modality}_usable"] = flag
            row[f"{modality}_dim"] = int(vector.size)
            row[f"{modality}_norm"] = round(float(np.linalg.norm(vector)), 6)
            row[f"{modality}_src_count"] = len(bucket.get(f"{modality}_feature_indices", []))
            mask_totals[modality] += flag
        rows.append(row)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    return {"rows": len(rows), "masks": mask_totals}


def _write_timeline_overview(path: Path, words: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["word_index", "word", "start_sec", "end_sec"])
        for index, word in enumerate(words):
            writer.writerow([index, word["word"], round(float(word["start"]), 4),
                             round(float(word["end"]), 4)])


def main() -> int:
    import argparse

    home = Path.home()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(home / "MathModel/work/manifest.jsonl"))
    parser.add_argument("--timelines", default=str(home / "MathModel/work/word_timelines.jsonl"))
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", default=str(home / "MathModel/work/examples"))
    parser.add_argument(
        "--aligned-dir",
        default=str(home / "MathModel/work/aligned_real"),
        help="对齐后的 50 位特征数组目录；存在时额外产出三类特征对应表（四.2.(3)）",
    )
    parser.add_argument("--ffmpeg", default=str(home / "miniconda3/envs/mosei/bin/ffmpeg"))
    parser.add_argument("--limit", type=int, default=6)
    args = parser.parse_args()

    root = Path(args.data_root).resolve()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    aligned_dir = Path(args.aligned_dir) if args.aligned_dir else None
    timelines = {row["id"]: row for row in read_jsonl(args.timelines)}

    picked = pick_examples(args.manifest, args.timelines, args.limit)
    summaries = []
    for item in picked:
        record = item["record"]
        timeline = timelines[record["id"]]
        summary = build_example(
            record, timeline, root, args.ffmpeg, out, aligned_dir=aligned_dir
        )
        summary["selection_reason"] = item["reason"]
        summaries.append(summary)
        extra = summary.get("feature_correspondence_rows")
        print(f"[{item['reason']}] {record['id']} "
              f"dur={summary['real_duration_sec']}s words={summary['word_count']}"
              + (f" 三类特征对应行={extra}" if extra else " (无对齐特征，未出对应表)"))

    (out / "examples_index.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"examples": len(summaries), "output_dir": str(out)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
