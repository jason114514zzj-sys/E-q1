"""Audit every competition clip: reported vs real duration and frame count.

Writes a CSV so the result can be inspected and attached as evidence.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import cv2

HOME = Path.home()
FFPROBE = str(HOME / "miniconda3/envs/mosei/bin/ffprobe")


def probe_frame_times(path: Path) -> list[float]:
    out = subprocess.run(
        [
            FFPROBE, "-v", "error", "-select_streams", "v:0",
            "-show_entries", "frame=pts_time", "-of", "json", str(path),
        ],
        capture_output=True, text=True,
    )
    try:
        frames = json.loads(out.stdout).get("frames", [])
    except json.JSONDecodeError:
        return []
    return [float(f["pts_time"]) for f in frames if "pts_time" in f]


def probe_streams(path: Path) -> dict:
    out = subprocess.run(
        [
            FFPROBE, "-v", "error", "-show_entries",
            "format=duration:stream=codec_type,duration,nb_frames,r_frame_rate",
            "-of", "json", str(path),
        ],
        capture_output=True, text=True,
    )
    try:
        return json.loads(out.stdout)
    except json.JSONDecodeError:
        return {}


def count_decodable(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    total = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        total += 1
    cap.release()
    return total


def main() -> int:
    manifest_path = HOME / "MathModel" / "work" / "manifest.jsonl"
    records = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    root = None
    for cand in (HOME / "MathModel" / "data").rglob("label-100.xlsx"):
        if (cand.parent / records[0]["video_relpath"]).exists():
            root = cand.parent
            break
    if root is None:
        print("data root not found")
        return 1

    rows = []
    for rec in records:
        path = root / rec["video_relpath"]
        info = probe_streams(path)
        real_duration = float(info.get("format", {}).get("duration", 0.0) or 0.0)
        streams = info.get("streams", [])
        video_dur = next(
            (float(s["duration"]) for s in streams
             if s.get("codec_type") == "video" and s.get("duration")), 0.0
        )
        audio_dur = next(
            (float(s["duration"]) for s in streams
             if s.get("codec_type") == "audio" and s.get("duration")), 0.0
        )
        pts = probe_frame_times(path)
        decoded = count_decodable(path)
        declared_frames = rec["frame_count"]
        ratio = rec["duration_sec"] / real_duration if real_duration else 0.0
        rows.append({
            "id": rec["id"],
            "manifest_frames": declared_frames,
            "ffprobe_pts_frames": len(pts),
            "cv2_decoded_frames": decoded,
            "manifest_duration": round(rec["duration_sec"], 4),
            "ffprobe_format_duration": round(real_duration, 4),
            "video_stream_duration": round(video_dur, 4),
            "audio_stream_duration": round(audio_dur, 4),
            "duration_ratio": round(ratio, 4),
            "last_pts": round(pts[-1], 4) if pts else 0.0,
            "frames_match": declared_frames == decoded,
        })

    out_csv = HOME / "MathModel" / "work" / "duration_audit.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    frame_mismatch = sum(1 for r in rows if not r["frames_match"])
    dur_mismatch = sum(1 for r in rows if abs(r["duration_ratio"] - 1) > 0.02)
    ratios = [r["duration_ratio"] for r in rows]

    print("audited samples      :", total)
    print("frame count mismatch :", frame_mismatch, f"({100*frame_mismatch/total:.0f}%)")
    print("duration mismatch    :", dur_mismatch, f"({100*dur_mismatch/total:.0f}%)")
    print("duration ratio min   :", round(min(ratios), 3))
    print("duration ratio max   :", round(max(ratios), 3))
    print("duration ratio mean  :", round(sum(ratios) / len(ratios), 3))
    print()
    print("worst 5 by ratio:")
    for row in sorted(rows, key=lambda r: -r["duration_ratio"])[:5]:
        print(
            "  {id:24s} manifest={manifest_duration:7.2f}s real={ffprobe_format_duration:7.2f}s "
            "ratio={duration_ratio:5.2f} frames {manifest_frames}->{cv2_decoded_frames}".format(**row)
        )
    print()
    print("csv:", out_csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
