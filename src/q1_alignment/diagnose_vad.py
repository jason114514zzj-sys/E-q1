"""Diagnose why the VAD back end under-covers the clip.

For each sample we compare:
  - how much of the clip the VAD considers voiced
  - how far the previous alignment actually reached
  - the energy profile, to see whether the global percentile threshold is the
    cause of the truncation
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / "MathModel" / "src"))

from q1_alignment.common import read_jsonl  # noqa: E402
from q1_alignment.forced_align import (  # noqa: E402
    _frame_energy,
    _voiced_spans,
    load_audio_mono,
)

HOME = Path.home()
FFMPEG = str(HOME / "miniconda3/envs/mosei/bin/ffmpeg")


def main() -> None:
    work = HOME / "MathModel" / "work"
    manifest = {r["id"]: r for r in read_jsonl(work / "manifest.jsonl")}
    timelines = {r["id"]: r for r in read_jsonl(work / "word_timelines.jsonl")}

    root = None
    for cand in (HOME / "MathModel" / "data").rglob("label-100.xlsx"):
        if (cand.parent / next(iter(manifest.values()))["video_relpath"]).exists():
            root = cand.parent
            break

    rows = []
    for sid, rec in manifest.items():
        tl = timelines.get(sid, {})
        words = tl.get("words", [])
        if not words:
            continue
        # Digitally silent clips: their words carry the -1.0 sentinel, so the
        # last-word coverage would be a negative number with no meaning.  They
        # are reported separately instead of being folded into the statistics.
        if not (tl.get("diagnostics") or {}).get("audio_present", True):
            print(f"  silent (no measurable speech): {sid} "
                  f"peak={float((tl.get('diagnostics') or {}).get('energy_peak', 0.0)):.2e}")
            continue
        video = root / rec["video_relpath"]
        try:
            wav, sr = load_audio_mono(video, 16000, FFMPEG)
        except Exception as exc:
            print(f"skip {sid}: {exc}")
            continue
        rms, centers = _frame_energy(wav, sr)
        spans = _voiced_spans(rms, centers, float(rec["duration_sec"]))
        voiced = sum(b - a for a, b in spans)
        dur = float(rec["duration_sec"])
        last_word = float(words[-1]["end"])
        rows.append({
            "id": sid,
            "duration": dur,
            "audio_seconds": wav.size / sr,
            "voiced_seconds": voiced,
            "voiced_ratio": voiced / dur if dur else 0,
            "last_word_end": last_word,
            "coverage": last_word / dur if dur else 0,
            "span_count": len(spans),
            "rms_p95": float(np.percentile(rms, 95)),
            "rms_median": float(np.median(rms)),
        })

    print(f"analyzed {len(rows)} samples")
    print()

    vr = [r["voiced_ratio"] for r in rows]
    cv = [r["coverage"] for r in rows]
    ar = [r["audio_seconds"] / r["duration"] for r in rows]

    print("=== 时长一致性（音频解码时长 / 声称时长）===")
    print(f"  min={min(ar):.3f} median={statistics.median(ar):.3f} max={max(ar):.3f}")
    print()

    print("=== VAD 判定有声占比 ===")
    print(f"  min={min(vr):.3f} median={statistics.median(vr):.3f} max={max(vr):.3f}")
    print()

    print("=== 对齐覆盖率 (末词结束/时长) ===")
    print(f"  min={min(cv):.3f} median={statistics.median(cv):.3f} max={max(cv):.3f}")
    print()

    # correlation: does audio duration mismatch explain the shortfall?
    print("=== 最差 10 条 ===")
    for r in sorted(rows, key=lambda x: x["coverage"])[:10]:
        print(
            "  {id:22s} dur={duration:6.2f} audio={audio_seconds:6.2f} "
            "voiced={voiced_ratio:5.2f} coverage={coverage:5.2f} spans={span_count}".format(**r)
        )

    print()
    print("=== 最好 5 条 ===")
    for r in sorted(rows, key=lambda x: -x["coverage"])[:5]:
        print(
            "  {id:22s} dur={duration:6.2f} audio={audio_seconds:6.2f} "
            "voiced={voiced_ratio:5.2f} coverage={coverage:5.2f} spans={span_count}".format(**r)
        )

    # key hypothesis test: is truncation driven by the audio track being shorter?
    short_audio = [r for r in rows if r["audio_seconds"] / r["duration"] < 0.8]
    print()
    print(f"音频时长 < 声明时长 80% 的样本: {len(short_audio)} / {len(rows)}")
    if short_audio:
        print("  这些样本的平均覆盖率: "
              f"{statistics.mean(r['coverage'] for r in short_audio):.3f}")
    ok_audio = [r for r in rows if r["audio_seconds"] / r["duration"] >= 0.8]
    if ok_audio:
        print("  音频正常样本的平均覆盖率: "
              f"{statistics.mean(r['coverage'] for r in ok_audio):.3f}")

    out = work / "vad_diagnosis.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print("wrote", out)


if __name__ == "__main__":
    main()
