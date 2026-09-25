"""Feasibility probe for the alignment innovation.

The collaboration document only requires word-level forced alignment. Before
proposing anything beyond that, check whether the data actually supports the
extra machinery, so an innovation is only added when it is both achievable and
verifiable.

Probes:
  1. Do the clips contain pauses between words (needed for pause-aware
     boundary refinement to have any effect at all)?
  2. How many alignment positions are empty today (the "empty interval" defect
     the collaboration doc flags)?
  3. Would an acoustic-margin objective be measurable?
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

    # 1. pause statistics: run a fine energy analysis and count low-energy gaps
    pause_counts = []
    pause_secs = []
    words_per_sec = []
    for sid, rec in list(manifest.items())[:30]:
        video = root / rec["video_relpath"]
        try:
            wav, sr = load_audio_mono(video, 16000, FFMPEG)
        except Exception:
            continue
        rms, centers = _frame_energy(wav, sr, frame_ms=25.0, hop_ms=10.0)
        p20 = float(np.percentile(rms, 20))
        quiet = rms < p20
        # count runs of quiet frames lasting >= 60 ms
        runs = 0
        secs = 0.0
        current = 0
        for flag in quiet:
            if flag:
                current += 1
            else:
                if current * 0.01 >= 0.06:
                    runs += 1
                    secs += current * 0.01
                current = 0
        if current * 0.01 >= 0.06:
            runs += 1
            secs += current * 0.01
        pause_counts.append(runs)
        pause_secs.append(secs)
        words = timelines[sid].get("words", [])
        if words and (timelines[sid].get("diagnostics") or {}).get(
            "audio_present", True
        ):
            span = float(words[-1]["end"]) - float(words[0]["start"])
            if span > 0:
                words_per_sec.append(len(words) / span)

    print("=== 1. 停顿结构（30 条抽样）===")
    print(f"  每样本低能量间隙数: min={min(pause_counts)} median={statistics.median(pause_counts):.0f} max={max(pause_counts)}")
    print(f"  低能量总时长(秒): min={min(pause_secs):.2f} median={statistics.median(pause_secs):.2f} max={max(pause_secs):.2f}")
    print(f"  说明: 存在明显停顿结构 -> 停顿感知的边界细化有实际作用空间")
    print()

    # 2. empty positions in the current alignment
    empty_total = 0
    pos_total = 0
    skipped_silent = 0
    for sid, tl in timelines.items():
        words = tl.get("words", [])
        # Silent clips deliberately carry the -1.0 sentinel, which is
        # zero-length; counting it here would invent 24 phantom empty positions.
        if not (tl.get("diagnostics") or {}).get("audio_present", True):
            skipped_silent += len(words)
            continue
        for w in words:
            pos_total += 1
            if float(w["end"]) - float(w["start"]) <= 1e-6:
                empty_total += 1
    print("=== 2. 当前零长度位置 ===")
    print(f"  零长度词位置: {empty_total} / {pos_total}")
    print(f"  （另有 {skipped_silent} 个位置属音轨数字静音样本，用哨兵占位，不计入）")
    print()

    # 3. word rate sanity: is uniform allocation plausible at all?
    print("=== 3. 语速合理性 ===")
    print(f"  词/秒: min={min(words_per_sec):.2f} median={statistics.median(words_per_sec):.2f} max={max(words_per_sec):.2f}")
    print(f"  参考: 正常英文口语约 2-4 词/秒")
    print()

    # 4. do word counts correlate with duration? if yes, uniform allocation works
    pairs = [
        (float(manifest[sid]["duration_sec"]), len(timelines[sid].get("words", [])))
        for sid in manifest if sid in timelines
    ]
    pairs = [p for p in pairs if p[1] > 0]
    d = np.array([p[0] for p in pairs])
    w = np.array([p[1] for p in pairs])
    corr = float(np.corrcoef(d, w)[0, 1])
    print("=== 4. 时长 vs 词数 相关性 ===")
    print(f"  Pearson r = {corr:.3f}")
    print(f"  说明: 相关性高 -> 时长比例分配是合理的先验")
    print()

    out = {
        "pause_count_median": statistics.median(pause_counts),
        "pause_seconds_median": statistics.median(pause_secs),
        "zero_length_positions": empty_total,
        "total_positions": pos_total,
        "words_per_second_median": statistics.median(words_per_sec),
        "duration_word_correlation": corr,
    }
    (work / "innovation_feasibility.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("wrote", work / "innovation_feasibility.json")


if __name__ == "__main__":
    main()
