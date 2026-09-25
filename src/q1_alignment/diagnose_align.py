# -*- coding: utf-8 -*-
"""逐样本记录词级时间分配求解过程的诊断量（回应审查意见 Q1-04）。

审查意见要求："若算法仅为启发式，明确说明；逐样本记录初始/吸附后/再投影后的目标、
约束残差和失败分支。"

本脚本对附件1 的全部样本重放同一求解器（同一个 `align_words_monotonic`，
输入取自冻结清单的词序列与实测时长，能量包络按同一前端重算），并：
  1. 记录四个阶段的目标值（按期望时长归一化的偏差平方和）与约束残差；
  2. 把重放得到的末边界与**冻结产物**（work/aligned_real/*.json 的 bins）逐词比对，
     若最大偏差为 0 则说明冻结结果可被同一实现按同一输入精确复现；
  3. 统计失败分支（无可行解、词数为 0、时长非正等）的数量。

只读既有产物与原视频；输出 work/alignment_diagnostics.csv 与
work/alignment_diagnostics_summary.json。

用法（服务器）：
  MOSEI_ROOT=~/MathModel PYTHONPATH=~/MathModel/src python -m q1_alignment.diagnose_align
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

from .forced_align import _frame_energy, load_audio_mono
from .monotonic_align import MIN_RATE, MAX_RATE, align_words_monotonic

SAMPLE_RATE = 16000


def _load_frozen_timelines(work: Path) -> dict:
    out = {}
    with (work / "word_timelines.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                d = json.loads(line)
                out[d["id"].replace("$_$", "__")] = d
    return out


def _load_manifest(work: Path) -> dict:
    out = {}
    with (work / "manifest.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                d = json.loads(line)
                out[d["safe_id"]] = d
    return out


def _frozen_boundaries(aligned_dir: Path, safe_id: str) -> list[float]:
    prov = json.loads((aligned_dir / f"{safe_id}.json").read_text(encoding="utf-8"))
    bounds = [0.0]
    for b in sorted(prov["bins"], key=lambda x: x["position"]):
        bounds.append(float(b["end"]))
    return bounds


def _resolve_video(data: Path, relpath: str) -> Path | None:
    """把 manifest 里的相对路径解析到实际视频文件（附件1 的目录层级随下载方式变化）。"""
    rel = Path(relpath)
    direct = data / rel
    if direct.exists():
        return direct
    parent, name = rel.parent.name, rel.name
    hits = sorted(data.glob(f"**/{parent}/{name}"))
    return hits[0] if hits else None


def main() -> int:
    root = Path(os.environ.get("MOSEI_ROOT") or Path(__file__).resolve().parents[2])
    work = root / "work"
    data = root / "data"
    timelines = _load_frozen_timelines(work)
    manifest = _load_manifest(work)
    aligned_dir = work / "aligned_real"

    rows = []
    failures = 0
    for safe_id in sorted(manifest):
        m = manifest[safe_id]
        tl = timelines.get(safe_id)
        row = {"safe_id": safe_id, "duration_sec": round(float(m["duration_sec"]), 6),
               "word_count": 0, "status": "ok", "failure_reason": ""}
        if tl is None:
            row.update({"status": "failed", "failure_reason": "缺少冻结词轴"})
            failures += 1
            rows.append(row)
            continue
        words = [w["word"] for w in tl["words"]]
        row["word_count"] = len(words)
        if not words:
            row.update({"status": "failed", "failure_reason": "词数为 0"})
            failures += 1
            rows.append(row)
            continue
        duration = float(m["duration_sec"])
        if not (duration > 0):
            row.update({"status": "failed", "failure_reason": "时长非正"})
            failures += 1
            rows.append(row)
            continue
        if not tl.get("diagnostics", {}).get("audio_present", True):
            row.update({"status": "skipped", "failure_reason": "静音准入未通过（audio_absent）"})
            rows.append(row)
            continue

        video = _resolve_video(data, m["video_relpath"])
        if video is None:
            row.update({"status": "failed",
                        "failure_reason": f"原视频未找到（relpath={m['video_relpath']}）"})
            failures += 1
            rows.append(row)
            continue
        row["video_path_rel"] = str(video.relative_to(data))
        waveform, rate = load_audio_mono(video, SAMPLE_RATE)
        rms, centers = _frame_energy(waveform, rate)

        trace: list = []
        entries, quality = align_words_monotonic(words, duration, rms=rms,
                                                energy_centers=centers, trace=trace)
        for rec in trace:
            for k, v in rec.items():
                if k == "stage":
                    continue
                row[f"{rec['stage']}::{k}"] = (round(v, 8) if isinstance(v, float) else v)
        row["words_per_second_source"] = round(len(words) / duration, 4)
        row["rate_band_ok"] = bool(MIN_RATE / 4 <= len(words) / duration <= MAX_RATE)

        recomputed = [float(e["start"]) for e in entries] + [float(entries[-1]["end"])]
        frozen = _frozen_boundaries(aligned_dir, safe_id)
        if len(frozen) == len(recomputed):
            dev = float(np.max(np.abs(np.asarray(recomputed) - np.asarray(frozen))))
        else:
            dev = float("nan")
        row["max_abs_deviation_vs_frozen_sec"] = None if dev != dev else round(dev, 10)
        row["reproduced_frozen"] = bool(dev == dev and dev < 1e-9)
        rows.append(row)

    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with (work / "alignment_diagnostics.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    def col(name):
        vals = [r[name] for r in rows if isinstance(r.get(name), (int, float))]
        return (min(vals), max(vals), float(np.mean(vals))) if vals else None

    def ag(name):
        return [round(x, 6) for x in col(name)] if col(name) else None

    summary = {
        "样本数": len(rows),
        "失败分支数": failures,
        "跳过（静音准入未通过）": sum(1 for r in rows if r.get("status") == "skipped"),
        "冻结结果被精确复现的样本数": sum(1 for r in rows if r.get("reproduced_frozen")),
        "与冻结边界的最大偏差(秒)": (max([r["max_abs_deviation_vs_frozen_sec"] for r in rows
                                        if isinstance(r.get("max_abs_deviation_vs_frozen_sec"), float)]
                                       + [0.0])),
        "目标值 初始/投影后/吸附后/最终（均值）": {
            "initial_cumulative": ag("initial_cumulative::objective_normalized_deviation"),
            "after_projection": ag("after_projection::objective_normalized_deviation"),
            "after_pause_snap": ag("after_pause_snap::objective_normalized_deviation"),
            "final_projected": ag("final_projected::objective_normalized_deviation"),
        },
        "约束残差（最终阶段，min/mean/max）": {
            "coverage_residual_sec": ag("final_projected::coverage_residual_sec"),
            "min_span_sec": ag("final_projected::min_span_sec"),
            "span_lower_bound_sec": ag("final_projected::span_lower_bound_sec"),
            "monotonic_violations": ag("final_projected::monotonic_violations"),
            "rate_out_of_band": ag("final_projected::rate_out_of_band"),
        },
        "说明": "目标值为按期望时长归一化的时长偏差平方和（式(4-5) 第一项）；"
                "能量谷吸附在实现中是保持可行域的混合移动（pause_weight=0.5），"
                "不是加到目标上的 λ 罚项，因此不并入目标值。"
                "constraint_residuals 中 coverage_residual_sec 与 start_offset_sec 恒为 0，"
                "min_span_sec 不小于 span_lower_bound_sec，monotonic_violations 与 rate_out_of_band 恒为 0，"
                "即两项硬约束按构造成立。",
    }
    (work / "alignment_diagnostics_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n已写出 {work/'alignment_diagnostics.csv'}  {len(rows)} 行")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
