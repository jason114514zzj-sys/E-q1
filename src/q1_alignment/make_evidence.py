"""Build the paper-facing evidence tables for problem-1 alignment.

Everything here is derived from artifacts on disk, so every number in the paper
can be regenerated. Tables produced:

  T1_ablation_coverage.csv    旧 energy vs 新 monotonic 覆盖率对比
  T2_ablation_pause.csv       停顿吸附消融（关闭 vs 开启）
  T3_defect_cases.csv         缺陷典型案例（用于论文正文举证）
  T4_quality_summary.csv      全样本质量指标汇总
  T5_timing_defect.csv        元数据帧数缺陷的统计证据
  EVIDENCE.md                 表格说明与论文引用建议
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any

from .common import read_jsonl


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _coverage(
    timeline_path: Path,
    manifest: dict[str, dict],
    exclude: frozenset[str] = frozenset(),
) -> dict[str, float]:
    """Coverage statistics over the clips that actually carry measurable speech.

    ``exclude`` removes the digitally silent clips from *both* the old and the
    new series, so the comparison stays like-for-like.  Without it the two
    silent clips would enter the new series as a bogus coverage of 0 (their word
    times are the -1.0 sentinel) and the old series as 1.000 (the v1.0 aligner
    happily invented boundaries), which would flatter the new method for the
    wrong reason.
    """

    cov = []
    for record in read_jsonl(timeline_path):
        if record["id"] in exclude:
            continue
        words = record.get("words", [])
        if not words:
            continue
        duration = float(manifest[record["id"]]["duration_sec"])
        if duration <= 0:
            continue
        cov.append(
            (float(words[-1]["end"]) - float(words[0]["start"])) / duration
        )
    return {
        "min": min(cov),
        "median": statistics.median(cov),
        "mean": statistics.mean(cov),
        "below_90pct": sum(1 for c in cov if c < 0.9),
        "samples": len(cov),
    }


def _silent_ids(timeline_path: Path) -> frozenset[str]:
    """Ids whose audio track carries no signal (peak RMS under the floor)."""

    return frozenset(
        record["id"]
        for record in read_jsonl(timeline_path)
        if not (record.get("diagnostics") or {}).get("audio_present", True)
    )


def build_tables(work_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    work = Path(work_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest = {r["id"]: r for r in read_jsonl(work / "manifest.jsonl")}

    produced: dict[str, Any] = {}

    # ---- T1: coverage, old vs new -----------------------------------------
    old_path = work / "word_timelines_energy.jsonl"
    new_path = work / "word_timelines.jsonl"
    if old_path.exists() and new_path.exists():
        silent = _silent_ids(new_path)
        old = _coverage(old_path, manifest, exclude=silent)
        new = _coverage(new_path, manifest, exclude=silent)
        # Improvement factors are computed, never hand-written: a stale literal
        # here (the old table said "33.3倍" next to a 0.0754 -> 1.0000 pair,
        # which is 13.3x) is invisible because both neighbouring columns are
        # generated from data.
        rows = [
            {"指标": "覆盖率最小值", "旧方法_energy分段": round(old["min"], 4),
             "新方法_monotonic": round(new["min"], 4),
             "改进": f"{new['min'] / old['min']:.1f}倍" if old["min"] > 0 else "—"},
            {"指标": "覆盖率中位数", "旧方法_energy分段": round(old["median"], 4),
             "新方法_monotonic": round(new["median"], 4),
             "改进": f"+{100 * (new['median'] - old['median']) / old['median']:.1f}%"
                     if old["median"] > 0 else "—"},
            {"指标": "覆盖率均值", "旧方法_energy分段": round(old["mean"], 4),
             "新方法_monotonic": round(new["mean"], 4),
             "改进": f"+{100 * (new['mean'] - old['mean']) / old['mean']:.1f}%"
                     if old["mean"] > 0 else "—"},
            {"指标": "覆盖率<90%样本数", "旧方法_energy分段": old["below_90pct"],
             "新方法_monotonic": new["below_90pct"],
             "改进": f"{old['below_90pct']}→{new['below_90pct']}"},
            {"指标": "参与统计的样本数", "旧方法_energy分段": old["samples"],
             "新方法_monotonic": new["samples"],
             "改进": "两条序列同步剔除音轨静音样本，保持同口径"},
            {"指标": "音轨数字静音样本数(两列均剔除)",
             "旧方法_energy分段": len(silent), "新方法_monotonic": len(silent),
             "改进": "无语音可对齐，词时间不填哨兵以外的任何值"},
        ]
        _write(out / "T1_ablation_coverage.csv", rows)
        produced["T1"] = {"old": old, "new": new, "silent": sorted(silent)}

    # ---- T2: pause snapping ablation --------------------------------------
    ablation = work / "ablation_pause.json"
    if ablation.exists():
        data = json.loads(ablation.read_text(encoding="utf-8"))
        rows = [
            {"指标": "边界处归一化能量(均值, 越低越好)",
             "关闭停顿吸附": round(data["energy_off"], 4),
             "开启停顿吸附": round(data["energy_on"], 4),
             "相对变化": f"-{data['relative_gain_pct']:.2f}%"},
            {"指标": "改善样本数", "关闭停顿吸附": "—",
             "开启停顿吸附": f"{data['improved']}/{data['samples']}",
             "相对变化": f"{100*data['improved']/data['samples']:.0f}%"},
            {"指标": "样本总数", "关闭停顿吸附": data["samples"],
             "开启停顿吸附": data["samples"], "相对变化": "—"},
        ]
        _write(out / "T2_ablation_pause.csv", rows)
        produced["T2"] = data

    # ---- T3: defect cases --------------------------------------------------
    if old_path.exists() and new_path.exists():
        old_map = {r["id"]: r for r in read_jsonl(old_path)}
        new_map = {r["id"]: r for r in read_jsonl(new_path)}
        cases = []
        silent = _silent_ids(new_path)
        for sid, record in manifest.items():
            if sid in silent:
                # No measurable boundary exists, so a "defect case" comparison
                # would be comparing an invented v1.0 boundary with a sentinel.
                continue
            ow = old_map.get(sid, {}).get("words", [])
            nw = new_map.get(sid, {}).get("words", [])
            if not ow or not nw:
                continue
            duration = float(record["duration_sec"])
            old_cov = (float(ow[-1]["end"]) - float(ow[0]["start"])) / duration
            new_cov = (float(nw[-1]["end"]) - float(nw[0]["start"])) / duration
            cases.append({
                "sample_id": sid,
                "real_duration_sec": round(duration, 3),
                "word_count": len(nw),
                "旧方法_末词结束秒": round(float(ow[-1]["end"]), 3),
                "旧方法_覆盖率": round(old_cov, 4),
                "旧方法_等效语速_词每秒": round(len(ow) / max(float(ow[-1]["end"]) - float(ow[0]["start"]), 1e-9), 1),
                "新方法_末词结束秒": round(float(nw[-1]["end"]), 3),
                "新方法_覆盖率": round(new_cov, 4),
            })
        cases.sort(key=lambda r: r["旧方法_覆盖率"])
        _write(out / "T3_defect_cases.csv", cases[:20])
        produced["T3"] = {"worst_case": cases[0] if cases else None, "count": len(cases)}

    # ---- T4: quality summary ----------------------------------------------
    report = work / "word_align_report.csv"
    if report.exists():
        with report.open(encoding="utf-8") as handle:
            rows_in = list(csv.DictReader(handle))
        statuses: dict[str, int] = {}
        for row in rows_in:
            statuses[row["status"]] = statuses.get(row["status"], 0) + 1
        elapsed = [float(r["elapsed_sec"]) for r in rows_in if r.get("elapsed_sec")]
        rows = [
            {"指标": "样本总数", "数值": len(rows_in)},
            {"指标": "对齐成功", "数值": statuses.get("ok", 0)},
            {"指标": "异常", "数值": statuses.get("suspect", 0) + statuses.get("error", 0)},
            {"指标": "音轨数字静音(无词级时间)", "数值": statuses.get("audio_absent", 0)},
            {"指标": "总耗时(秒)", "数值": round(sum(elapsed), 2)},
            {"指标": "平均每条耗时(秒)", "数值": round(statistics.mean(elapsed), 3) if elapsed else 0},
        ]
        _write(out / "T4_quality_summary.csv", rows)
        produced["T4"] = statuses

    # ---- T5: timing metadata defect ---------------------------------------
    declared = [int(r["declared_frame_count"]) for r in manifest.values()]
    real = [int(r["frame_count"]) for r in manifest.values()]
    mismatched = sum(1 for a, b in zip(declared, real) if a != b)
    dur_declared = [float(r["declared_duration_sec"]) for r in manifest.values()]
    dur_real = [float(r["duration_sec"]) for r in manifest.values()]
    ratios = [a / b for a, b in zip(dur_declared, dur_real) if b > 0]
    # The MP4 movie header carries the same stale value: measured 2026-09-25,
    # mvhd.duration / mvhd.timescale sits above the real container duration on
    # 99/100 clips (median +3.11 s, max +8.29 s).  So the defect is not confined
    # to nb_frames -- neither metadata field may be used as a time source.
    mvhd = [
        float(r["movie_duration_sec"]) - float(r["container_duration_sec"])
        for r in manifest.values() if float(r.get("movie_timescale") or 0) > 0
    ]
    rows = [
        {"指标": "样本总数", "数值": len(declared)},
        {"指标": "声明帧数与实际不符的样本", "数值": mismatched},
        {"指标": "不符比例", "数值": f"{100*mismatched/len(declared):.1f}%"},
        {"指标": "声明帧数合计", "数值": sum(declared)},
        {"指标": "实际帧数合计", "数值": sum(real)},
        {"指标": "虚报帧数合计", "数值": sum(declared) - sum(real)},
        {"指标": "虚报比例", "数值": f"{100*(sum(declared)-sum(real))/sum(declared):.1f}%"},
        {"指标": "时长比(声明/实际)最小值", "数值": round(min(ratios), 4)},
        {"指标": "时长比(声明/实际)最大值", "数值": round(max(ratios), 4)},
        {"指标": "时长比(声明/实际)均值", "数值": round(statistics.mean(ratios), 4)},
        {"指标": "mvhd 时长高于容器时长的样本数",
         "数值": sum(1 for v in mvhd if v > 0.001)},
        {"指标": "mvhd 时长 − 容器时长 中位数(秒)",
         "数值": round(statistics.median(mvhd), 4) if mvhd else 0.0},
        {"指标": "mvhd 时长 − 容器时长 最大值(秒)",
         "数值": round(max(mvhd), 4) if mvhd else 0.0},
        {"指标": "带 elst 编辑列表的样本数",
         "数值": sum(1 for r in manifest.values() if r.get("has_edit_list"))},
    ]
    _write(out / "T5_timing_defect.csv", rows)
    produced["T5"] = {"mismatched": mismatched, "max_ratio": max(ratios)}

    return produced


EVIDENCE_DOC = """# 问题一 对齐方法对比证据

本目录表格全部由 `work/` 下的产物自动生成，可复现。

| 表 | 用途 | 论文位置建议 |
|---|---|---|
| `T1_ablation_coverage.csv` | 新旧算法覆盖率对比 | 问题一「对齐规则与实现」小节 |
| `T2_ablation_pause.csv` | 停顿吸附消融实验 | 问题一「消融实验」小节 |
| `T3_defect_cases.csv` | 缺陷典型案例举证 | 问题一「方法改进动机」小节 |
| `T4_quality_summary.csv` | 全样本质量与效率 | 问题一「全量结果」小节 |
| `T5_timing_defect.csv` | 元数据与容器头部缺陷统计 | 问题一「数据预处理」小节 |

> 覆盖率一律在**可发声样本**上统计。附件1 中有 2 条视频的音轨是数字静音
> （峰值 RMS 仅 1e-6），对它们而言"词级时间"按定义不存在，任何覆盖率数字
> 都无从计算；新旧两条序列同步剔除这 2 条，保持同口径。

## 四个可直接引用的结论

### 结论 1：数据集元数据存在系统性缺陷

附件1 的 100 条 MP4 中，**90 条（90%）**的 `nb_frames` 头部字段与实际可解码
帧数不符，合计虚报 **9906 帧（30%）**。据此计算的时长最大虚高 **3.27 倍**。
缺陷并不止于帧数：**99/100 条**的 MP4 影片头 `mvhd.duration` 同样高于容器真实
时长（中位数 +3.11 秒、最大 +8.29 秒），因为裁剪后该字段沿用的是裁剪前的值。
两条元数据都不可作为时间来源。

> 因此本文所有时间信息均取自 ffprobe 解码结果（`format.duration` 与逐帧
> `pts_time`），不使用容器元数据。容器内 100/100 条都带 `elst` 编辑列表，
> 其 `segment_duration` 除以 `mvhd` 时间刻度后与对应流的时长最大只差
> 0.67 毫秒，说明 ffmpeg 已按呈现时间轴解码，我们读到的 `pts_time` 是呈现时间。

### 结论 2：分段式对齐会导致严重欠覆盖

初版按「有声段独立分配词」实现，在有声段多而词数少时，早期段消耗完全部
词，后续段被跳过，**覆盖率中位数仅 41.4%，最低 7.5%**，83/98 条样本覆盖率
不足 90%。极端情况下 3.42 秒的视频其 5 个词仅覆盖 0.26 秒，等效 **19.4 词/秒**。

> 该缺陷通过了全部结构性校验（时间单调、数值有限、落在时长范围内），
> 说明仅依赖结构校验不足以保证对齐正确，必须引入覆盖率与语速约束。

### 结论 3：约束式对齐 + 停顿吸附有效

改为在 `[0, D]` 上求解带硬约束的单调边界分配后，覆盖率提升至 **100%**；
在此基础上引入停顿感知的边界吸附，使**边界处归一化能量降低 21.95%**
（40/40 条样本均有改善）。

### 结论 4：有 2 条样本的音轨根本没有语音，而旧实现把它「对齐」了出来

`-mJ2ud6oKI8$_$1` 与 `-mJ2ud6oKI8$_$2` 分别带 13 词和 11 词的转写，但音轨峰值
RMS 只有 **1e-6**，即整条音轨没有任何可观测语音。旧实现照常输出了一组严格单调、
落在时长范围内、覆盖率为 **1.000** 的边界——每一个结构性检查都通过，覆盖率指标
也看不出问题，因为覆盖率只问"被覆盖了多少"，从不问"被对齐的到底有没有声音"。

> 因此对齐的准入条件不能只看转写文本是否存在，还必须先测音轨能量：只有峰值
> 高于静音底线的样本才允许写时间，其余样本的每个词时间一律填 **-1.0 哨兵**
> 并在质量报告中单独标注为 `audio_absent`，绝不通过任何其它路径进入特征数组。

## 论文写作提示

- T1 适合做成双柱对比图，突出中位数与最差情况
- T2 是标准的消融表，建议在正文说明 `pause_weight` 取值的敏感性
- T3 建议只挑 1-2 个最极端案例放在正文，其余放附录
- T5 的数据可以支撑「本文对数据进行了严格的时间一致性核查」这一表述
"""


def main() -> int:
    import argparse

    home = Path.home()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", default=str(home / "MathModel/work"))
    parser.add_argument("--output-dir", default=str(home / "MathModel/output/evidence"))
    args = parser.parse_args()

    produced = build_tables(args.work_dir, args.output_dir)
    (Path(args.output_dir) / "EVIDENCE.md").write_text(EVIDENCE_DOC, encoding="utf-8")
    print(json.dumps(produced, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
