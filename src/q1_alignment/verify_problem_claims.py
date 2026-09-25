"""Verify the problem statement's factual claims against the rebuilt manifest."""

from __future__ import annotations

import csv
import json
from pathlib import Path

HOME = Path.home()
WORK = HOME / "MathModel" / "work"


def main() -> None:
    records = [
        json.loads(line)
        for line in (WORK / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    durations = sorted(r["duration_sec"] for r in records)
    declared = sorted(r["declared_duration_sec"] for r in records)

    print("=== 题目声明 vs 实测 ===")
    print(f"题目称时长范围: 2.648s ~ 34.567s")
    print(f"实测真实时长  : {durations[0]:.3f}s ~ {durations[-1]:.3f}s")
    print(f"元数据声称时长: {declared[0]:.3f}s ~ {declared[-1]:.3f}s")
    print()

    print("=== 题目称 37 个子文件夹 ===")
    vids = {r["video_id"] for r in records}
    print(f"实测 video_id 数: {len(vids)}")
    print()

    print("=== 标注分布（题目/修复说明） ===")
    from collections import Counter
    ann = Counter(r["annotation"] for r in records)
    print("实测:", dict(ann))
    print("修复说明称: Negative=18, Neutral=25, Positive=57")
    print()

    print("=== 极端样本 ===")
    print("最短 3 条:")
    for r in sorted(records, key=lambda x: x["duration_sec"])[:3]:
        print(f"  {r['id']:24s} real={r['duration_sec']:6.3f}s  declared={r['declared_duration_sec']:6.3f}s  words={len(r['text'].split())}")
    print("最长 3 条:")
    for r in sorted(records, key=lambda x: -x["duration_sec"])[:3]:
        print(f"  {r['id']:24s} real={r['duration_sec']:6.3f}s  declared={r['declared_duration_sec']:6.3f}s  words={len(r['text'].split())}")

    print()
    print("=== 问题: 时长是否超出题目声明上限 34.567s ===")
    over = [r for r in records if r["duration_sec"] > 34.567 + 0.01]
    print(f"真实时长超过 34.567s 的样本: {len(over)}")
    for r in over[:5]:
        print(f"  {r['id']}: {r['duration_sec']:.3f}s")

    print()
    print("=== label 取值分布 ===")
    labels = sorted(r["label"] for r in records)
    print(f"min={labels[0]:.4f} max={labels[-1]:.4f}")
    neg = sum(1 for x in labels if x < 0)
    zero = sum(1 for x in labels if x == 0)
    pos = sum(1 for x in labels if x > 0)
    print(f"负向={neg} 中性={zero} 正向={pos}")


if __name__ == "__main__":
    main()
