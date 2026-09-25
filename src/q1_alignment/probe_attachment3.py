"""Probe attachment 3 (missing-modality test set) to learn its real structure.

The problem statement says only: "若干无标签样本对应的处理后的带有随机模态缺失
的特征文件", where missing means "一个或多个模态中存在特征值全部为零的随机连续
序列区间". This script measures what that means concretely:

  * how many samples, and how the two versions (aligned / unaligned) differ
  * which modality is missing in each sample, and how long the missing run is
  * whether any labels are present
  * whether the file structure matches attachment 2's field layout
"""

from __future__ import annotations

import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

DATA = Path.home() / "MathModel" / "data"
ATT3 = DATA / "附件3-模态缺失特征样本"


def describe(obj, name="obj", depth=0, max_depth=2):
    pad = "  " * depth
    if isinstance(obj, dict):
        print(f"{pad}{name}: dict[{len(obj)}] keys={list(obj)[:12]}")
        if depth < max_depth:
            for key in list(obj)[:12]:
                describe(obj[key], key, depth + 1, max_depth)
    elif isinstance(obj, np.ndarray):
        print(f"{pad}{name}: ndarray {obj.shape} {obj.dtype}")
    elif isinstance(obj, list):
        print(f"{pad}{name}: list[{len(obj)}]")
        if obj and depth < max_depth:
            describe(obj[0], f"{name}[0]", depth + 1, max_depth)
    else:
        text = str(obj)
        print(f"{pad}{name}: {type(obj).__name__} = {text[:70]}")


def find_zero_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return (start, end) of consecutive True runs in a 1-D boolean array."""

    runs = []
    start = None
    for index, flag in enumerate(mask):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def main() -> None:
    print("=" * 78)
    print("附件3 结构探查")
    print("=" * 78)

    for version in ("对齐版本", "未对齐版本"):
        directory = ATT3 / version
        if not directory.exists():
            print(f"\n[缺失] {directory}")
            continue
        files = sorted(directory.glob("*.pkl"))
        print(f"\n{'='*78}")
        print(f"{version}  ({len(files)} 个文件)")
        print("=" * 78)

        # inspect the first file in detail
        with files[0].open("rb") as handle:
            sample = pickle.load(handle)
        print(f"\n--- {files[0].name} 结构 ---")
        describe(sample, "root")

        # now scan all files for missing patterns
        print(f"\n--- 扫描全部 {len(files)} 个文件的缺失模式 ---")
        modality_missing = Counter()
        run_stats = []
        n_samples = 0

        for path in files:
            with path.open("rb") as handle:
                obj = pickle.load(handle)
            # the object may be a dict of arrays, or a list of them
            items = [obj] if isinstance(obj, dict) else list(obj)
            for item in items:
                if not isinstance(item, dict):
                    continue
                n_samples += 1
                missing_here = []
                for modality, key in (("text", "text"), ("audio", "audio"),
                                      ("vision", "vision")):
                    arr = item.get(key)
                    if not isinstance(arr, np.ndarray) or arr.ndim < 2:
                        continue
                    # a time step is "missing" when the whole feature vector is 0
                    zero_steps = np.all(arr == 0, axis=-1)
                    if zero_steps.ndim > 1:
                        zero_steps = zero_steps.reshape(zero_steps.shape[0], -1).all(axis=1)
                    runs = find_zero_runs(zero_steps)
                    if runs:
                        missing_here.append((modality, runs, len(zero_steps)))
                if missing_here:
                    key = "+".join(sorted(m for m, _, _ in missing_here))
                    modality_missing[key] += 1
                    for modality, runs, total in missing_here:
                        longest = max(e - s for s, e in runs)
                        run_stats.append((modality, longest, total, len(runs)))

        print(f"\n  样本总数: {n_samples}")
        print(f"\n  缺失组合分布:")
        for combo, count in modality_missing.most_common():
            print(f"    {combo:24s}: {count:3d} 个")

        if run_stats:
            print(f"\n  缺失区间统计（按模态）:")
            for modality in ("text", "audio", "vision"):
                subset = [r for r in run_stats if r[0] == modality]
                if not subset:
                    continue
                longest = [r[1] for r in subset]
                ratios = [r[1] / r[2] for r in subset]
                print(f"    {modality:8s}: 涉及 {len(subset):3d} 样本, "
                      f"最长缺失 {min(longest):2d}~{max(longest):3d} 步, "
                      f"占序列 {min(ratios)*100:4.1f}%~{max(ratios)*100:4.1f}%")


if __name__ == "__main__":
    sys.exit(main())
