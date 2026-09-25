"""Determine how attachment 3 actually encodes the missing modality.

A first pass looking for all-zero time steps found nothing, so the missingness
must be encoded differently than "a run of zero vectors". This checks each
candidate mechanism explicitly.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

DATA = Path.home() / "MathModel" / "data"
ATT3 = DATA / "附件3-模态缺失特征样本"


def find_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs, start = [], None
    for index, flag in enumerate(mask):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def analyse_file(path: Path, version: str) -> None:
    with path.open("rb") as handle:
        obj = pickle.load(handle)

    block = obj["test"]
    print(f"\n--- {path.name} ---")
    for key in sorted(block.keys()):
        arr = block[key]
        if key == "raw_text":
            print(f"  {key:10s}: {arr} ")
            continue
        if not isinstance(arr, np.ndarray):
            continue
        print(f"  {key:10s}: shape={str(arr.shape):18s} dtype={str(arr.dtype):8s} "
              f"min={np.nanmin(arr):9.4f} max={np.nanmax(arr):9.4f}")

    # candidate encodings of missingness
    cands = {}
    if "text_bert" in block:
        # text_bert is (1,3,50): token ids / attention mask / segment ids?
        tb = block["text_bert"][0]
        print(f"\n  text_bert 三通道分析:")
        for c in range(tb.shape[0]):
            row = tb[c]
            print(f"    通道{c}: 范围[{row.min():.1f},{row.max():.1f}] "
                  f"唯一值数={len(np.unique(row))} 前8={row[:8].astype(int).tolist()}")
        cands["text_attention_zero"] = tb[1] == 0
        cands["text_token_zero"] = tb[0] == 0

    for key in ("audio", "vision"):
        arr = block[key][0]
        all_zero = np.all(arr == 0, axis=-1)
        nan_any = np.isnan(arr).any(axis=-1)
        cands[f"{key}_all_zero"] = all_zero
        cands[f"{key}_nan"] = nan_any
        print(f"\n  {key}: 全零时间步={int(all_zero.sum())}/{len(all_zero)}, "
              f"含NaN时间步={int(nan_any.sum())}/{len(nan_any)}")
        # per-dimension zero fraction to spot a zeroed block
        zf = (arr == 0).mean(axis=0)
        print(f"       逐维零占比: min={zf.min():.3f} max={zf.max():.3f} "
              f"（若某维整列为0则说明该模态被清空）")

    print("\n  缺失编码候选:")
    for name, mask in cands.items():
        if mask is None:
            continue
        count = int(np.asarray(mask).sum())
        if count:
            print(f"    {name}: {count} 个位置为真 -> runs={find_runs(np.asarray(mask))}")


def main() -> None:
    print("=" * 78)
    print("附件3 缺失编码方式分析")
    print("=" * 78)

    for version, pattern in (("对齐版本", "附件3_[0-9]*.pkl"),
                             ("未对齐版本", "附件3_未对齐版本_*.pkl")):
        directory = ATT3 / version
        files = sorted(directory.glob(pattern))
        if not files:
            continue
        print(f"\n{'#'*78}\n# {version}  ({len(files)} 文件)\n{'#'*78}")
        for path in files[:3]:
            analyse_file(path, version)

    # cross-file comparison: are the 30 files the same sample or different?
    print(f"\n{'#'*78}\n# 30 个文件之间的差异\n{'#'*78}")
    for version, pattern in (("对齐版本", "附件3_[0-9]*.pkl"),
                             ("未对齐版本", "附件3_未对齐版本_*.pkl")):
        files = sorted((ATT3 / version).glob(pattern))
        vals = []
        for path in files:
            with path.open("rb") as handle:
                obj = pickle.load(handle)
            block = obj["test"]
            a = block["audio"]
            v = block["vision"]
            vals.append((a.mean(), v.mean(), a.shape))
        print(f"\n  {version}:")
        print(f"    文件数 {len(files)}")
        print(f"    audio 均值 范围 [{min(x[0] for x in vals):.4f}, {max(x[0] for x in vals):.4f}]")
        print(f"    vision 均值 范围 [{min(x[1] for x in vals):.4f}, {max(x[1] for x in vals):.4f}]")
        shapes = set(x[2] for x in vals)
        print(f"    audio 形状: {shapes}")
        # are files identical?
        import hashlib
        digests = []
        for path in files:
            digests.append(hashlib.md5(path.read_bytes()).hexdigest()[:12])
        print(f"    文件内容唯一数: {len(set(digests))} / {len(files)}")


if __name__ == "__main__":
    sys.exit(main())
