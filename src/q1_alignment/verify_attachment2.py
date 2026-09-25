"""Verify the structure of attachment 2 after upload.

The problem statement documents the expected layout, so this script checks the
file against that specification rather than assuming it:

  top-level keys : train / valid / test
  access pattern : data[split][field][j]
  shapes         : aligned   text (N,50,768) audio (N,50,74) vision (N,50,35)
                   unaligned text (N,50,768) audio (N,500,74) vision (N,500,35)
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

ROOT = Path.home() / "MathModel" / "data" / "附件2-数据集特征文件"

EXPECTED = {
    "aligned_50.pkl": {"text": (50, 768), "audio": (50, 74), "vision": (50, 35)},
    "unaligned_50.pkl": {"text": (50, 768), "audio": (500, 74), "vision": (500, 35)},
}


def describe(value, depth=0, max_depth=2):
    pad = "    " * depth
    if isinstance(value, dict):
        print(f"{pad}dict, {len(value)} keys")
        if depth < max_depth:
            for key in list(value)[:12]:
                print(f"{pad}  [{key}]")
                describe(value[key], depth + 2, max_depth)
            if len(value) > 12:
                print(f"{pad}  ... 另有 {len(value)-12} 个键")
    elif isinstance(value, np.ndarray):
        print(f"{pad}ndarray {value.shape} {value.dtype}")
    elif isinstance(value, list):
        print(f"{pad}list, {len(value)} 项")
        if value and depth < max_depth:
            print(f"{pad}  首项类型: {type(value[0]).__name__}")
            describe(value[0], depth + 2, max_depth)
    else:
        text = str(value)
        print(f"{pad}{type(value).__name__}: {text[:60]}")


def main() -> int:
    for name, expected in EXPECTED.items():
        path = ROOT / name
        if not path.exists():
            print(f"[缺失] {name}")
            return 1
        print("=" * 74)
        print(f"{name}   ({path.stat().st_size/1e6:.1f} MB)")
        print("=" * 74)

        with path.open("rb") as handle:
            # load only the outer dict first to inspect keys cheaply
            data = pickle.load(handle)

        print(f"顶层类型: {type(data).__name__}")
        if not isinstance(data, dict):
            print("!! 顶层不是 dict，与题目说明不符")
            return 1
        print(f"顶层键  : {sorted(data.keys())}")
        print()

        for split in ("train", "valid", "test"):
            if split not in data:
                print(f"[!] 缺少划分: {split}")
                continue
            block = data[split]
            print(f"--- {split} ---")
            if not isinstance(block, dict):
                print(f"    !! 不是 dict，而是 {type(block).__name__}")
                continue
            print(f"    字段: {sorted(block.keys())}")
            for field in ("text", "audio", "vision"):
                if field not in block:
                    continue
                array = block[field]
                if not isinstance(array, np.ndarray):
                    print(f"    {field}: {type(array).__name__}")
                    continue
                exp = expected[field]
                mark = "OK " if tuple(array.shape[1:]) == exp else "!! "
                print(f"    [{mark}] {field:8s} {str(array.shape):18s} {array.dtype}"
                      f"   期望后两维 {exp}")
            for field in ("id", "raw_text", "annotations",
                          "classification_labels", "regression_labels",
                          "audio_lengths", "vision_lengths",
                          "text_bert", "audio", "vision"):
                if field in block:
                    value = block[field]
                    if isinstance(value, np.ndarray):
                        print(f"        {field:22s} ndarray {value.shape} {value.dtype}")
                    elif isinstance(value, list):
                        print(f"        {field:22s} list[{len(value)}]"
                              f"  例: {str(value[0])[:40] if value else ''}")
            print()

        # consistency: same index across fields
        if "train" in data and isinstance(data["train"], dict):
            block = data["train"]
            counts = {f: len(block[f]) for f in ("text", "audio", "vision") if f in block}
            print(f"train 各模态样本数: {counts}")
            if len(set(counts.values())) > 1:
                print("!! 各模态样本数不一致")
            else:
                print("OK 各模态样本数一致")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
