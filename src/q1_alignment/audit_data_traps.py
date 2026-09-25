"""Statistical trap audit on attachment 2.

Checks for issues that would silently corrupt modelling:
  * duplicate sample ids within or across splits (train/test leakage)
  * identical raw_text shared by many samples
  * label distribution imbalance and annotation/label consistency
  * audio_lengths / vision_lengths plausibility
  * padding rows that are not actually zero (a classic trap)
"""

from __future__ import annotations

import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

DATA = Path.home() / "MathModel" / "data" / "附件2-数据集特征文件"


def main() -> None:
    path = DATA / "aligned_50.pkl"
    print("=" * 78)
    print("附件2 统计陷阱审计 (aligned_50.pkl)")
    print("=" * 78)
    with path.open("rb") as handle:
        data = pickle.load(handle)

    # ---- 1. duplicate ids across splits ----
    print("\n[1] 样本 ID 跨划分重复检查")
    split_ids = {s: list(data[s]["id"]) for s in ("train", "valid", "test")}
    for split, ids in split_ids.items():
        dup = [k for k, v in Counter(ids).items() if v > 1]
        print(f"  {split:6s}: {len(ids)} 条, 内部重复 {len(dup)}")
        if dup[:3]:
            print(f"          例: {dup[:3]}")
    overlaps = [
        ("train", "valid"), ("train", "test"), ("valid", "test")
    ]
    for a, b in overlaps:
        inter = set(split_ids[a]) & set(split_ids[b])
        flag = "!! 数据泄露" if inter else "OK"
        print(f"  {a} ∩ {b}: {len(inter)}  [{flag}]")
        if inter and len(inter) <= 5:
            print(f"          {sorted(inter)}")

    # ---- 2. duplicate text ----
    print("\n[2] raw_text 重复检查")
    for split in ("train", "valid", "test"):
        texts = [str(t) for t in data[split]["raw_text"]]
        counts = Counter(texts)
        dup = [(t, c) for t, c in counts.items() if c > 1]
        print(f"  {split:6s}: {len(texts)} 条, 唯一 {len(counts)}, 重复文本组 {len(dup)}")
        for text, count in sorted(dup, key=lambda x: -x[1])[:3]:
            print(f"          x{count}: {text[:70]}")

    # cross-split text leakage
    train_texts = set(str(t) for t in data["train"]["raw_text"])
    for split in ("valid", "test"):
        other = set(str(t) for t in data[split]["raw_text"])
        shared = train_texts & other
        flag = "!! 文本泄露" if shared else "OK"
        print(f"  train ∩ {split} (按文本): {len(shared)}  [{flag}]")

    # ---- 3. label statistics ----
    print("\n[3] 标签分布与一致性")
    for split in ("train", "valid", "test"):
        reg = np.asarray(data[split]["regression_labels"], dtype=np.float64).ravel()
        cls = np.asarray(data[split]["classification_labels"], dtype=np.float64).ravel()
        neg = int((reg < 0).sum())
        zero = int((reg == 0).sum())
        pos = int((reg > 0).sum())
        print(f"  {split:6s}: n={len(reg):5d} 范围[{reg.min():6.3f},{reg.max():6.3f}] "
              f"均值{reg.mean():6.3f}  负{neg:5d} 中{zero:4d} 正{pos:5d}")
        # consistency: classification label must match the sign convention
        derived = np.where(reg > 0, 2, np.where(reg < 0, 0, 1))
        mismatch = int((derived != cls).sum())
        flag = "OK" if mismatch == 0 else "!! 不一致"
        print(f"          regression/classification 一致性: 不符 {mismatch} 条  [{flag}]")
        print(f"          classification 取值: {sorted(set(cls.tolist()))}")

    # ---- 4. padding rows ----
    print("\n[4] 填充位是否为真零（经典陷阱）")
    for split in ("train", "test"):
        block = data[split]
        for field in ("audio", "vision"):
            arr = block[field]
            # unaligned not loaded here; aligned has no length field
            zeros = np.all(arr == 0, axis=2)
            all_zero_samples = int(zeros.all(axis=1).sum())
            print(f"  {split}/{field}: 全零样本 {all_zero_samples}, "
                  f"全零位置均值 {zeros.sum(axis=1).mean():.1f}/{arr.shape[1]}")

    # ---- 5. value range sanity ----
    print("\n[5] 数值范围与 NaN 检查")
    for split in ("train", "valid", "test"):
        for field in ("text", "audio", "vision"):
            arr = np.asarray(data[split][field])
            finite = np.isfinite(arr).all()
            print(f"  {split:6s}/{field:7s} shape={str(arr.shape):20s} "
                  f"dtype={str(arr.dtype):8s} "
                  f"范围[{arr.min():8.3f},{arr.max():8.3f}] "
                  f"有限={finite}")

    # ---- 6. unaligned lengths ----
    print("\n[6] unaligned 长度字段合理性")
    upath = DATA / "unaligned_50.pkl"
    if upath.exists():
        with upath.open("rb") as handle:
            udata = pickle.load(handle)
        for split in ("train", "valid", "test"):
            block = udata[split]
            a_len = np.asarray(block["audio_lengths"])
            v_len = np.asarray(block["vision_lengths"])
            print(f"  {split:6s}: audio_len 范围[{a_len.min()},{a_len.max()}] "
                  f"均值{a_len.mean():.1f} | vision_len 范围[{v_len.min()},{v_len.max()}] "
                  f"均值{v_len.mean():.1f}")
            over_a = int((a_len > 500).sum())
            over_v = int((v_len > 500).sum())
            if over_a or over_v:
                print(f"          !! 超出 500 的位置数: audio={over_a} vision={over_v}")
            # verify padding beyond declared length is zero
            arr = block["audio"]
            bad = 0
            for j in range(0, len(a_len), 200):
                n = int(a_len[j])
                if n < arr.shape[1] and not np.all(arr[j, n:, :] == 0):
                    bad += 1
            print(f"          抽样检查超长部分非零的样本数: {bad}")


if __name__ == "__main__":
    sys.exit(main())
