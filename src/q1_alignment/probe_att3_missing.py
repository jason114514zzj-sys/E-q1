"""Determine, per file, exactly which modality is zeroed in attachment 3.

The batch-level shape (1, 50, 74) hid the zero rows: collapsing with
`reshape(shape[0], -1)` keeps the batch axis and therefore never sees an
all-zero time step. This redoes the analysis per sample (dropping the batch
axis) and reports, for all 30 files, which modalities are missing and for how
long.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

HOME = Path.home()
ATT3 = HOME / "MathModel" / "data" / "附件3-模态缺失特征样本"


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    out, start = [], None
    for i, f in enumerate(mask):
        if f and start is None:
            start = i
        elif not f and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(mask)))
    return out


def analyse(path: Path) -> dict:
    with path.open("rb") as handle:
        obj = pickle.load(handle)
    block = obj["test"]

    result = {"file": path.name}

    for key in ("audio", "vision"):
        arr = np.asarray(block[key])
        if arr.ndim == 3:
            arr = arr[0]                      # drop batch axis
        zero = np.all(arr == 0, axis=-1)
        result[key] = {
            "zeros": int(zero.sum()),
            "total": int(len(zero)),
            "runs": runs(zero),
        }

    if "text_bert" in block:
        tb = np.asarray(block["text_bert"])
        if tb.ndim == 3:
            tb = tb[0]
        attn = tb[1]                          # channel 1 = attention mask
        zero = attn == 0
        result["text"] = {
            "zeros": int(zero.sum()),
            "total": int(len(zero)),
            "runs": runs(zero),
        }
    else:
        result["text"] = None

    return result


def main() -> None:
    for version, pattern in (("对齐版本", "附件3_[0-9][0-9].pkl"),
                             ("未对齐版本", "附件3_未对齐版本_*.pkl")):
        files = sorted((ATT3 / version).glob(pattern))
        if not files:
            continue
        print("=" * 100)
        print(f"{version}   ({len(files)} 文件)")
        print("=" * 100)
        print(f"{'文件':30s} {'audio缺失':>14s} {'vision缺失':>14s} {'text缺失':>12s}")
        print("-" * 100)

        summary = {"audio": 0, "vision": 0, "text": 0}
        for path in files:
            r = analyse(path)
            cells = []
            for key in ("audio", "vision", "text"):
                info = r[key]
                if info is None:
                    cells.append(f"{'n/a':>14s}")
                    continue
                if info["zeros"]:
                    summary[key] += 1
                    longest = max((e - s for s, e in info["runs"]), default=0)
                    cells.append(f"{info['zeros']:4d}/{info['total']:<4d} 最长{longest:2d}")
                else:
                    cells.append(f"{'无':>14s}")
            print(f"{path.name:30s} {cells[0]:>16s} {cells[1]:>16s} {cells[2]:>14s}")

        total = len(files)
        print("-" * 100)
        print(f"缺失统计: audio {summary['audio']}/{total}, "
              f"vision {summary['vision']}/{total}, text {summary['text']}/{total}")

        # which combination dominates
        combos = {}
        for path in files:
            r = analyse(path)
            key = tuple(k for k in ("text", "audio", "vision")
                        if r[k] and r[k]["zeros"] > 0)
            combos[key] = combos.get(key, 0) + 1
        print(f"\n缺失组合分布:")
        for combo, count in sorted(combos.items(), key=lambda x: -x[1]):
            label = "+".join(combo) if combo else "（无缺失）"
            print(f"  {label:30s}: {count} 个文件")
        print()


if __name__ == "__main__":
    sys.exit(main())
