"""Complete field-level inventory of attachments 2, 3 and 4.

Produces, for every file and every field: shape, dtype, value range, a sample
value, whether it can contain padding, and whether it carries labels. This is
the authoritative reference for "what is each file, and what is it for".
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

HOME = Path.home()
DATA = HOME / "MathModel" / "data"
ATT2 = DATA / "附件2-数据集特征文件"
ATT3 = DATA / "附件3-模态缺失特征样本"
ATT4 = DATA / "附件4-可解释专项视频样本与特征文件"


def field_report(name: str, value, indent: str = "    ") -> None:
    """Print one field's full characteristics."""

    if isinstance(value, np.ndarray):
        info = f"{str(value.shape):20s} {str(value.dtype):10s}"
        if value.dtype.kind in "fc" and value.size:
            finite = np.isfinite(value)
            info += f" 范围[{np.nanmin(value):9.3f}, {np.nanmax(value):9.3f}]"
            if not finite.all():
                info += f" !!含非有限值 {int((~finite).sum())}"
            # padding signature
            if value.ndim >= 2:
                flat = value.reshape(value.shape[0], -1)
                zero_rows = np.all(flat == 0, axis=1)
                if zero_rows.any():
                    info += f" 全零行{int(zero_rows.sum())}/{len(zero_rows)}"
        elif value.dtype.kind in "US" and value.size:
            info += f" 例: {str(value.ravel()[0])[:50]}"
        elif value.dtype.kind in "iu" and value.size:
            uniq = np.unique(value)
            info += f" 唯一值{len(uniq)}"
            if len(uniq) <= 12:
                info += f" {uniq.tolist()}"
        print(f"{indent}{name:22s} {info}")
    elif isinstance(value, list):
        print(f"{indent}{name:22s} list[{len(value)}]")
        if value:
            first = value[0]
            if isinstance(first, str):
                print(f"{indent}{'':22s}   例: {first[:60]}")
            elif isinstance(first, (int, float, np.integer, np.floating)):
                arr = np.asarray(value)
                print(f"{indent}{'':22s} 范围[{arr.min()}, {arr.max()}] 均值{arr.mean():.1f}")
    elif isinstance(value, dict):
        print(f"{indent}{name:22s} dict keys={list(value)}")
        for key, sub in value.items():
            field_report(key, sub, indent + "    ")
    else:
        print(f"{indent}{name:22s} {type(value).__name__} = {str(value)[:60]}")


def scan_block(label: str, block: dict) -> None:
    print(f"\n  --- {label} ---")
    for key in sorted(block.keys()):
        field_report(key, block[key])


def main() -> None:
    # ================= 附件2 =================
    for fname in ("aligned_50.pkl", "unaligned_50.pkl"):
        path = ATT2 / fname
        print("=" * 100)
        print(f"【附件2】{fname}   ({path.stat().st_size/1e6:.1f} MB)")
        print("=" * 100)
        with path.open("rb") as handle:
            data = pickle.load(handle)
        print(f"顶层键: {list(data.keys())}")
        for split in ("train", "valid", "test"):
            block = data[split]
            n = len(block.get("text", []))
            print(f"\n{'':2s}[{split}]  {n} 条样本")
            scan_block(f"{split}", block)
        # label xlsx
        print(f"\n  配套文件 label.xlsx")
        import openpyxl
        wb = openpyxl.load_workbook(ATT2 / "label.xlsx", data_only=True, read_only=True)
        ws = wb["label"]
        rows = list(ws.iter_rows(values_only=True))
        print(f"    工作表: {wb.sheetnames}")
        print(f"    表头  : {rows[0]}")
        print(f"    行数  : {len(rows)-1}")
        for r in rows[1:3]:
            print(f"    样例  : {r}")

    # ================= 附件3 =================
    print("\n" + "=" * 100)
    print("【附件3】模态缺失特征样本")
    print("=" * 100)
    for version, pattern in (("对齐版本", "附件3_[0-9][0-9].pkl"),
                             ("未对齐版本", "附件3_未对齐版本_*.pkl")):
        files = sorted((ATT3 / version).glob(pattern))
        print(f"\n{'─'*100}")
        print(f"{version}: {len(files)} 个文件, 共 {len(files)} 条样本")
        print(f"{'─'*100}")
        if not files:
            continue
        with files[0].open("rb") as handle:
            obj = pickle.load(handle)
        print(f"  顶层键: {list(obj.keys())}")
        for top_key in obj:
            block = obj[top_key]
            print(f"\n  [{top_key}] {len(block.get('text', block.get('raw_text', [])))} 条")
            scan_block(f"{version}/{top_key}", block)

        # missing summary across all files
        print(f"\n  --- 全部 {len(files)} 个文件的缺失摘要 ---")
        print(f"  {'文件':28s} {'audio全零':>10s} {'vision全零':>11s} {'text掩码零':>11s}")
        for path in files[:6]:
            with path.open("rb") as handle:
                o = pickle.load(handle)
            b = o[list(o.keys())[0]]
            a = np.asarray(b["audio"])
            v = np.asarray(b["vision"])
            a_zero = int(np.all(a.reshape(a.shape[0], -1) == 0, axis=1).sum())
            v_zero = int(np.all(v.reshape(v.shape[0], -1) == 0, axis=1).sum())
            if "text_bert" in b:
                tb = np.asarray(b["text_bert"])
                # (1, 3, 50) at batch level, or (3, 50) per sample
                if tb.ndim == 3:
                    tb = tb[0]
                t_zero = int((tb[1] == 0).sum())
            else:
                t_zero = -1
            total = a.shape[-2] if a.ndim == 3 else a.shape[0]
            print(f"  {path.name:28s} {a_zero:5d}/{total:<4d} {v_zero:6d}/{total:<4d} {t_zero:6d}")

    # ================= 附件4 =================
    print("\n" + "=" * 100)
    print("【附件4】可解释专项视频样本与特征文件")
    print("=" * 100)
    base = None
    for cand in ATT4.rglob("*"):
        if cand.is_dir() and (cand / "对齐版本").exists():
            base = cand
            break
    print(f"根目录: {base}")
    for version in ("对齐版本", "未对齐版本"):
        d = base / version
        pkls = sorted(d.glob("*.pkl"))
        vids = sorted((d / "videos").glob("*.mp4")) if (d / "videos").exists() else []
        print(f"\n{'─'*100}")
        print(f"{version}: {len(pkls)} pkl + {len(vids)} mp4")
        print(f"{'─'*100}")
        if pkls:
            with pkls[0].open("rb") as handle:
                obj = pickle.load(handle)
            scan_block(f"{version}/{pkls[0].name}", obj)

    # ================= 字段对照 =================
    print("\n" + "=" * 100)
    print("字段对照总表")
    print("=" * 100)
    print(f"\n{'字段':24s} {'附件2 aligned':>16s} {'附件2 unaligned':>18s} {'附件3':>16s} {'附件4':>16s}")
    print("-" * 100)
    rows = [
        ("id", "list[N]", "list[N]", "无", "str"),
        ("raw_text", "(N,) str", "(N,) str", "附件3未对齐有", "(,) str"),
        ("text", "(N,50,768)", "(N,50,768)", "无(用text_bert)", "(50,768)"),
        ("text_bert", "(N,3,50)", "(N,3,50)", "✅(1,3,50)", "(3,50)"),
        ("audio", "(N,50,74)", "(N,500,74)", "✅(1,50,74)", "(50,74)"),
        ("vision", "(N,50,35)", "(N,500,35)", "✅(1,50,35)", "(50,35)"),
        ("audio_lengths", "无", "list[N]", "无", "未对齐版有"),
        ("vision_lengths", "无", "list[N]", "无", "未对齐版有"),
        ("classification_labels", "(N,)", "(N,)", "无", "无"),
        ("regression_labels", "(N,)", "(N,)", "无", "无"),
        ("annotations", "题目提及", "题目提及", "无", "无"),
    ]
    for r in rows:
        print(f"{r[0]:24s} {r[1]:>16s} {r[2]:>18s} {r[3]:>16s} {r[4]:>16s}")


if __name__ == "__main__":
    sys.exit(main())
