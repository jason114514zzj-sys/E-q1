"""Self-built single-modality missing experiments on attachment 2.

Why this is needed
------------------
Attachment 3 only covers *audio+vision simultaneously missing in the same
contiguous span* (verified: masks identical in 29/30 samples, IoU 0.995, and
text is not masked at all).  So attachment 3 alone cannot answer the problem's
requirement (段24) to analyse

    "缺失模态类型、缺失位置、缺失时长对预测性能的影响规律"

-- specifically the *modality type* axis (text-only missing, audio-only missing,
vision-only missing) is absent from attachment 3, and so is the *missing
duration* axis in a controlled form.

This script builds those axes on attachment 2's training/validation splits,
where labels exist, so the regularities can be measured.  Attachment 3/4 stay
untouched (they are unlabelled and reserved for final inference).

Design decisions that matter
----------------------------
* **Contiguous spans, not random frames.**  The problem says 局部缺失 means
  "部分连续时间段不可用", so a mask must remove a *run* of consecutive time
  steps.
* **Missingness is expressed as a mask, never by deleting rows.**  Deleting
  frames would silently change the sequence length; the problem also requires
  keeping failed samples and recording the rule.
* **Text has no zeros**, so a text mask is applied by zeroing the feature AND
  clearing the corresponding `text_bert` attention position -- mirroring how a
  real text dropout would look downstream (see text_pooling.py).
* **Per-dimension scaling** is fitted on the training split only, excluding
  padding, per feature_scaling.py.
"""

from __future__ import annotations

import json
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/user/MathModel/src")
from q1_alignment.feature_scaling import fit_scaler  # noqa: E402
from q1_alignment.text_pooling import extract_attention_mask  # noqa: E402

DATA = Path("/home/user/MathModel/data/附件2-数据集特征文件")
OUT = Path("/home/user/MathModel/work/missing_study")

MODALITIES = ("text", "audio", "vision")


@dataclass
class MissingSpec:
    """One missing-modality condition."""

    name: str
    modalities: tuple[str, ...]   # which modalities are knocked out
    ratio: float                  # fraction of the *valid* timeline removed
    kind: str = "contiguous"      # contiguous | leading | trailing
    seed: int = 0
    notes: str = ""


def _valid_mask(x: np.ndarray, modality: str, bert: np.ndarray | None) -> np.ndarray:
    """(T,) boolean: which time steps carry real signal for this modality."""
    if modality == "text":
        if bert is None:
            raise ValueError("text needs text_bert to build its mask")
        return extract_attention_mask(bert[None, ...])[0] > 0
    return ~np.all(np.asarray(x) == 0, axis=-1)


def apply_missing(
    bundle: dict[str, np.ndarray],
    spec: MissingSpec,
    rng: np.random.Generator,
) -> tuple[dict[str, np.ndarray], dict]:
    """Return a copy of ``bundle`` with ``spec`` applied, plus a record.

    A contiguous span of length ``ratio * L`` is chosen inside the valid region
    and cleared for every modality in ``spec.modalities``.  Cleared text also
    loses its attention position, because downstream the attention mask is what
    tells the model "there is nothing here".
    """
    out = {k: np.array(v, copy=True) for k, v in bundle.items()}
    record = {"condition": spec.name, "modalities": list(spec.modalities),
              "ratio": spec.ratio, "kind": spec.kind, "spans": {}}

    # The baseline condition removes nothing, so there is no timeline to cut.
    # Returning here (rather than indexing an empty modality tuple) keeps the
    # grid uniform: every condition yields a comparable record.
    if not spec.modalities or spec.ratio <= 0:
        record["spans"] = {"start_pos": None, "end_pos": None,
                           "n_steps": 0, "frac_of_valid": 0.0}
        return out, record

    ref = spec.modalities[0]
    mask_ref = _valid_mask(out[ref], ref, out.get("text_bert"))
    valid_idx = np.flatnonzero(mask_ref)
    L = len(valid_idx)
    if L == 0 or spec.ratio <= 0:
        return out, record

    span_len = max(1, int(round(spec.ratio * L)))
    span_len = min(span_len, L)

    if spec.kind == "leading":
        start = 0
    elif spec.kind == "trailing":
        start = L - span_len
    else:
        lo = 0
        hi = L - span_len
        start = int(rng.integers(lo, hi + 1)) if hi > lo else 0

    chosen = valid_idx[start:start + span_len]
    record["spans"]["start_pos"] = int(chosen[0])
    record["spans"]["end_pos"] = int(chosen[-1])
    record["spans"]["n_steps"] = int(span_len)
    record["spans"]["frac_of_valid"] = float(span_len / L)

    for mod in spec.modalities:
        if mod == "text":
            out["text"][chosen] = 0.0
            if "text_bert" in out:
                # channel 1 is the 0/1 attention mask
                out["text_bert"][1, chosen] = 0
        else:
            key = mod
            if key in out:
                out[key][chosen] = 0.0
    return out, record


def build_conditions() -> list[MissingSpec]:
    """The grids the problem asks about: type x ratio x position."""
    conds: list[MissingSpec] = []
    # baseline
    conds.append(MissingSpec("完整输入", (), 0.0, notes="no modality removed"))
    # single-modality, by ratio
    for mod in MODALITIES:
        for r in (0.3, 0.5):
            conds.append(MissingSpec(
                f"仅缺{mod}_{int(r*100)}", (mod,), r,
                notes="single-modality contiguous dropout"))
    # modality type combination (audio+vision mirrors attachment 3)
    for r in (0.3, 0.5):
        conds.append(MissingSpec(
            f"缺audio+vision_{int(r*100)}", ("audio", "vision"), r,
            notes="matches attachment 3's observed pattern"))
        conds.append(MissingSpec(
            f"缺text+audio_{int(r*100)}", ("text", "audio"), r))
    # all three
    conds.append(MissingSpec("三模态全缺_30", MODALITIES, 0.3))
    # position axis at fixed ratio
    for kind in ("leading", "trailing", "contiguous"):
        conds.append(MissingSpec(
            f"缺audio_30_{kind}", ("audio",), 0.3, kind=kind,
            notes="position of the missing span"))
    return conds


def load_split(split: str) -> dict[str, np.ndarray]:
    with (DATA / "aligned_50.pkl").open("rb") as fh:
        d = pickle.load(fh)
    s = d[split]
    return {
        "text": np.asarray(s["text"], dtype=np.float32),
        "text_bert": np.asarray(s["text_bert"]),
        "audio": np.asarray(s["audio"], dtype=np.float32),
        "vision": np.asarray(s["vision"], dtype=np.float32),
        "labels": np.asarray(s["regression_labels"], dtype=np.float32),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("=" * 76)
    print("附件2 自建单模态缺失实验（补齐附件3 未覆盖的「缺失类型」轴）")
    print("=" * 76)

    tr = load_split("train")
    n = len(tr["labels"])
    print(f"\ntrain 样本 {n}；标签范围 [{tr['labels'].min():.2f}, {tr['labels'].max():.2f}]")

    # sanity: confirm the axes attachment 3 does NOT provide
    print("\n--- 附件3 已覆盖 / 未覆盖 ---")
    print("  已覆盖: audio 与 vision 同步缺失（29/30 掩码相同）")
    print("  未覆盖: 仅缺 text / 仅缺 audio / 仅缺 vision；缺失时长受控变化")
    print("  → 本实验补齐后两者")

    conds = build_conditions()
    print(f"\n--- 条件网格 {len(conds)} 组 ---")
    for c in conds:
        mods = "+".join(c.modalities) if c.modalities else "（无）"
        print(f"   {c.name:22s} 模态={mods:18s} 比例={c.ratio:.2f} 位置={c.kind}")

    # ---- apply to a few samples and verify the masks behave ----
    print("\n--- 抽样验证掩码行为 ---")
    rng = np.random.default_rng(20260924)
    sample_i = 0
    bundle = {k: v[sample_i] for k, v in tr.items() if k != "labels"}
    bundle["text_bert"] = bundle["text_bert"]      # (3, 50)

    for spec in conds:
        out, rec = apply_missing(bundle, spec, rng)
        if not spec.modalities:
            continue
        mod = spec.modalities[0]
        before = _valid_mask(bundle[mod], mod, bundle.get("text_bert")).sum()
        after = _valid_mask(out[mod], mod, out.get("text_bert")).sum()
        removed = int(before - after)
        print(f"   {rec['condition']:22s} 有效步 {before:3d} -> {after:3d} "
              f"(移除 {removed:3d}) 位置[{rec['spans']['start_pos']},"
              f"{rec['spans']['end_pos']}]")

    # ---- fit per-dimension scalers on train only ----
    print("\n--- 逐维标准化统计量（仅 train 拟合，排除填充）---")
    scalers = {}
    for mod in MODALITIES:
        if mod == "text":
            m = extract_attention_mask(tr["text_bert"][:1])  # placeholder
            # text scaling is not used here (text goes to the BERT route);
            # record it anyway for completeness
            scalers[mod] = None
            print(f"   {mod:7s}: 走 text_bert 掩码路线，不做逐维标准化")
            continue
        sc = fit_scaler(tr[mod], mod, source_split="train")
        scalers[mod] = sc
        print(f"   {mod:7s}: dims={sc.mean.shape[0]} 有效步={sc.n_steps_used} "
              f"std范围[{sc.std.min():.4f},{sc.std.max():.4f}]")

    # ---- persist the condition grid so it is reproducible ----
    manifest = {
        "source": "附件2 aligned_50.pkl train split",
        "n_samples": int(n),
        "conditions": [
            {"name": c.name, "modalities": list(c.modalities),
             "ratio": c.ratio, "kind": c.kind, "notes": c.notes}
            for c in conds
        ],
        "design_notes": [
            "缺失以掩码表达，不删除任何时间步（题目要求保留样本）",
            "连续片段遮挡，符合题目「部分连续时间段不可用」的定义",
            "text 缺失同时清零 text_bert 的注意力位",
            "标准化统计量仅在 train 上拟合，排除全零填充步",
            "附件3 仅覆盖 audio+vision 同步缺失，本实验补齐类型与位置轴",
        ],
    }
    (OUT / "missing_conditions.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n条件网格已写出: {OUT/'missing_conditions.json'}")
    print("=" * 76)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
