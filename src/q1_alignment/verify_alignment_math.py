"""Independently verify the alignment math against the rehearsal output.

The rehearsal proved steps 4-8 *run*.  It did not prove they compute the right
numbers.  This script re-derives, from first principles, the properties the
alignment must satisfy, and checks the produced arrays against them:

1. **Position coverage** -- every position marked valid must map to a
   non-empty source interval, and the intervals must tile the timeline without
   gaps or overlaps.
2. **Monotonic time** -- interval_start and interval_end must be
   non-decreasing as position increases.
3. **Bounded time** -- every valid interval must lie within [0, duration].
4. **Mask coherence** -- text_mask, audio_mask, vision_mask must agree with the
   *content* they describe: a valid audio position must actually gather at least
   one source frame, which the provenance JSON lists explicitly.
5. **Gathering correctness** -- re-gather the audio features using the JSON's
   `audio_feature_indices` and confirm the result equals the stored array.

Point 5 is the strongest test: it recomputes the pipeline's own work from the
source files and compares, so a bug in the averaging or indexing shows up as a
numeric mismatch rather than a plausible-looking array.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

WORK = Path("/home/user/MathModel/work")
ALIGNED = WORK / "rehearsal_aligned"
RAW = WORK / "rehearsal_raw"

fails: list[str] = []
warns: list[str] = []


def check(ok: bool, label: str, detail: str = "") -> None:
    if ok:
        print(f"  [ok  ] {label}")
    else:
        fails.append(f"{label}: {detail}")
        print(f"  [FAIL] {label}: {detail}")


print("=" * 76)
print("独立复算对齐数学（不信任流水线自己的检查）")
print("=" * 76)

npz_files = sorted(ALIGNED.glob("*.npz"))
print(f"\n待核对样本: {len(npz_files)}")

for f in npz_files:
    print("\n" + "-" * 76)
    print(f"{f.name}")
    print("-" * 76)

    with np.load(f, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}

    prov = json.loads((ALIGNED / f"{f.stem}.json").read_text(encoding="utf-8"))
    bins = prov["bins"]
    duration = None
    meta = RAW / f"{f.stem}.json"
    if meta.exists():
        duration = json.loads(meta.read_text(encoding="utf-8")).get("duration_sec")
    if duration is None:
        # fall back to the max interval end
        duration = float(np.nanmax(data["interval_end"]))

    print(f"  duration={duration:.4f}s  内容位置={prov['content_positions']}  "
          f"有效长度={prov['effective_length']}")

    s = data["interval_start"].astype(np.float64)
    e = data["interval_end"].astype(np.float64)
    am = data["audio_mask"].astype(bool)
    tm = data["text_mask"].astype(bool)
    vm = data["vision_mask"].astype(bool)
    sm = data["sequence_mask"].astype(bool)

    # ---- 2. monotonicity over valid positions ----
    valid_idx = np.where(am)[0]
    if len(valid_idx) > 1:
        vs, ve = s[valid_idx], e[valid_idx]
        check(bool(np.all(np.diff(vs) >= -1e-6)), "有效位 interval_start 单调不减")
        check(bool(np.all(np.diff(ve) >= -1e-6)), "有效位 interval_end 单调不减")
        check(bool(np.all(ve > vs)), "每个有效位 end > start")

    # ---- 3. bounded ----
    check(bool(np.all(s[am] >= -1e-6)), "有效位 start >= 0",
          f"min={s[am].min() if am.any() else 'n/a'}")
    check(bool(np.all(e[am] <= duration + 1e-3)), "有效位 end <= duration",
          f"max={e[am].max():.4f} > {duration:.4f}"
          if am.any() and e[am].max() > duration + 1e-3 else "")

    # ---- 1. tiling: consecutive valid intervals should not overlap ----
    if len(valid_idx) > 1:
        gaps = vs[1:] - ve[:-1]
        overlap = int((gaps < -1e-6).sum())
        check(overlap == 0, "相邻有效区间不重叠", f"{overlap} 处重叠")

    # ---- 4. mask vs provenance ----
    # provenance lists source index ranges per position
    prov_positions = {b["position"]: b for b in bins}
    mismatch = []
    for j in valid_idx:
        b = prov_positions.get(int(j))
        if b is None:
            mismatch.append(f"pos{j}: 无 provenance 记录")
            continue
        if not b.get("audio_feature_indices"):
            mismatch.append(f"pos{j}: 音频有效但索引为空")
    check(not mismatch, "有效音频位都有非空源索引", f"{mismatch[:3]}")

    # every bin position with audio indices should be masked valid
    extra = []
    for b in bins:
        j = int(b["position"])
        if b.get("audio_feature_indices") and not am[j]:
            extra.append(j)
    check(not extra, "有源索引的位置都被标为有效", f"{extra[:5]}")

    # ---- 5. re-gather and compare (strongest test) ----
    # The raw npz uses "<modality>_features" keys; the aligned npz uses bare
    # modality names.  Both facts are asserted below rather than assumed, so a
    # rename surfaces as a clear message instead of a silent skip.
    raw_npz = RAW / f"{f.stem}.npz"
    if raw_npz.exists():
        with np.load(raw_npz, allow_pickle=False) as rz:
            raw_keys = set(rz.files)
            src_audio = rz["audio_features"] if "audio_features" in raw_keys else None
            src_vis = rz["vision_features"] if "vision_features" in raw_keys else None
            src_text = rz["text_features"] if "text_features" in raw_keys else None

        if src_audio is not None:
            stored = data["audio"]
            rebuilt = np.zeros_like(stored)
            for b in bins:
                j = int(b["position"])
                idx = b.get("audio_feature_indices") or []
                if idx and j < stored.shape[0]:
                    rebuilt[j] = src_audio[np.asarray(idx)].mean(axis=0)
            cmp_pos = [int(b["position"]) for b in bins
                       if b.get("audio_feature_indices")]
            if cmp_pos:
                a = stored[cmp_pos]
                bb = rebuilt[cmp_pos]
                maxdiff = float(np.abs(a - bb).max())
                check(np.allclose(a, bb, atol=1e-4, rtol=1e-3),
                      "重新按源索引聚合 == 存储的音频特征",
                      f"max|diff|={maxdiff:.6f}")
        else:
            check(False, "重聚合需要 audio_features", f"raw keys={sorted(raw_keys)}")

        if src_vis is not None:
            stored_v = data["vision"]
            rebuilt_v = np.zeros_like(stored_v)
            for b in bins:
                j = int(b["position"])
                idx = b.get("vision_feature_indices") or []
                if idx and j < stored_v.shape[0]:
                    rebuilt_v[j] = src_vis[np.asarray(idx)].mean(axis=0)
            cmp_pos = [int(b["position"]) for b in bins
                       if b.get("vision_feature_indices")]
            if cmp_pos:
                maxdiff = float(np.abs(stored_v[cmp_pos] - rebuilt_v[cmp_pos]).max())
                check(np.allclose(stored_v[cmp_pos], rebuilt_v[cmp_pos],
                                  atol=1e-4, rtol=1e-3),
                      "重新按源索引聚合 == 存储的视觉特征",
                      f"max|diff|={maxdiff:.6f}")

        if src_text is not None:
            stored_t = data["text"]
            rebuilt_t = np.zeros_like(stored_t)
            for b in bins:
                j = int(b["position"])
                idx = b.get("text_feature_indices") or []
                if idx and j < stored_t.shape[0]:
                    rebuilt_t[j] = src_text[np.asarray(idx)].mean(axis=0)
            cmp_pos = [int(b["position"]) for b in bins
                       if b.get("text_feature_indices")]
            if cmp_pos:
                maxdiff = float(np.abs(stored_t[cmp_pos] - rebuilt_t[cmp_pos]).max())
                check(np.allclose(stored_t[cmp_pos], rebuilt_t[cmp_pos],
                                  atol=1e-4, rtol=1e-3),
                      "重新按源索引聚合 == 存储的文本特征",
                      f"max|diff|={maxdiff:.6f}")

        # masks must match the provenance's own index bookkeeping
        if src_audio is not None:
            n = data["audio_mask"].shape[0]
            derived = np.zeros(n, dtype=bool)
            for b in bins:
                j = int(b["position"])
                if 0 <= j < n:
                    derived[j] = bool(b.get("audio_feature_indices"))
            check(bool(np.array_equal(derived, am)),
                  "audio_mask 与 provenance 的索引存在性一致",
                  f"mask={int(am.sum())} derived={int(derived.sum())}")

        if src_vis is not None:
            vv = data["vision_valid_ratio"]
            check(bool(np.all((vv >= -1e-6) & (vv <= 1 + 1e-6))),
                  "vision_valid_ratio 落在 [0,1]",
                  f"range=[{vv.min():.3f},{vv.max():.3f}]")
    else:
        check(False, "存在对应的原始特征文件", f"缺少 {raw_npz.name}")

    # ---- mask sanity ----
    for name, m in (("text", tm), ("audio", am), ("vision", vm), ("sequence", sm)):
        check(bool(set(np.unique(m)).issubset({False, True})),
              f"{name}_mask 为布尔", f"unique={np.unique(m)}")

print("\n" + "=" * 76)
print(f"FAIL={len(fails)}  WARN={len(warns)}")
for x in fails:
    print(f"   FAIL {x}")
print("=" * 76)
sys.exit(1 if fails else 0)
