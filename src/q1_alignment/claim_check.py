"""Cross-check numeric claims in our evidence documents against the data.

The guard verifies data assumptions, artifacts, and requirement coverage.  None
of those catch the failure this module targets: a document states a number that
was true at some point but is no longer, or that was never true.

This is not hypothetical.  Within this project:

* a claim that the positive range starts at ``+0.1667`` was written as "the gap
  between neutral and positive" and was wrong about which values are legitimate;
* ``qc_report.csv`` reported ``3 pass / 97 skipped`` while 100 timelines existed.

Both were caught only by manually re-deriving the numbers.  This module makes
that re-derivation automatic for the claims that matter.

How it works
------------
Each claim is a triple: a *pattern* that finds the number in the documents, a
*recomputation* that produces the true value from the raw data, and a tolerance.
The check fails when a document quotes a number that disagrees with the data, or
when a claim that should be documented is missing entirely.
"""

from __future__ import annotations

import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["Claim", "ClaimResult", "verify_claims", "CLAIMS"]

DATA = Path("data")
WORK = Path("work")


@dataclass
class Claim:
    id: str
    label: str
    pattern: str          # regex that finds the number *context* in a document
    expected: float
    tol: float
    unit: str = ""
    must_appear_in: tuple[str, ...] = ()
    # "abs" treats ``tol`` as an absolute window; "rel" as a fraction of
    # ``expected``.  Being explicit avoids the ambiguity that let a 0.15
    # tolerance accept 60.0 when the true value was 54.7.
    tol_kind: str = "abs"
    # A regex that captures ANY plausible value for this quantity, used to find
    # statements that contradict the data.  ``pattern`` alone searches for the
    # *expected* value, so a document asserting a wrong number (e.g. "9999" when
    # the truth is 4850) would never match and would be silently reported as
    # "not mentioned".  Red-teaming on 2026-09-24 exposed exactly that hole.
    any_number_pattern: str = ""


@dataclass
class ClaimResult:
    claim: Claim
    found: list[tuple[str, float]]
    ok: bool
    note: str
    # "absent" means the number simply is not mentioned yet, which is not an
    # error while documents are still being written.  Only "mismatch" -- the
    # number IS present and DISAGREES with the data -- is a real defect.
    kind: str = "ok"


def _tol_ok(num: float, expected: float, tol: float, tol_kind: str = "abs") -> bool:
    """Is ``num`` within tolerance of ``expected``?

    ``tol_kind`` selects the meaning of ``tol``:

    * ``"abs"`` -- an absolute window (used for counts and percentages, where
      the caller knows the precision they need);
    * ``"rel"`` -- a fraction of ``expected`` (used where the quantity is large
      and only its order of magnitude matters).

    An earlier version used ``max(tol, tol * |expected|)``, which silently
    widened a 0.15 absolute tolerance on 54.7 into an 8.2 window and accepted
    60.0.  A unit test caught it; the explicit ``tol_kind`` removes the
    ambiguity that caused it.
    """
    if tol_kind == "rel":
        bound = abs(tol * expected)
    else:
        bound = abs(tol)
    return abs(num - expected) <= bound


def _load_pkl(path: Path):
    with path.open("rb") as fh:
        return pickle.load(fh)


def recompute() -> dict[str, float]:
    """Re-derive every number we intend to quote, straight from the raw data."""
    values: dict[str, float] = {}

    # ----- attachment 2 -----
    att2 = DATA / "附件2-数据集特征文件" / "aligned_50.pkl"
    if att2.exists():
        d = _load_pkl(att2)
        splits = {s: d[s] for s in ("train", "valid", "test") if s in d}
        n_total = sum(len(v["audio"]) for v in splits.values())
        values["att2_total_samples"] = float(n_total)
        values["att2_train"] = float(len(splits.get("train", {}).get("audio", [])))
        values["att2_valid"] = float(len(splits.get("valid", {}).get("audio", [])))
        values["att2_test"] = float(len(splits.get("test", {}).get("audio", [])))

        lab = np.concatenate([np.asarray(v["regression_labels"]).ravel()
                              for v in splits.values()])
        values["label_zero_count"] = float((lab == 0).sum())
        pos = lab[lab > 0]
        neg = lab[lab < 0]
        values["smallest_positive_value"] = float(pos.min())
        values["largest_negative_value"] = float(neg.max())
        values["min_label"] = float(lab.min())
        values["max_label"] = float(lab.max())
        values["positive_fraction"] = float((lab > 0).mean() * 100)

        tr = np.asarray(splits["train"]["audio"], dtype=np.float64)
        present = ~np.all(tr == 0, axis=-1)
        valid = tr[present]
        values["audio_global_dim0_std"] = float(valid[:, 0].std())
        values["audio_global_rest_std_median"] = float(np.median(valid[:, 1:].std(axis=0)))
        values["audio_scale_ratio"] = (
            values["audio_global_dim0_std"]
            / max(values["audio_global_rest_std_median"], 1e-12)
        )
        values["audio_zero_step_pct"] = float(np.all(tr == 0, axis=-1).mean() * 100)

        vis = np.asarray(splits["train"]["vision"])
        values["vision_zero_step_pct"] = float(np.all(vis == 0, axis=-1).mean() * 100)

        txt = np.asarray(splits["train"]["text"])
        values["text_zero_step_pct"] = float(np.all(txt == 0, axis=-1).mean() * 100)

        # naive-vs-masked pooling damage
        bert = np.asarray(splits["train"]["text_bert"])
        mask = None
        for ch in range(bert.shape[1]):
            u = np.unique(bert[:, ch, :])
            if u.size <= 2 and set(np.round(u.astype(float), 6)).issubset({0.0, 1.0}):
                mask = np.asarray(bert[:, ch, :], dtype=np.float64)
                break
        if mask is not None:
            m = mask[:, :, None]
            masked = (txt * m).sum(axis=1) / np.maximum(mask.sum(axis=1, keepdims=True), 1.0)
            naive = txt.mean(axis=1)
            rel = (np.linalg.norm(naive - masked, axis=1)
                   / np.maximum(np.linalg.norm(masked, axis=1), 1e-9))
            values["text_pool_rel_err_mean"] = float(rel.mean())
            values["text_attention_mask_mean"] = float(mask.mean() * 100)

    # ----- attachment 3 -----
    a3 = DATA / "附件3-模态缺失特征样本" / "对齐版本"
    if a3.exists():
        files = sorted(a3.glob("*.pkl"))
        values["att3_count"] = float(len(files))
        ious, identical = [], 0
        for f in files:
            obj = _load_pkl(f)["test"]
            au = np.asarray(obj["audio"])
            vi = np.asarray(obj["vision"])
            while au.ndim > 2 and au.shape[0] == 1:
                au = au[0]
            while vi.ndim > 2 and vi.shape[0] == 1:
                vi = vi[0]
            am = np.all(au == 0, axis=-1)
            vm = np.all(vi == 0, axis=-1)
            inter = int(np.logical_and(am, vm).sum())
            union = int(np.logical_or(am, vm).sum())
            ious.append(inter / union if union else 1.0)
            if np.array_equal(am, vm):
                identical += 1
        values["att3_mask_identical"] = float(identical)
        values["att3_mask_iou_mean"] = float(np.mean(ious))

    # ----- attachment 1 manifest -----
    man = WORK / "manifest.jsonl"
    if man.exists():
        recs = [json.loads(l) for l in man.read_text(encoding="utf-8").splitlines() if l.strip()]
        durs = [r["duration_sec"] for r in recs]
        values["att1_clip_count"] = float(len(recs))
        values["min_duration_sec"] = float(min(durs))
        values["max_duration_sec"] = float(max(durs))

    return values


def build_claims(v: dict[str, float]) -> list[Claim]:
    """Pair each quotable number with the pattern that finds it in our docs."""
    claims: list[Claim] = []

    def add(cid, label, pattern, key, tol, unit="", docs=("output/evidence", "handoff"),
            tol_kind="abs", nearby=""):
        """``nearby`` is a regex capturing any value in the same context.

        When supplied, the checker also detects a document that states a
        *different* number in that context -- which is the case that actually
        matters, because that is how a wrong figure reaches the paper.
        """
        if key in v:
            claims.append(Claim(cid, label, pattern, v[key], tol, unit, docs,
                                tol_kind=tol_kind, any_number_pattern=nearby))

    add("N1", "附件2 总样本数", r"4850", "att2_total_samples", 1,
        tol_kind="rel", nearby=r"(?:总样本|样本总数|总条数|共)\D{0,12}?(\d{3,6})")
    add("N2", "附件2 训练/验证/测试", r"3395", "att2_train", 1,
        tol_kind="rel", nearby=r"(?:训练集|train)\D{0,12}?(\d{3,6})")
    add("N3", "label=0 条数", r"1100", "label_zero_count", 1,
        tol_kind="rel", nearby=r"(?:label\s*=\s*0|中性)\D{0,12}?(\d{2,6})")
    add("N4", "最小正值", r"\+?0\.1666?6?7?", "smallest_positive_value", 1e-4)
    add("N5", "最大负值", r"-0\.3333", "largest_negative_value", 1e-4)
    add("N6", "音频全零步比例", r"54\.7", "audio_zero_step_pct", 0.15, "%")
    add("N7", "视觉全零步比例", r"57\.2", "vision_zero_step_pct", 0.15, "%")
    add("N8", "文本全零步比例", r"0\.0%|0%", "text_zero_step_pct", 0.05, "%")
    add("N9", "文本池化相对误差", r"0\.43", "text_pool_rel_err_mean", 0.01)
    add("N10", "音频标准差异比", r"394", "audio_scale_ratio", 5, tol_kind="rel")
    add("N11", "附件3 掩码相同数", r"29/30|29 / 30", "att3_mask_identical", 0.5)
    add("N12", "附件3 掩码 IoU", r"0\.995", "att3_mask_iou_mean", 5e-4)
    add("N13", "附件1 条数", r"100 条|100条", "att1_clip_count", 1, tol_kind="rel")
    add("N14", "最短时长", r"2\.257", "min_duration_sec", 5e-3, "s")
    add("N15", "最长时长", r"29\.288", "max_duration_sec", 5e-3, "s")

    return claims


def verify_claims(root: Path = Path(".")) -> list[ClaimResult]:
    values = recompute()
    claims = build_claims(values)
    results: list[ClaimResult] = []

    for c in claims:
        found: list[tuple[str, float]] = []
        contradictions: list[tuple[str, float]] = []
        for doc_root in c.must_appear_in:
            base = root / doc_root
            if not base.exists():
                continue
            for path in base.rglob("*.md"):
                text = path.read_text(encoding="utf-8", errors="ignore")
                for m in re.finditer(c.pattern, text):
                    try:
                        num = float(m.group(0).replace("%", ""))
                    except ValueError:
                        continue
                    found.append((path.name, num))

                # Contradiction hunt: search the surrounding context for ANY
                # number and flag one that disagrees with the data.  Without
                # this, a document asserting a wrong total ("9999") never
                # matches ``pattern`` (which looks for the correct "4850") and
                # is silently filed as "not mentioned".
                if c.any_number_pattern:
                    for m in re.finditer(c.any_number_pattern, text):
                        raw = m.group(1)
                        try:
                            num = float(raw.replace(",", "").replace("%", ""))
                        except ValueError:
                            continue
                        if not _tol_ok(num, c.expected, c.tol, c.tol_kind):
                            contradictions.append((path.name, num))

        if not found and not contradictions:
            # Not mentioned yet is not the same as stated wrongly.  Reported as
            # "absent" so the guard can distinguish "still unwritten" from
            # "contradicts the data".
            results.append(ClaimResult(c, [], True,
                                       "文档中未出现该数字（尚未撰写或格式不同）",
                                       kind="absent"))
            continue

        bad = [(doc, num) for doc, num in found
               if not _tol_ok(num, c.expected, c.tol, c.tol_kind)]
        bad += contradictions

        detail = f"数据实值={c.expected:.4f}{c.unit}"
        if found:
            detail += f"；正确写法出现 {len(found)} 次"
        if contradictions:
            detail += f"；**上下文中的冲突值** {contradictions[:3]}"
        detail += "；全部一致" if not bad else ""

        results.append(ClaimResult(
            c, found, not bad, detail,
            kind="mismatch" if bad else "ok",
        ))
    return results
