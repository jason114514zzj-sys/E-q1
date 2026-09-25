"""Adapt the collaboration-document handoff into this project's internal format.

The collaboration technical document defines what the feature-extraction
teammate delivers::

    q1_feature_handoff/
    ├─ manifest.csv
    ├─ config.yaml
    ├─ versions.txt
    ├─ run_log.csv
    └─ samples/<sample_id>/
        ├─ metadata.json
        ├─ text_words.csv
        ├─ text_features.npy
        ├─ audio_features.npy
        ├─ audio_intervals.npy
        ├─ vision_features.npy
        ├─ vision_timestamps.npy
        ├─ vision_valid.npy
        └─ vision_confidence.npy

The aligner reads one ``<safe_id>.npz`` per sample with different key names.
This module bridges the two, and — importantly — *validates* rather than
silently coerces, because a shape or time-axis mismatch here would corrupt
every downstream result.

Field mapping
-------------
=====================  =====================
document delivers       internal key
=====================  =====================
text_features.npy       text_features
audio_features.npy      audio_features
audio_intervals.npy     audio_times      (renamed)
vision_features.npy     vision_features
vision_timestamps.npy   vision_times     (renamed)
vision_valid.npy        vision_valid
vision_confidence.npy   vision_confidence (extra, preserved)
text_words.csv          text_words       (extra, preserved)
metadata.json           _metadata        (extra, preserved)
=====================  =====================

Array values are stored as float32 for features and uint8 for masks, matching
the internal contract.
"""

from __future__ import annotations

import csv
import json
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .common import read_jsonl, safe_id
from .timeline import tokenize_transcript
from .word_axis import build_mapping

# document file name -> internal array key
FILE_TO_KEY = {
    "text_features.npy": "text_features",
    "audio_features.npy": "audio_features",
    "audio_intervals.npy": "audio_times",
    "vision_features.npy": "vision_features",
    "vision_timestamps.npy": "vision_times",
    "vision_valid.npy": "vision_valid",
    "vision_confidence.npy": "vision_confidence",
    "audio_valid.npy": "audio_valid",
    "text_valid.npy": "text_valid",
}

# Required and optional are kept in lockstep with handoff_selfcheck.REQUIRED_FILES.
# They diverged once (on 2026-09-24 the self-check demanded text_words.csv and
# vision_valid.npy while this adapter called them optional), which meant a
# teammate following the frozen contract could be rejected by the self-check for
# omitting files the contract itself listed as "recommended".  Both lists are now
# asserted equal by test_interface_consistency.
REQUIRED_FILES = (
    "text_features.npy",
    "audio_features.npy",
    "audio_intervals.npy",
    "vision_features.npy",
    "vision_timestamps.npy",
    "text_words.csv",
    "vision_valid.npy",
)

OPTIONAL_FILES = (
    "vision_confidence.npy",
    "metadata.json",
    # Introduced by the per-modality time-mask requirement (contract section 10).
    # They let problem 2 distinguish "this modality is missing here" from
    # "this modality happens to read zero", which attachment 3 exercises.
    "audio_valid.npy",
    "text_valid.npy",
)


@dataclass
class SampleIssues:
    """Per-sample validation outcome."""

    sample_id: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    #: 交付 token 行 → 规范词索引 的逐行映射（仅当分词不同但已通过对账时设置）
    word_axis_mapping: list[int] | None = None

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass
class HandoffReport:
    """Summary of a handoff-directory validation or import run."""

    sample_count: int = 0
    imported: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    per_sample: list[SampleIssues] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors and self.failed == 0


_APOSTROPHES = "\u2019\u2018\u02bc\u2032\u00b4\u0060"


def _norm_word(word: str) -> str:
    """Compare-friendly form of a word.

    Typography must not read as a mapping error, so curly/backtick apostrophes
    are folded onto the straight one and case is removed.  Note that NFKC alone
    is **not** enough here: U+2019 RIGHT SINGLE QUOTATION MARK has no
    compatibility decomposition, so it survives NFKC unchanged (verified
    2026-09-25 -- the first version of this function relied on NFKC and a test
    caught it).
    """

    folded = unicodedata.normalize("NFKC", word)
    for apostrophe in _APOSTROPHES:
        folded = folded.replace(apostrophe, "'")
    return folded.strip().casefold()


def read_text_words(path: Path) -> list[str]:
    """Return the ``word`` column of ``text_words.csv`` (empty if unusable)."""

    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [str(r.get("word", "")).strip() for r in rows]


def check_word_axis(
    sample_dir: Path,
    expected_words: "Sequence[str]",
    issues: SampleIssues,
    text_rows: int | None = None,
) -> None:
    """Verify the delivered word axis against OUR canonical tokenisation.

    Why this is a hard check
    ------------------------
    ``alignment.align_sample`` maps the delivered per-word text features onto
    **our** word indices, taken from ``timeline.tokenize_transcript``.  The dry
    run of 2026-09-25 could not test this, because it generated
    ``text_words.csv`` with that same tokeniser.  If the two tokenisations
    differ, the failure modes are:

    * delivered words **more** than ours -> the aligner raises (loud, fine);
    * delivered words **fewer** than ours -> the tail words silently lose their
      text features and get ``text_mask = 0``;
    * same count, **different segmentation** -> every feature row is attached to
      the wrong word, silently.

    The last two are the dangerous ones, so this function fails loudly instead.
    ``handoff/q1_expected_words.csv`` publishes the canonical list, so the
    extractor side can satisfy it without guessing.
    """

    words_path = sample_dir / "text_words.csv"
    if not words_path.exists():
        issues.errors.append("missing required file: text_words.csv")
        return

    if not expected_words:
        # No transcript to compare against (an empty manifest text, or a caller
        # that did not pass one).  There is no contract to enforce, but saying so
        # beats pretending the axis was checked.
        issues.warnings.append(
            "text_words.csv word axis not cross-checked: no canonical "
            "tokenisation available for this sample"
        )
        return

    try:
        with words_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            rows = list(reader)
    except Exception as exc:  # pragma: no cover - defensive
        issues.errors.append(f"text_words.csv: {exc}")
        return

    for column in ("word_idx", "word"):
        if column not in fieldnames:
            issues.errors.append(f"text_words.csv: missing column {column!r}")
    if issues.errors and not rows:
        return

    if text_rows is not None and text_rows != len(rows):
        issues.errors.append(
            f"text_features rows {text_rows} != text_words.csv rows {len(rows)}"
        )

    if len(rows) != len(expected_words):
        # 2026-09-25：实测交付包用的是另一套分词（标点独立成 token：78/100 条；
        # 数字逐字符拆：4 条；引号/插入符等标注：2 条；逐词一致：14 条）。
        # 两套分词指的是同一段文本，因此按**字符级对账**把交付 token 归到我们的词；
        # 只有对账成功（差异全部落在标点/标注字符上、且单调覆盖全部词）才降级为警告，
        # 否则仍然报错——绝不能按行号静默错配。
        mapping, rep = build_mapping(expected_words, [r.get("word", "") for r in rows],
                                     sample_id=sample_dir.name)
        if rep.ok:
            issues.warnings.append(
                "text_words.csv 的分词与规范词轴不同但可逐字符对账："
                f"{rep.note}；对账规则见 q1_alignment/word_axis.py")
            issues.word_axis_mapping = mapping
            return
        issues.errors.append(
            f"text_words.csv has {len(rows)} words but this sample's transcript "
            f"tokenises to {len(expected_words)}; the word axis must match "
            f"handoff/q1_expected_words.csv exactly, or be reconcilable by the "
            f"character-level rule in q1_alignment/word_axis.py "
            f"（对账失败：{'；'.join(rep.errors[:2])}）"
        )
        return

    try:
        indices = [int(r["word_idx"]) for r in rows]
    except (TypeError, ValueError):
        issues.errors.append("text_words.csv: word_idx is not an integer column")
        return
    if indices != list(range(len(rows))):
        issues.errors.append(
            "text_words.csv: word_idx must be 0..N-1 in file order; "
            f"got {indices[:5]}{'...' if len(indices) > 5 else ''}"
        )

    delivered = [str(r.get("word", "")) for r in rows]
    normalized = [_norm_word(w) for w in delivered]
    expected_norm = [_norm_word(w) for w in expected_words]
    if normalized != expected_norm:
        # 词数相同但分词不同：同样先尝试字符级对账（见本函数开头的说明）
        mapping, rep = build_mapping(expected_words, delivered,
                                     sample_id=sample_dir.name)
        if rep.ok:
            issues.warnings.append(
                "text_words.csv 与规范词轴在个别位置分词不同但可逐字符对账："
                f"{rep.note}；对账规则见 q1_alignment/word_axis.py")
            issues.word_axis_mapping = mapping
        else:
            diffs = [
                (i, delivered[i], expected_words[i])
                for i in range(len(rows))
                if normalized[i] != expected_norm[i]
            ]
            issues.errors.append(
                f"text_words.csv word sequence differs from the canonical "
                f"tokenisation at {len(diffs)} position(s); first: "
                + "; ".join(f"#{i} delivered {d!r} != expected {e!r}"
                            for i, d, e in diffs[:3])
                + f"（字符级对账也失败：{'；'.join(rep.errors[:2])}）"
            )
    elif delivered != list(expected_words):
        diffs = [
            (i, delivered[i], expected_words[i])
            for i in range(len(rows))
            if delivered[i] != expected_words[i]
        ]
        issues.warnings.append(
            f"text_words.csv differs from the canonical spelling at "
            f"{len(diffs)} position(s) but is equivalent after "
            f"apostrophe/case folding "
            f"(e.g. #{diffs[0][0]} {diffs[0][1]!r} vs {diffs[0][2]!r})"
        )


def _load_npy(path: Path, issues: SampleIssues, key: str) -> np.ndarray | None:
    try:
        return np.load(path, allow_pickle=False)
    except Exception as exc:
        issues.errors.append(f"{key}: cannot load {path.name}: {exc}")
        return None


def _check_times(
    array: np.ndarray,
    name: str,
    issues: SampleIssues,
    duration: float | None,
    frame_count: int | None,
) -> None:
    """Validate a time array: shape, monotonicity, range."""

    if array.ndim == 2:
        if array.shape[1] != 2:
            issues.errors.append(f"{name}: expected [T,2] or [T], got {array.shape}")
            return
        if np.any(array[:, 0] > array[:, 1]):
            issues.errors.append(f"{name}: interval start exceeds end")
        centers = array.mean(axis=1)
    elif array.ndim == 1:
        centers = array
    else:
        issues.errors.append(f"{name}: expected 1D or [T,2], got {array.shape}")
        return

    if not np.isfinite(centers).all():
        issues.errors.append(f"{name}: contains NaN/Inf")
        return
    if centers.size > 1 and np.any(np.diff(centers) < -1e-9):
        issues.errors.append(f"{name}: timestamps are not monotonic")
    if centers.size and centers.min() < -0.05:
        issues.errors.append(f"{name}: negative timestamp {centers.min():.3f}")

    # The competition clips carry unreliable frame counts, so a strict
    # duration check is only a warning unless the mismatch is gross.
    if duration and centers.size:
        overrun = centers.max() - duration
        if overrun > max(0.5, 0.1 * duration):
            issues.errors.append(
                f"{name}: times exceed clip duration by {overrun:.2f}s "
                f"(max {centers.max():.3f} vs {duration:.3f})"
            )
        elif overrun > 0.05:
            issues.warnings.append(f"{name}: times exceed clip duration by {overrun:.3f}s")

    if frame_count and centers.size != frame_count:
        # Only vision timestamps are expected to line up with decodable frames.
        # Audio frames run at an independent rate (e.g. 100 Hz), so comparing
        # them against the video frame count produces a meaningless warning.
        if name.startswith("vision") and abs(centers.size - frame_count) > max(
            2, 0.02 * frame_count
        ):
            issues.warnings.append(
                f"{name}: {centers.size} rows vs {frame_count} decodable frames"
            )


def validate_sample(
    sample_dir: Path,
    sample_id: str,
    duration: float | None = None,
    frame_count: int | None = None,
    expected_words: Sequence[str] | None = None,
) -> SampleIssues:
    """Validate one sample directory without writing anything."""

    issues = SampleIssues(sample_id=sample_id)
    if not sample_dir.is_dir():
        issues.errors.append(f"missing sample directory: {sample_dir}")
        return issues

    for name in REQUIRED_FILES:
        if not (sample_dir / name).exists():
            issues.errors.append(f"missing required file: {name}")
    for name in OPTIONAL_FILES:
        if not (sample_dir / name).exists():
            issues.warnings.append(f"missing optional file: {name}")

    if issues.errors:
        return issues

    arrays: dict[str, np.ndarray] = {}
    for name, key in FILE_TO_KEY.items():
        path = sample_dir / name
        if not path.exists():
            continue
        array = _load_npy(path, issues, key)
        if array is not None:
            arrays[key] = array

    if issues.errors:
        return issues

    # ---- dtype / shape checks ----
    for key in ("text_features", "audio_features", "vision_features"):
        array = arrays.get(key)
        if array is None:
            continue
        if array.ndim != 2:
            issues.errors.append(f"{key}: expected 2D [T,D], got {array.shape}")
        elif not np.issubdtype(array.dtype, np.number):
            issues.errors.append(f"{key}: non-numeric dtype {array.dtype}")
        elif not np.isfinite(array).all():
            issues.errors.append(f"{key}: contains NaN/Inf")

    # ---- length consistency ----
    pairs = (
        ("audio_features", "audio_times"),
        ("vision_features", "vision_times"),
    )
    for feat, times in pairs:
        f, t = arrays.get(feat), arrays.get(times)
        if f is None or t is None:
            continue
        if f.shape[0] != t.shape[0]:
            issues.errors.append(
                f"{feat}/{times}: row count mismatch {f.shape[0]} vs {t.shape[0]}"
            )

    for key in ("audio_times", "vision_times"):
        if key in arrays:
            _check_times(arrays[key], key, issues, duration, frame_count)

    # Each per-frame valid mask must line up with its own modality's time axis.
    # The earlier version compared every mask against vision_times, so an
    # audio_valid array of the wrong length passed silently.
    mask_to_axis = {
        "vision_valid": "vision_times",
        "audio_valid": "audio_times",
        "text_valid": "text_features",
    }
    for key, axis_key in mask_to_axis.items():
        array = arrays.get(key)
        if array is None:
            continue
        if array.ndim != 1:
            issues.errors.append(f"{key}: expected 1D, got {array.shape}")
            continue
        axis = arrays.get(axis_key)
        if axis is not None and array.shape[0] != axis.shape[0]:
            issues.errors.append(
                f"{key}: length {array.shape[0]} != {axis_key} {axis.shape[0]}"
            )
        if array.size and not np.isin(array, [0, 1]).all():
            issues.warnings.append(f"{key}: values outside {0,1}")

    # ---- the word axis must match OUR canonical tokenisation ---------------
    # Without this, a delivery whose per-word rows do not line up with our word
    # indices corrupts the text channel *silently* (see check_word_axis).
    if expected_words is not None:
        text_features = arrays.get("text_features")
        check_word_axis(
            sample_dir,
            expected_words,
            issues,
            text_rows=(text_features.shape[0] if text_features is not None else None),
        )

    return issues


def load_sample(
    sample_dir: Path,
    issues: SampleIssues,
) -> dict[str, np.ndarray] | None:
    """Load one sample directory into the internal array dictionary."""

    arrays: dict[str, np.ndarray] = {}
    for name, key in FILE_TO_KEY.items():
        path = sample_dir / name
        if not path.exists():
            continue
        try:
            array = np.load(path, allow_pickle=False)
        except Exception as exc:
            issues.errors.append(f"{key}: {exc}")
            return None
        if key.endswith("_features"):
            arrays[key] = np.asarray(array, dtype=np.float32)
        elif key in ("vision_valid", "audio_valid", "text_valid"):
            # All three per-frame masks share the 0/1 uint8 convention so that
            # alignment.py can treat them uniformly.
            arrays[key] = np.asarray(array).astype(np.uint8).reshape(-1)
        elif key == "vision_confidence":
            arrays[key] = np.asarray(array, dtype=np.float32).reshape(-1)
        else:
            arrays[key] = np.asarray(array, dtype=np.float64)

    # preserve the per-word text record as an integer array of word indices
    words_csv = sample_dir / "text_words.csv"
    if words_csv.exists():
        try:
            with words_csv.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            if rows and "word_idx" in rows[0]:
                # 若分词与规范词轴不同，validate_sample 已经把**对账后的映射**
                # （交付 token 行 → 规范词索引）放在 issues 里；优先使用它。
                if (issues.word_axis_mapping is not None
                        and len(issues.word_axis_mapping) == len(rows)):
                    arrays["text_word_index"] = np.array(
                        issues.word_axis_mapping, dtype=np.int64)
                    arrays["text_word_axis_reconciled"] = np.array(1, dtype=np.int64)
                else:
                    arrays["text_word_index"] = np.array(
                        [int(r["word_idx"]) for r in rows], dtype=np.int64
                    )
                arrays["text_word_count"] = np.array(len(rows), dtype=np.int64)
        except Exception as exc:
            issues.warnings.append(f"text_words.csv: {exc}")

    return arrays


def import_handoff(
    handoff_dir: str | Path,
    manifest_path: str | Path,
    output_dir: str | Path,
    strict: bool = False,
) -> HandoffReport:
    """Convert a document-format handoff directory into internal npz bundles.

    Parameters
    ----------
    handoff_dir
        Root of the teammate's delivery (contains ``samples/``).
    manifest_path
        This project's ``manifest.jsonl``, used for ``sample_id`` -> ``safe_id``
        translation and for duration/frame-count cross-checks.
    output_dir
        Where ``<safe_id>.npz`` files are written.
    strict
        When True, any validation error aborts instead of being recorded.
    """

    root = Path(handoff_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = HandoffReport()

    manifest = {r["id"]: r for r in read_jsonl(manifest_path)}
    samples_root = root / "samples"
    if not samples_root.is_dir():
        report.errors.append(f"no samples/ directory under {root}")
        return report

    # sample_id in the handoff may use either separator
    def resolve(sample_id: str) -> dict[str, Any] | None:
        if sample_id in manifest:
            return manifest[sample_id]
        for candidate, record in manifest.items():
            if safe_id(candidate) == sample_id:
                return record
        return None

    directories = sorted(p for p in samples_root.iterdir() if p.is_dir())
    report.sample_count = len(directories)

    for directory in directories:
        sample_id = directory.name
        record = resolve(sample_id)
        if record is None:
            issues = SampleIssues(sample_id=sample_id)
            issues.errors.append("sample_id not found in manifest")
            report.per_sample.append(issues)
            report.failed += 1
            report.errors.append(f"{sample_id}: unknown sample id")
            if strict:
                raise ValueError(f"{sample_id}: unknown sample id")
            continue

        issues = validate_sample(
            directory,
            record["id"],
            duration=float(record.get("duration_sec", 0.0)) or None,
            frame_count=int(record.get("frame_count", 0)) or None,
            # the same gate the standalone validate-handoff command applies:
            # without it, import would happily accept a mis-aligned word axis
            expected_words=tokenize_transcript(record.get("text") or ""),
        )
        arrays = None
        if issues.ok:
            arrays = load_sample(directory, issues)

        if not issues.ok or arrays is None:
            report.per_sample.append(issues)
            report.failed += 1
            for message in issues.errors:
                report.errors.append(f"{record['id']}: {message}")
            if strict:
                raise ValueError(f"{record['id']}: {'; '.join(issues.errors)}")
            continue

        np.savez_compressed(out / f"{record['safe_id']}.npz", **arrays)
        report.imported += 1
        report.per_sample.append(issues)
        for message in issues.warnings:
            report.warnings.append(f"{record['id']}: {message}")

    return report


def write_report(report: HandoffReport, path: str | Path) -> None:
    """Write the validation/import outcome as JSON for the run log."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "sample_count": report.sample_count,
        "imported": report.imported,
        "failed": report.failed,
        "ok": report.ok,
        "errors": report.errors,
        "warnings": report.warnings,
        "per_sample": [
            {
                "sample_id": item.sample_id,
                "ok": item.ok,
                "errors": item.errors,
                "warnings": item.warnings,
            }
            for item in report.per_sample
        ],
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
