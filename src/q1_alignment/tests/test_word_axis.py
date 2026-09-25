"""The delivered word axis must line up with OUR canonical tokenisation.

Why this file exists
--------------------
The 2026-09-25 interface dry-run fed a spec-conformant delivery through the
published entry points and reported 6/6 green.  It could **not** test the word
axis, because it generated ``text_words.csv`` with the same tokeniser the
aligner uses (``timeline.tokenize_transcript``).  A user question exposed the
hole: if the extractor side segments the transcript differently, the failure is
*silent* --

* delivered words **more** than ours  -> aligner raises (loud);
* delivered words **fewer** than ours -> tail words lose their text features and
  quietly get ``text_mask = 0``;
* same count, different segmentation  -> every row attaches to the wrong word.

These tests pin the three defences added for that:

1. ``adapter.check_word_axis`` (used by both ``validate-handoff`` and
   ``import-handoff``) fails loudly on count / order / segmentation mismatch, and
   only warns when the difference is pure typography (curly vs straight quote);
2. ``align_sample`` records a ``text_word_coverage`` block in the provenance so a
   partial mapping is never invisible;
3. ``run_alignment`` surfaces that block in its report instead of staying quiet.
"""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from q1_alignment.adapter import SampleIssues, check_word_axis
from q1_alignment.alignment import align_sample
from q1_alignment.pipeline import run_alignment
from q1_alignment.timeline import tokenize_transcript

WORDS = ["They've", "been", "able", "to", "find", "solutions"]
SAMPLE_ID = "AAA$_$0"


def _write_words(path: Path, words: list[str], *, header: tuple[str, ...] = ("word_idx", "word")) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for index, word in enumerate(words):
            writer.writerow([index, word])


class TestCheckWordAxis(unittest.TestCase):
    def _issues(self, tmp: Path, words: list[str], text_rows: int | None = None,
                expected: list[str] | None = None, **kwargs) -> SampleIssues:
        _write_words(tmp / "text_words.csv", words, **kwargs)
        issues = SampleIssues(sample_id=SAMPLE_ID)
        check_word_axis(tmp, expected if expected is not None else WORDS, issues,
                        text_rows=text_rows)
        return issues

    def test_exact_match_passes(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            issues = self._issues(Path(d), list(WORDS), text_rows=len(WORDS))
            self.assertEqual(issues.errors, [])
            self.assertEqual(issues.warnings, [])

    def test_fewer_words_is_an_error_not_a_silent_shortfall(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            issues = self._issues(Path(d), WORDS[:4], text_rows=4)
            self.assertFalse(issues.ok)
            self.assertIn("tokenises to 6", issues.errors[0])

    def test_more_words_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            issues = self._issues(Path(d), WORDS + ["extra"], text_rows=7)
            self.assertFalse(issues.ok)
            self.assertIn("6", issues.errors[0])

    def test_same_count_different_segmentation_is_an_error(self) -> None:
        """The silent mis-alignment case: rows map to the wrong words."""
        with tempfile.TemporaryDirectory() as d:
            other = ["They", "ve", "been", "able", "to", "find"]
            issues = self._issues(Path(d), other, text_rows=6)
            self.assertFalse(issues.ok)
            self.assertIn("canonical tokenisation", issues.errors[0])
            self.assertIn("#0", issues.errors[0])

    def test_non_sequential_word_idx_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "text_words.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["word_idx", "word"])
                for i, w in enumerate(WORDS):
                    writer.writerow([i * 2, w])
            issues = SampleIssues(sample_id=SAMPLE_ID)
            check_word_axis(Path(d), WORDS, issues, text_rows=len(WORDS))
            self.assertFalse(issues.ok)
            self.assertIn("0..N-1", issues.errors[0])

    def test_text_features_row_count_must_match_words(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            issues = self._issues(Path(d), list(WORDS), text_rows=99)
            self.assertFalse(issues.ok)
            self.assertTrue(any("text_features rows" in e for e in issues.errors))

    def test_missing_word_column_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            issues = self._issues(Path(d), list(WORDS), header=("word_idx", "token"))
            self.assertFalse(issues.ok)
            self.assertTrue(any("missing column" in e for e in issues.errors))

    def test_curly_apostrophe_only_warns(self) -> None:
        """Typography is not a mapping error -- the comparison folds it away."""
        with tempfile.TemporaryDirectory() as d:
            curly = ["They\u2019ve"] + WORDS[1:]
            issues = self._issues(Path(d), curly, text_rows=len(WORDS))
            self.assertEqual(issues.errors, [], issues.errors)
            self.assertEqual(len(issues.warnings), 1)
            self.assertIn("equivalent after", issues.warnings[0])

    def test_missing_transcript_skips_with_a_warning(self) -> None:
        """No canonical tokenisation -> say so, do not invent a failure."""
        with tempfile.TemporaryDirectory() as d:
            issues = self._issues(Path(d), list(WORDS), text_rows=len(WORDS),
                                  expected=[])
            self.assertEqual(issues.errors, [])
            self.assertTrue(any("not cross-checked" in w for w in issues.warnings))


def _sample() -> dict:
    return {"id": SAMPLE_ID, "safe_id": "AAA___0", "duration_sec": 4.0,
            "text": " ".join(WORDS)}


def _timeline() -> dict:
    n = len(WORDS)
    span = 4.0 / n
    return {
        "id": SAMPLE_ID,
        "method": "monotonic",
        "alignment_method": "monotonic_constrained_pause_aware",
        "final_usable": True,
        "words": [{"word": w, "start": i * span, "end": (i + 1) * span,
                   "confidence": None} for i, w in enumerate(WORDS)],
        "diagnostics": {"audio_present": True, "energy_peak": 0.4},
    }


def _bundle(word_index: "np.ndarray | None", rows: int) -> dict:
    rng = np.random.default_rng(11)
    times = np.linspace(0.0, 4.0, 40, dtype=np.float64)
    bundle = {
        "text_features": rng.standard_normal((rows, 8)).astype(np.float32),
        "audio_features": rng.standard_normal((40, 4)).astype(np.float32),
        "audio_times": times,
        "vision_features": rng.standard_normal((20, 5)).astype(np.float32),
        "vision_times": np.linspace(0.0, 4.0, 20, dtype=np.float64),
        "vision_valid": np.ones(20, dtype=np.int64),
    }
    if word_index is not None:
        bundle["text_word_index"] = np.asarray(word_index, dtype=np.int64)
    return bundle


class TestAlignSampleCoverage(unittest.TestCase):
    def test_full_coverage_is_recorded_as_complete(self) -> None:
        _, prov = align_sample(_sample(), _timeline(),
                              _bundle(np.arange(len(WORDS)), len(WORDS)))
        cov = prov["text_word_coverage"]
        self.assertEqual(cov["covered"], cov["total"])
        self.assertIsNone(cov["first_uncovered"])

    def test_partial_coverage_is_recorded_with_first_missing_index(self) -> None:
        _, prov = align_sample(_sample(), _timeline(),
                              _bundle(np.arange(4), 4))
        cov = prov["text_word_coverage"]
        self.assertEqual((cov["covered"], cov["total"]), (4, 6))
        self.assertEqual(cov["first_uncovered"], 4)


class TestRunAlignmentSurfacesCoverage(unittest.TestCase):
    def _run(self, word_index, rows) -> tuple[list[dict], Path]:
        tmp = Path(tempfile.mkdtemp(prefix="q1_wordaxis_"))
        (tmp / "manifest.jsonl").write_text(json.dumps(_sample()) + "\n", encoding="utf-8")
        (tmp / "word_timelines.jsonl").write_text(json.dumps(_timeline()) + "\n",
                                                 encoding="utf-8")
        features = tmp / "raw"
        features.mkdir()
        np.savez(features / "AAA___0.npz",
                 **{k: np.asarray(v) for k, v in _bundle(word_index, rows).items()})
        report = run_alignment(tmp / "manifest.jsonl", tmp / "word_timelines.jsonl",
                              features, tmp / "aligned")
        return report, tmp / "aligned"

    def test_partial_mapping_is_flagged_in_the_report(self) -> None:
        report, aligned = self._run(np.arange(4), 4)
        self.assertEqual(len(report), 1)
        self.assertIn("text_word_coverage", report[0])
        self.assertIn("4/6", report[0]["issue"])
        provenance = json.loads((aligned / "AAA___0.json").read_text(encoding="utf-8"))
        self.assertEqual(provenance["text_word_coverage"]["first_uncovered"], 4)

    def test_complete_mapping_leaves_no_issue(self) -> None:
        report, _ = self._run(np.arange(len(WORDS)), len(WORDS))
        self.assertNotIn("issue", report[0])


class TestTokenizerIsTheContract(unittest.TestCase):
    """The tokeniser is the interface, so its behaviour must be pinned."""

    def test_contractions_stay_one_word(self) -> None:
        self.assertEqual(tokenize_transcript("They've been able"),
                         ["They've", "been", "able"])

    def test_tokenizer_preserves_the_apostrophe_glyph_it_is_given(self) -> None:
        # It does NOT normalise; that is deliberate, because the transcript is
        # authoritative.  Glyph equivalence is handled at comparison time
        # (adapter._norm_word), which is why this test asserts a *difference*.
        straight = tokenize_transcript("They've been")
        curly = tokenize_transcript("They\u2019ve been")
        self.assertNotEqual(straight, curly)
        self.assertEqual([w.replace("\u2019", "'") for w in curly], straight)

    def test_comparison_folds_apostrophe_glyphs_together(self) -> None:
        from q1_alignment.adapter import _norm_word

        for variant in ("They\u2019ve", "They\u2018ve", "They\u02bcve",
                        "They\u2032ve", "They`ve", "THEY'VE"):
            self.assertEqual(_norm_word(variant), _norm_word("They've"),
                             variant)


if __name__ == "__main__":
    unittest.main()
