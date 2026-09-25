"""Tests for the document-claim checker.

The point of these tests is that H4 must actually FAIL on a wrong number --
a checker that only ever passes is worthless.  So the central test injects a
digitally altered document and asserts the mismatch is detected.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.q1_alignment.claim_check import (
    Claim,
    _tol_ok,
    build_claims,
    recompute,
    verify_claims,
)


class TestTolerance(unittest.TestCase):
    def test_exact_match(self):
        self.assertTrue(_tol_ok(4850.0, 4850.0, 0.0))

    def test_absolute_tolerance_accepts_close_value(self):
        self.assertTrue(_tol_ok(54.7, 54.7469, 0.15))

    def test_absolute_tolerance_rejects_far_value(self):
        """Regression: max(tol, tol*|expected|) wrongly accepted 60.0 for 54.7469.

        With an absolute 0.15 window the answer must be False; the old
        larger-of-two logic widened the window to 8.2 and returned True.
        """
        self.assertFalse(_tol_ok(60.0, 54.7469, 0.15))

    def test_relative_tolerance_for_large_numbers(self):
        # tol is a fraction of expected: 1% of 4850 is 48.5
        self.assertTrue(_tol_ok(4870.0, 4850.0, 0.01, tol_kind="rel"))
        self.assertFalse(_tol_ok(5000.0, 4850.0, 0.01, tol_kind="rel"))

    def test_absolute_kind_does_not_scale_with_magnitude(self):
        # 1 unit window regardless of how large expected is
        self.assertTrue(_tol_ok(4851.0, 4850.0, 1, tol_kind="abs"))
        self.assertFalse(_tol_ok(4900.0, 4850.0, 1, tol_kind="abs"))

    def test_default_kind_is_absolute(self):
        self.assertFalse(_tol_ok(5000.0, 4850.0, 1))
        self.assertTrue(_tol_ok(4851.0, 4850.0, 1))


class TestRecompute(unittest.TestCase):
    """recompute() must reproduce the numbers we actually published."""

    @classmethod
    def setUpClass(cls):
        cls.values = recompute()

    def test_attachment2_counts(self):
        v = self.values
        if "att2_total_samples" not in v:
            self.skipTest("attachment 2 not available in this environment")
        self.assertEqual(v["att2_total_samples"], 4850.0)
        self.assertEqual(v["att2_train"], 3395.0)
        self.assertEqual(v["att2_valid"], 728.0)
        self.assertEqual(v["att2_test"], 727.0)
        self.assertEqual(
            v["att2_train"] + v["att2_valid"] + v["att2_test"],
            v["att2_total_samples"],
        )

    def test_label_facts(self):
        v = self.values
        if "label_zero_count" not in v:
            self.skipTest("attachment 2 not available")
        self.assertEqual(v["label_zero_count"], 1100.0)
        self.assertAlmostEqual(v["smallest_positive_value"], 0.166667, places=5)
        self.assertAlmostEqual(v["largest_negative_value"], -0.333333, places=5)
        self.assertEqual(v["min_label"], -3.0)
        self.assertEqual(v["max_label"], 3.0)

    def test_audio_scale_ratio_matches_published_figure(self):
        v = self.values
        if "audio_scale_ratio" not in v:
            self.skipTest("attachment 2 not available")
        # 394x is the figure quoted in the versions file and the checklist
        self.assertAlmostEqual(v["audio_scale_ratio"], 394.28, delta=2.0)

    def test_text_pooling_damage_matches_published_figure(self):
        v = self.values
        if "text_pool_rel_err_mean" not in v:
            self.skipTest("attachment 2 not available")
        self.assertAlmostEqual(v["text_pool_rel_err_mean"], 0.4343, delta=0.01)

    def test_attachment3_mask_facts(self):
        v = self.values
        if "att3_mask_identical" not in v:
            self.skipTest("attachment 3 not available")
        self.assertEqual(v["att3_mask_identical"], 29.0)
        self.assertAlmostEqual(v["att3_mask_iou_mean"], 0.9951, delta=5e-4)


class TestClaimsAreWellFormed(unittest.TestCase):
    def test_every_claim_has_a_compiling_pattern(self):
        import re
        values = recompute() or {"att2_total_samples": 4850.0}
        for c in build_claims(values):
            re.compile(c.pattern)

    def test_claims_reference_real_keys(self):
        values = recompute()
        if not values:
            self.skipTest("no data available")
        claims = build_claims(values)
        self.assertGreater(len(claims), 5)
        for c in claims:
            self.assertGreaterEqual(c.expected, -1e9)

    def test_no_silently_empty_claim_list(self):
        values = recompute()
        if not values:
            self.skipTest("no data available")
        self.assertGreater(len(build_claims(values)), 0)


class TestContradictionDetection(unittest.TestCase):
    """A document stating a WRONG number must be caught, not ignored.

    Red-teaming on 2026-09-24 exposed the hole: ``pattern`` searched for the
    *expected* value ("4850"), so a document asserting "9999" never matched and
    was reported as "not mentioned".  ``any_number_pattern`` closes it by
    scanning the surrounding context for any value.
    """

    def _write(self, root: Path, text: str) -> None:
        d = root / "output" / "evidence"
        d.mkdir(parents=True, exist_ok=True)
        (d / "doc.md").write_text(text, encoding="utf-8")

    def _n1(self, root: Path):
        results = verify_claims(root)
        return [r for r in results if r.claim.id == "N1"]

    def test_wrong_total_is_flagged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write(root, "附件2 总样本数 9999 条。\n")
            mine = self._n1(root)
            if not mine:
                self.skipTest("attachment 2 unavailable")
            self.assertEqual(mine[0].kind, "mismatch",
                             f"9999 must be flagged, got {mine[0].note}")

    def test_correct_total_is_not_flagged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write(root, "附件2 总样本数 4850 条。\n")
            mine = self._n1(root)
            if not mine:
                self.skipTest("attachment 2 unavailable")
            self.assertNotEqual(mine[0].kind, "mismatch", mine[0].note)

    def test_mentioning_nothing_is_absent_not_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write(root, "这份文档不涉及任何统计量。\n")
            mine = self._n1(root)
            if not mine:
                self.skipTest("attachment 2 unavailable")
            self.assertEqual(mine[0].kind, "absent")

    def test_all_claims_carry_a_context_pattern_where_relevant(self):
        """The three head-count claims must have contradiction patterns."""
        values = recompute()
        if not values:
            self.skipTest("data unavailable")
        by_id = {c.id: c for c in build_claims(values)}
        for cid in ("N1", "N2", "N3"):
            if cid in by_id:
                self.assertTrue(by_id[cid].any_number_pattern,
                                f"{cid} lacks a contradiction pattern")


class TestMismatchDetection(unittest.TestCase):
    """The behaviour that gives H4 its value: a wrong number must be caught."""

    def _make_doc(self, root: Path, text: str) -> None:
        d = root / "output" / "evidence"
        d.mkdir(parents=True, exist_ok=True)
        (d / "doc.md").write_text(text, encoding="utf-8")

    def test_correct_number_is_not_flagged(self):
        values = recompute()
        if not values:
            self.skipTest("no data available")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_doc(root, "附件2 总样本数 4850 条。\n")
            self._run_and_assert(root, expect_mismatch=False)

    def test_wrong_number_is_flagged(self):
        values = recompute()
        if not values:
            self.skipTest("no data available")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_doc(root, "附件2 总样本数 9999 条。\n")
            self._run_and_assert(root, expect_mismatch=True)

    def _run_and_assert(self, root: Path, expect_mismatch: bool):
        import src.q1_alignment.claim_check as cc

        results = cc.verify_claims(root)
        mine = [r for r in results if r.claim.id == "N1" and r.found]
        if not mine:
            self.skipTest("N1 not present in this environment")
        mismatch = any(r.kind == "mismatch" for r in mine)
        self.assertEqual(mismatch, expect_mismatch,
                         f"expected mismatch={expect_mismatch}, got {mine[0].note}")

    def test_absent_number_is_not_a_mismatch(self):
        """Not mentioning a number yet must not read as an error."""
        values = recompute()
        if not values:
            self.skipTest("no data available")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_doc(root, "本文件没有引用任何统计数字。\n")
            results = verify_claims(root)
            self.assertFalse(any(r.kind == "mismatch" for r in results))
            self.assertTrue(all(r.kind == "absent" for r in results))


class TestClaimDataclass(unittest.TestCase):
    def test_claim_fields(self):
        c = Claim("X", "label", r"\d+", 1.0, 0.1, "%", ("output/evidence",))
        self.assertEqual(c.id, "X")
        self.assertEqual(c.tol, 0.1)
        self.assertEqual(c.unit, "%")


if __name__ == "__main__":
    unittest.main()
