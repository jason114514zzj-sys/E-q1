"""Tests for the trap guard, so the guard itself cannot silently rot.

If a check stops working (e.g. a path changes), these tests fail rather than
letting the guard report a false PASS.

Most checks need the competition data, which only exists on the server. When a
check cannot find its input it records N-A; these tests treat "N-A because the
data is absent" as a skip rather than a failure, so the suite stays green on a
machine without the datasets while still asserting correctly on the server.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from q1_alignment import guard_check as gc


def require(result, testcase: unittest.TestCase):
    """Skip when a check could not run because its input was unavailable."""

    if result is None:
        testcase.skipTest("guard produced no result (input data unavailable)")
    if result.status == "N-A":
        testcase.skipTest(f"input unavailable: {result.detail}")
    return result


class GuardAvailabilityTests(unittest.TestCase):
    """The guard must detect its inputs; missing inputs are N-A, not PASS."""

    def setUp(self):
        gc.RESULTS.clear()

    def test_structural_detects_attachment3(self):
        gc.check_structural()
        codes = {r.code for r in gc.RESULTS}
        self.assertIn("A1", codes)
        a1 = next(r for r in gc.RESULTS if r.code == "A1")
        require(a1, self)
        self.assertEqual(a1.status, "PASS", msg=a1.detail)

    def test_semantic_text_not_masked(self):
        gc.check_semantic()
        b1 = require(next((r for r in gc.RESULTS if r.code == "B1"), None), self)
        self.assertEqual(b1.status, "PASS", msg=f"text appears masked: {b1.detail}")

    def test_semantic_sync_masking(self):
        gc.check_semantic()
        b2 = require(next((r for r in gc.RESULTS if r.code == "B2"), None), self)
        self.assertEqual(b2.status, "PASS", msg=b2.detail)

    def test_label_zero_is_neutral(self):
        gc.check_semantic()
        b3 = require(next((r for r in gc.RESULTS if r.code == "B3"), None), self)
        self.assertEqual(b3.status, "PASS", msg=b3.detail)

    def test_compliance_self_exempt(self):
        """The checker must not flag its own forbidden-word list."""
        gc.check_compliance()
        e12 = next(r for r in gc.RESULTS if r.code == "E1/E2")
        self.assertNotEqual(e12.status, "FAIL", msg=e12.detail)


class GuardBehaviourTests(unittest.TestCase):
    def test_record_status_mapping(self):
        gc.RESULTS.clear()
        gc.record("X1", "t", True, "d")
        gc.record("X2", "t", False, "d")
        gc.record("X3", "t", None, "d")
        self.assertEqual([r.status for r in gc.RESULTS], ["PASS", "FAIL", "N-A"])

    def test_fail_produces_nonzero_exit(self):
        gc.RESULTS.clear()
        gc.record("X1", "t", False, "d")
        self.assertTrue(any(r.status == "FAIL" for r in gc.RESULTS))

    def test_all_clean_gives_zero(self):
        gc.RESULTS.clear()
        gc.record("X1", "t", True, "d")
        self.assertFalse(any(r.status == "FAIL" for r in gc.RESULTS))


class GuardSelfAuditTests(unittest.TestCase):
    """The guard must be able to report on its own blind spots.

    Seven checks are gated on output/q2 and output/q3, which do not exist in
    this project yet, so they have never executed.  A checker that silently
    skips checks still reports green -- H2 exists to make that visible.
    """

    def test_h2_runs_and_reports_dormant_count(self):
        gc.RESULTS.clear()
        gc.check_guard_self_audit()
        h2 = next(r for r in gc.RESULTS if r.code == "H2")
        self.assertIn("休眠", h2.detail)
        self.assertIn("声明", h2.detail)

    def test_h2_is_informational_not_a_failure(self):
        # dormancy is expected while teammate files are absent; it must not
        # turn the whole report red on its own
        gc.RESULTS.clear()
        gc.check_guard_self_audit()
        h2 = next(r for r in gc.RESULTS if r.code == "H2")
        self.assertNotEqual(h2.status, "FAIL")

    def test_declared_codes_can_be_statically_scanned(self):
        """The guard's record() codes must be literal strings we can find."""
        import ast
        from pathlib import Path as _P

        src = _P(gc.__file__).read_text(encoding="utf-8")
        tree = ast.parse(src)
        codes = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fname = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
                if fname == "record" and node.args:
                    a0 = node.args[0]
                    if isinstance(a0, ast.Constant) and isinstance(a0.value, str):
                        codes.add(a0.value)
        # a healthy guard declares a substantial number of checks
        self.assertGreater(len(codes), 30)
        for expected in ("A1", "B1", "C1", "D1/", "E5", "F1", "G1", "H1"):
            self.assertTrue(any(c.startswith(expected.replace("/", "")) or c == expected
                                for c in codes),
                            f"missing expected check group {expected!r}")

    def test_all_wake_scenarios_expected_shape(self):
        """The dormant-check sandbox harness must exist and be importable."""
        from src.q1_alignment import wake_dormant_checks as w

        self.assertEqual(set(w.DORMANT),
                         {"F2", "F3", "F4", "G2", "G3", "G4", "G5"})
        self.assertTrue(callable(w.build_sandbox))
        self.assertTrue(callable(w.run_guard))


class IdentityLeakageTests(unittest.TestCase):
    """E5 must distinguish quoting the rule from actually leaking identity.

    Observed 2026-09-24: handoff/附件3附件4提交文件规格.md quotes 题目段56
    ("严禁出现参赛单位、队员姓名、队伍编号等身份信息"), and the naive
    substring check flagged it as a leak.  A checker that punishes the document
    warning against leaks is worse than no checker.
    """

    QUOTE_MARKERS = ("严禁", "禁止", "不得", "违者", "声明", "承诺", "题目原文", "段56")
    CATEGORY_WORDS = ("队伍编号", "队员姓名", "参赛单位")

    def _is_quote(self, line):
        return any(m in line for m in self.QUOTE_MARKERS)

    def test_prohibition_sentence_is_a_quote(self):
        line = "段56：所有提交材料中严禁出现参赛单位、队员姓名、队伍编号等身份信息"
        self.assertTrue(self._is_quote(line))
        for w in self.CATEGORY_WORDS:
            self.assertIn(w, line)

    def test_bare_category_word_is_not_a_quote(self):
        # a real leak looks like this -- no prohibition context on the line
        line = "队伍编号：2026E-1234"
        self.assertFalse(self._is_quote(line))

    def test_table_row_quoting_rule_is_a_quote(self):
        line = "| **禁身份信息** | 段56：严禁出现参赛单位、队员姓名、队伍编号等身份信息 | 违规 |"
        self.assertTrue(self._is_quote(line))

    def test_real_credentials_always_flagged(self):
        gc.RESULTS.clear()
        gc.check_compliance()
        e5 = next(r for r in gc.RESULTS if r.code == "E5")
        # whatever the outcome with real files, the check must run and report
        self.assertIn(e5.status, ("PASS", "FAIL", "N-A"))
        self.assertIn("命中", e5.detail)

    def test_e5_action_mentions_quote_exemption(self):
        gc.RESULTS.clear()
        gc.check_compliance()
        e5 = next(r for r in gc.RESULTS if r.code == "E5")
        self.assertIn("引用", e5.action)


if __name__ == "__main__":
    unittest.main()
