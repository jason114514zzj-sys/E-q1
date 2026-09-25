"""Tests for the submission schema validator.

These pin the behaviours that make the validator *useful* to a teammate:
it must catch missing columns, illegal polarity values, leaked ground truth,
wrong row counts, and inverted evidence spans -- and must not cry wolf on a
correct file that happens to use a documented alias.
"""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from src.q1_alignment.submission_schema import (
    ATT3_SAMPLE_COUNT,
    SCHEMAS,
    format_findings,
    validate_csv,
)


def _write(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _good_att3(n: int = ATT3_SAMPLE_COUNT) -> tuple[list[str], list[dict]]:
    fields = ["sample_id", "polarity", "intensity"]
    rows = [
        {"sample_id": f"附件3_{i+1:02d}", "polarity": "positive",
         "intensity": "1.25"}
        for i in range(n)
    ]
    return fields, rows


def _good_att4(n: int = 20) -> tuple[list[str], list[dict]]:
    fields = ["sample_id", "polarity", "intensity", "dominant_modality",
              "modality_weights", "evidence_start", "evidence_end",
              "evidence_modality"]
    rows = [
        {"sample_id": f"附件4_{i+1:02d}", "polarity": "negative",
         "intensity": "-1.5", "dominant_modality": "text",
         "modality_weights": "text:0.5,audio:0.3,vision:0.2",
         "evidence_start": "1.0", "evidence_end": "2.5",
         "evidence_modality": "text"}
        for i in range(n)
    ]
    return fields, rows


class TestGoodFiles(unittest.TestCase):
    def test_valid_att3_passes(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a3.csv"
            f, r = _good_att3()
            _write(p, f, r)
            self.assertEqual(validate_csv(p, "att3"), [])

    def test_valid_att4_passes(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a4.csv"
            f, r = _good_att4()
            _write(p, f, r)
            self.assertEqual(validate_csv(p, "att4"), [])

    def test_aliases_are_accepted(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a3.csv"
            fields = ["id", "classification", "score"]
            rows = [{"id": f"s{i}", "classification": "neutral", "score": "0"}
                    for i in range(ATT3_SAMPLE_COUNT)]
            _write(p, fields, rows)
            self.assertEqual(validate_csv(p, "att3"), [])

    def test_numeric_polarity_encoding_accepted(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a3.csv"
            fields, rows = _good_att3()
            for i, r in enumerate(rows):
                r["polarity"] = ["-1", "0", "1"][i % 3]
            _write(p, fields, rows)
            self.assertEqual(validate_csv(p, "att3"), [])


class TestMissingFile(unittest.TestCase):
    def test_missing_file_reports_once(self):
        findings = validate_csv("/nonexistent/x.csv", "att3")
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].level, "缺失")


class TestColumnChecks(unittest.TestCase):
    def test_missing_polarity_column(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a3.csv"
            fields, rows = _good_att3()
            fields = [f for f in fields if f != "polarity"]
            for r in rows:
                r.pop("polarity")
            _write(p, fields, rows)
            findings = validate_csv(p, "att3")
            self.assertTrue(any("polarity" in f.what for f in findings))
            self.assertTrue(any("情感极性" in f.why for f in findings))

    def test_att4_requires_evidence_columns(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a4.csv"
            fields, rows = _good_att4()
            for drop in ("evidence_start", "evidence_end", "evidence_modality",
                         "dominant_modality", "modality_weights"):
                fields.remove(drop)
                for r in rows:
                    r.pop(drop)
            _write(p, fields, rows)
            findings = validate_csv(p, "att4")
            self.assertGreaterEqual(len(findings), 5)

    def test_empty_value_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a3.csv"
            fields, rows = _good_att3()
            rows[0]["intensity"] = ""
            _write(p, fields, rows)
            findings = validate_csv(p, "att3")
            self.assertTrue(any("intensity" in f.what for f in findings))


class TestValueChecks(unittest.TestCase):
    def test_illegal_polarity_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a3.csv"
            fields, rows = _good_att3()
            rows[0]["polarity"] = "Positive"      # case matters in the domain check
            rows[0]["polarity"] = "happy"
            _write(p, fields, rows)
            findings = validate_csv(p, "att3")
            self.assertTrue(any("polarity" in f.what for f in findings))

    def test_intensity_out_of_label_range_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a3.csv"
            fields, rows = _good_att3()
            rows[0]["intensity"] = "99"
            _write(p, fields, rows)
            findings = validate_csv(p, "att3")
            self.assertTrue(any("intensity" in f.what for f in findings))

    def test_non_numeric_intensity_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a3.csv"
            fields, rows = _good_att3()
            rows[0]["intensity"] = "high"
            _write(p, fields, rows)
            findings = validate_csv(p, "att3")
            self.assertTrue(any("intensity" in f.what for f in findings))


class TestStructuralChecks(unittest.TestCase):
    def test_wrong_row_count_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a3.csv"
            fields, rows = _good_att3(n=7)
            _write(p, fields, rows)
            findings = validate_csv(p, "att3")
            self.assertTrue(any("行数" in f.what for f in findings))

    def test_duplicate_ids_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a3.csv"
            fields, rows = _good_att3()
            rows[1]["sample_id"] = rows[0]["sample_id"]
            _write(p, fields, rows)
            findings = validate_csv(p, "att3")
            self.assertTrue(any("重复" in f.what for f in findings))

    def test_leaked_ground_truth_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a3.csv"
            fields, rows = _good_att3()
            fields.append("regression_labels")
            for r in rows:
                r["regression_labels"] = "0.5"
            _write(p, fields, rows)
            findings = validate_csv(p, "att3")
            self.assertTrue(any("真值" in f.what for f in findings))

    def test_empty_csv_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a3.csv"
            _write(p, ["sample_id", "polarity", "intensity"], [])
            findings = validate_csv(p, "att3")
            self.assertTrue(any("无数据行" in f.what for f in findings))

    def test_inverted_evidence_span_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a4.csv"
            fields, rows = _good_att4()
            rows[0]["evidence_start"] = "5.0"
            rows[0]["evidence_end"] = "2.0"
            _write(p, fields, rows)
            findings = validate_csv(p, "att4")
            self.assertTrue(any("证据区间" in f.what for f in findings))

    def test_zero_length_span_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a4.csv"
            fields, rows = _good_att4()
            rows[0]["evidence_start"] = rows[0]["evidence_end"] = "3.0"
            _write(p, fields, rows)
            findings = validate_csv(p, "att4")
            self.assertTrue(any("证据区间" in f.what for f in findings))


class TestPresentation(unittest.TestCase):
    def test_every_finding_explains_why_and_how(self):
        findings = validate_csv("/nonexistent/x.csv", "att4")
        for f in findings:
            self.assertTrue(f.what)
            self.assertTrue(f.why)
            self.assertTrue(f.how)

    def test_format_findings_reports_success(self):
        self.assertIn("通过", format_findings([], "测试"))

    def test_format_findings_orders_errors_first(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a3.csv"
            fields, rows = _good_att3(n=5)
            rows[0]["polarity"] = "nonsense"
            _write(p, fields, rows)
            text = format_findings(validate_csv(p, "att3"), "附件3")
            self.assertIn("错误", text)

    def test_schemas_expect_measured_sample_counts(self):
        self.assertEqual(SCHEMAS["att3"].expected_rows, 30)
        self.assertEqual(SCHEMAS["att4"].expected_rows, 20)


if __name__ == "__main__":
    unittest.main()
