"""Tests for the reliable-timing layer and the CSV reports."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from q1_alignment.common import write_jsonl
from q1_alignment.ffprobe_utils import _parse_rate
from q1_alignment.reports import build_alignment_trace, build_qc_report


class FrameRateParsingTests(unittest.TestCase):
    def test_rational_and_plain_rates(self):
        self.assertAlmostEqual(_parse_rate("30000/1001"), 29.97002997, places=6)
        self.assertEqual(_parse_rate("30/1"), 30.0)
        self.assertEqual(_parse_rate("25"), 25.0)

    def test_missing_rate_is_zero(self):
        self.assertEqual(_parse_rate(None), 0.0)
        self.assertEqual(_parse_rate("N/A"), 0.0)
        self.assertEqual(_parse_rate("0/0"), 0.0)


class ReportTests(unittest.TestCase):
    def _fixture(self, root: Path) -> None:
        manifest = root / "manifest.jsonl"
        write_jsonl(
            manifest,
            [{
                "id": "video$_$1",
                "safe_id": "video__1",
                "duration_sec": 1.0,
                "text": "hello world",
            }],
        )
        npz = root / "video__1.npz"
        np.savez_compressed(
            npz,
            text=np.zeros((50, 4), dtype=np.float32),
            audio=np.zeros((50, 2), dtype=np.float32),
            vision=np.zeros((50, 3), dtype=np.float32),
            sequence_mask=np.array([1, 1, 1] + [0] * 47, dtype=np.uint8),
            attention_mask=np.array([1, 1, 1] + [0] * 47, dtype=np.uint8),
            text_mask=np.array([1, 1, 1] + [0] * 47, dtype=np.uint8),
            audio_mask=np.array([1, 1, 1] + [0] * 47, dtype=np.uint8),
            vision_mask=np.array([1, 1, 1] + [0] * 47, dtype=np.uint8),
            vision_valid_ratio=np.array([0.0, 1.0, 0.5] + [0.0] * 47, dtype=np.float32),
            interval_start=np.array([-1.0, 0.0, 0.5] + [-1.0] * 47, dtype=np.float32),
            interval_end=np.array([-1.0, 0.5, 1.0] + [-1.0] * 47, dtype=np.float32),
        )
        provenance = {
            "id": "video$_$1",
            "safe_id": "video__1",
            "bins": [
                {
                    "position": 1,
                    "word_start_index": 0,
                    "word_end_index": 0,
                    "text": "hello",
                    "start": 0.0,
                    "end": 0.5,
                    "text_feature_indices": [0],
                    "audio_feature_indices": [0, 1],
                    "vision_feature_indices": [0],
                },
                {
                    "position": 2,
                    "word_start_index": 1,
                    "word_end_index": 1,
                    "text": "world",
                    "start": 0.5,
                    "end": 1.0,
                    "text_feature_indices": [1],
                    "audio_feature_indices": [2],
                    "vision_feature_indices": [1],
                },
            ],
        }
        (root / "video__1.json").write_text(json.dumps(provenance), encoding="utf-8")

    def test_alignment_trace_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._fixture(root)
            out = root / "trace.csv"
            written = build_alignment_trace(root / "manifest.jsonl", root, out)
            self.assertEqual(written, 2)
            rows = list(csv.DictReader(out.open(encoding="utf-8")))
            self.assertEqual(rows[0]["words"], "hello")
            self.assertEqual(rows[0]["audio_frame_range"], "0-1")
            self.assertEqual(rows[0]["audio_frame_count"], "2")
            self.assertEqual(rows[1]["vision_valid_ratio"], "0.5")
            self.assertEqual(rows[0]["duration_sec"], "0.5")

    def test_qc_report_passes_clean_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._fixture(root)
            out = root / "qc.csv"
            result = build_qc_report(root / "manifest.jsonl", root, out)
            self.assertEqual(result["passed"], 1)
            self.assertEqual(result["failed"], 0)
            row = list(csv.DictReader(out.open(encoding="utf-8")))[0]
            self.assertEqual(row["status"], "pass")
            self.assertEqual(row["shape_ok"], "True")

    def test_qc_report_flags_missing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_jsonl(
                root / "manifest.jsonl",
                [{"id": "v$_$1", "safe_id": "v__1", "duration_sec": 1.0, "text": "a"}],
            )
            result = build_qc_report(root / "manifest.jsonl", root, root / "qc.csv")
            self.assertEqual(result["failed"], 0)
            self.assertEqual(result["missing"], 1)

    def test_qc_report_partial_run_marks_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_jsonl(
                root / "manifest.jsonl",
                [{"id": "v$_$1", "safe_id": "v__1", "duration_sec": 1.0, "text": "a"}],
            )
            result = build_qc_report(
                root / "manifest.jsonl", root, root / "qc.csv", allow_partial=True
            )
            self.assertEqual(result["failed"], 0)
            self.assertEqual(result["skipped"], 1)


class QcReportCompletenessTests(unittest.TestCase):
    """A '3 pass / 97 skipped' tally is ambiguous on its own.

    It can mean either "a 3-sample pilot" or "97 samples were silently
    dropped".  Observed on the real project 2026-09-24: qc_report.csv reported
    97 skipped while word_timelines.jsonl held all 100 samples.  These tests
    pin the fields that make the two cases distinguishable.
    """

    def _manifest(self, root, n):
        write_jsonl(
            root / "manifest.jsonl",
            [
                {"id": f"v$_${i}", "safe_id": f"v__{i}",
                 "duration_sec": 1.0, "text": "a"}
                for i in range(n)
            ],
        )

    def test_full_run_with_all_outputs_is_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._manifest(root, 2)
            for i in range(2):
                np.savez(root / f"v__{i}.npz", text=np.zeros((2, 2)))
                (root / f"v__{i}.json").write_text("{}")
            result = build_qc_report(root / "manifest.jsonl", root, root / "qc.csv")
            self.assertTrue(result["complete"])
            self.assertEqual(result["unprocessed_count"], 0)
            self.assertEqual(result["run_mode"], "full")

    def test_full_run_with_missing_outputs_is_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._manifest(root, 3)
            np.savez(root / "v__0.npz", text=np.zeros((2, 2)))
            (root / "v__0.json").write_text("{}")
            result = build_qc_report(root / "manifest.jsonl", root, root / "qc.csv")
            self.assertFalse(result["complete"])
            self.assertEqual(result["unprocessed_count"], 2)
            self.assertEqual(sorted(result["unprocessed_ids"]), ["v$_$1", "v$_$2"])
            self.assertEqual(result["run_mode"], "full")

    def test_partial_run_is_labelled_partial(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._manifest(root, 3)
            np.savez(root / "v__0.npz", text=np.zeros((2, 2)))
            (root / "v__0.json").write_text("{}")
            result = build_qc_report(
                root / "manifest.jsonl", root, root / "qc.csv", allow_partial=True
            )
            self.assertEqual(result["run_mode"], "partial")
            self.assertFalse(result["complete"])
            self.assertEqual(result["skipped"], 2)

    def test_counts_and_reason_are_reported_together(self):
        """The summary must never report a skip count without saying whether
        the run was meant to cover everything."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._manifest(root, 2)
            result = build_qc_report(root / "manifest.jsonl", root, root / "qc.csv")
            for key in ("samples", "passed", "failed", "missing", "skipped",
                        "complete", "unprocessed_count", "run_mode",
                        "awaiting_features", "verified", "feature_step_done",
                        "timeline_ready", "timeline_coverage"):
                self.assertIn(key, result)
            self.assertEqual(result["samples"], 2)
            self.assertEqual(result["unprocessed_count"], 2)


class QcTimelineVersusFeaturesTests(unittest.TestCase):
    """The word timeline and the feature arrays are separate deliverables.

    Regression guard for defect H3: with 100 timelines present and only 3
    feature arrays, QC reported "3 pass / 97 skipped", which reads exactly like
    a full run that dropped 97 samples.
    """

    def _manifest(self, root, n):
        write_jsonl(
            root / "manifest.jsonl",
            [
                {"id": f"v$_${i}", "safe_id": f"v__{i}",
                 "duration_sec": 1.0, "text": "a"}
                for i in range(n)
            ],
        )

    def _timelines(self, root, n):
        write_jsonl(
            root / "word_timelines.jsonl",
            [
                {"id": f"v$_${i}",
                 "words": [{"start": 0.0, "end": 0.5, "word": "a"}]}
                for i in range(n)
            ],
        )

    def test_timeline_ready_is_not_reported_as_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._manifest(root, 4)
            self._timelines(root, 4)
            # no feature arrays at all
            result = build_qc_report(
                root / "manifest.jsonl", root, root / "qc.csv",
                allow_partial=True,
                timeline_path=root / "word_timelines.jsonl",
            )
            self.assertEqual(result["skipped"], 0, "timeline-ready samples must not be 'skipped'")
            self.assertEqual(result["awaiting_features"], 4)
            self.assertEqual(result["timeline_ready"], 4)
            self.assertEqual(result["timeline_coverage"], 1.0)

    def test_without_timeline_path_behaviour_is_unchanged(self):
        """Callers that do not pass a timeline keep the old semantics."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._manifest(root, 3)
            self._timelines(root, 3)
            result = build_qc_report(
                root / "manifest.jsonl", root, root / "qc.csv", allow_partial=True
            )
            self.assertEqual(result["skipped"], 3)
            self.assertIsNone(result["timeline_ready"])

    def test_partial_timeline_reports_true_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._manifest(root, 4)
            self._timelines(root, 2)          # only 2 of 4 done
            result = build_qc_report(
                root / "manifest.jsonl", root, root / "qc.csv",
                allow_partial=True,
                timeline_path=root / "word_timelines.jsonl",
            )
            self.assertEqual(result["timeline_ready"], 2)
            self.assertEqual(result["timeline_coverage"], 0.5)

    def test_empty_timeline_words_counts_as_not_ready(self):
        """A record whose 'words' list is empty did not really produce a timeline."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._manifest(root, 2)
            write_jsonl(
                root / "word_timelines.jsonl",
                [{"id": "v$_$0", "words": []},
                 {"id": "v$_$1", "words": [{"start": 0.0, "end": 0.5, "word": "a"}]}],
            )
            result = build_qc_report(
                root / "manifest.jsonl", root, root / "qc.csv",
                allow_partial=True,
                timeline_path=root / "word_timelines.jsonl",
            )
            self.assertEqual(result["timeline_ready"], 1)

    def test_missing_timeline_file_degrades_gracefully(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._manifest(root, 2)
            result = build_qc_report(
                root / "manifest.jsonl", root, root / "qc.csv",
                allow_partial=True,
                timeline_path=root / "does_not_exist.jsonl",
            )
            self.assertEqual(result["timeline_ready"], 0)
            self.assertEqual(result["skipped"], 2)

    def test_timeline_ok_column_is_written(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._manifest(root, 2)
            self._timelines(root, 2)
            out = root / "qc.csv"
            build_qc_report(
                root / "manifest.jsonl", root, out,
                allow_partial=True,
                timeline_path=root / "word_timelines.jsonl",
            )
            with out.open(encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            self.assertIn("timeline_ok", rows[0])
            self.assertEqual(set(r["status"] for r in rows), {"awaiting_features"})

    def test_complete_does_not_imply_verified(self):
        """All-awaiting-features must not read as a finished run.

        complete=True only means "nothing was silently dropped"; verified counts
        samples that actually passed the feature-level checks.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._manifest(root, 3)
            self._timelines(root, 3)
            result = build_qc_report(
                root / "manifest.jsonl", root, root / "qc.csv",
                allow_partial=True,
                timeline_path=root / "word_timelines.jsonl",
            )
            self.assertTrue(result["complete"], "nothing was dropped")
            self.assertEqual(result["verified"], 0, "but nothing is verified either")
            self.assertFalse(result["feature_step_done"])

    def test_feature_step_done_requires_all_samples_verified(self):
        """With every feature array present the step is genuinely finished."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._manifest(root, 2)
            for i in range(2):
                np.savez(root / f"v__{i}.npz",
                         text=np.zeros((2, 2), dtype=np.float32),
                         audio=np.zeros((2, 3), dtype=np.float32),
                         vision=np.zeros((2, 4), dtype=np.float32))
                (root / f"v__{i}.json").write_text("{}")
            result = build_qc_report(root / "manifest.jsonl", root, root / "qc.csv")
            self.assertEqual(result["verified"], 2)
            self.assertTrue(result["feature_step_done"])
            self.assertEqual(result["awaiting_features"], 0)

    def test_verified_plus_unprocessed_accounts_for_all_samples(self):
        """Counts must partition the sample set -- no sample lost in between."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._manifest(root, 3)
            np.savez(root / "v__0.npz",
                     text=np.zeros((2, 2), dtype=np.float32),
                     audio=np.zeros((2, 3), dtype=np.float32),
                     vision=np.zeros((2, 4), dtype=np.float32))
            (root / "v__0.json").write_text("{}")
            result = build_qc_report(root / "manifest.jsonl", root, root / "qc.csv")
            self.assertEqual(
                result["verified"] + result["unprocessed_count"], result["samples"]
            )


if __name__ == "__main__":
    unittest.main()
