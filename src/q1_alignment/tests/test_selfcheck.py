"""Tests for the teammate-facing handoff self-check tool.

The tool is the teammate's only feedback loop before delivery, so its messages
must be accurate: a missed defect means a broken delivery, and a false alarm
wastes their time.
"""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from q1_alignment.handoff_selfcheck import check_sample

DURATION = 4.0


def build(root: Path, sample_id: str = "v$_$1") -> Path:
    d = root / "samples" / sample_id
    d.mkdir(parents=True, exist_ok=True)
    # root-level delivery manifest files, as the collaboration document requires
    for name in ("manifest.csv", "config.yaml", "versions.txt", "run_log.csv"):
        (root / name).write_text("placeholder\n", encoding="utf-8")
    n_audio, n_vision = int(DURATION * 100), int(DURATION * 30)
    np.save(d / "text_features.npy", np.random.randn(6, 768).astype(np.float32))
    np.save(d / "audio_features.npy", np.random.randn(n_audio, 74).astype(np.float32))
    st = np.arange(n_audio) * (DURATION / n_audio)
    np.save(d / "audio_intervals.npy", np.column_stack([st, st + DURATION / n_audio]))
    np.save(d / "vision_features.npy", np.random.randn(n_vision, 35).astype(np.float32))
    np.save(d / "vision_timestamps.npy",
            np.linspace(0, DURATION, n_vision, endpoint=False))
    np.save(d / "vision_valid.npy", np.ones(n_vision, dtype=np.uint8))
    np.save(d / "vision_confidence.npy", np.full(n_vision, 0.9, dtype=np.float32))
    # Per-modality time masks (contract section 10).  Written here so that a
    # "clean sample" really produces zero findings.
    np.save(d / "audio_valid.npy", np.ones(n_audio, dtype=np.uint8))
    np.save(d / "text_valid.npy", np.ones(6, dtype=np.uint8))
    with (d / "text_words.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["word_idx", "word"])
        for i in range(6):
            w.writerow([i, f"w{i}"])
    (d / "metadata.json").write_text(json.dumps({"sample_id": sample_id}))
    return d


def levels(report) -> set[str]:
    return {f.level for f in report.findings}


class CleanSampleTests(unittest.TestCase):
    def test_clean_sample_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = build(Path(tmp))
            report = check_sample(d, duration=DURATION)
            self.assertTrue(report.ok, msg=[f.what for f in report.errors])
            self.assertEqual(report.findings, [])

    def test_stats_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = build(Path(tmp))
            report = check_sample(d, duration=DURATION)
            self.assertEqual(report.stats["text_dim"], 768)
            self.assertEqual(report.stats["audio_dim"], 74)
            self.assertEqual(report.stats["vision_dim"], 35)
            self.assertEqual(report.stats["word_count"], 6)


class RequiredFileTests(unittest.TestCase):
    def test_missing_text_words_is_an_error(self):
        """v1.1 promoted this from a warning: the paper needs it for tracing."""
        with tempfile.TemporaryDirectory() as tmp:
            d = build(Path(tmp))
            (d / "text_words.csv").unlink()
            report = check_sample(d, duration=DURATION)
            self.assertFalse(report.ok)
            self.assertTrue(any("text_words.csv" in f.what for f in report.errors))
            self.assertTrue(all(f.level == "缺失" for f in report.errors))

    def test_missing_vision_valid_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = build(Path(tmp))
            (d / "vision_valid.npy").unlink()
            report = check_sample(d, duration=DURATION)
            self.assertFalse(report.ok)
            self.assertTrue(any("vision_valid" in f.what for f in report.errors))

    def test_missing_confidence_is_only_a_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = build(Path(tmp))
            (d / "vision_confidence.npy").unlink()
            report = check_sample(d, duration=DURATION)
            self.assertTrue(report.ok)
            self.assertIn("警告", levels(report))

    def test_every_finding_has_guidance(self):
        """A finding without 'why' and 'how' is not actionable."""
        with tempfile.TemporaryDirectory() as tmp:
            d = build(Path(tmp))
            (d / "text_features.npy").unlink()
            report = check_sample(d, duration=DURATION)
            for finding in report.findings:
                self.assertTrue(finding.what, msg="missing what")
                self.assertTrue(finding.why, msg=f"missing why: {finding.what}")
                self.assertTrue(finding.how, msg=f"missing how: {finding.what}")
                self.assertIn(finding.level, ("错误", "缺失", "警告"))


class ContentTests(unittest.TestCase):
    def test_nan_reports_row_numbers(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = build(Path(tmp))
            a = np.load(d / "text_features.npy")
            a[3, 0] = np.nan
            np.save(d / "text_features.npy", a)
            report = check_sample(d, duration=DURATION)
            message = " ".join(f.what for f in report.errors)
            self.assertIn("NaN", message)
            self.assertIn("[3]", message)

    def test_non_monotonic_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = build(Path(tmp))
            v = np.load(d / "vision_timestamps.npy")
            v[10] = v[4]
            np.save(d / "vision_timestamps.npy", v)
            report = check_sample(d, duration=DURATION)
            self.assertTrue(any("递增" in f.what for f in report.errors))

    def test_row_count_mismatch_vision(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = build(Path(tmp))
            np.save(d / "vision_timestamps.npy", np.linspace(0, DURATION, 7))
            report = check_sample(d, duration=DURATION)
            self.assertTrue(any("vision" in f.what and "行" in f.what
                                for f in report.errors))

    def test_severe_overrun_is_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = build(Path(tmp))
            n = np.load(d / "vision_timestamps.npy").shape[0]
            np.save(d / "vision_timestamps.npy",
                    np.linspace(0, DURATION * 3, n))
            report = check_sample(d, duration=DURATION)
            self.assertTrue(any("超出视频时长" in f.what for f in report.errors))
            match = [f for f in report.errors if "超出视频时长" in f.what][0]
            self.assertIn("ffprobe", match.how)

    def test_minor_overrun_is_warning_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = build(Path(tmp))
            n = np.load(d / "vision_timestamps.npy").shape[0]
            np.save(d / "vision_timestamps.npy",
                    np.linspace(0, DURATION + 0.2, n))
            report = check_sample(d, duration=DURATION)
            self.assertTrue(report.ok)
            self.assertTrue(any("略超" in f.what for f in report.findings))

    def test_negative_time_is_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = build(Path(tmp))
            n = np.load(d / "vision_timestamps.npy").shape[0]
            np.save(d / "vision_timestamps.npy",
                    np.linspace(-1.0, DURATION - 1.0, n))
            report = check_sample(d, duration=DURATION)
            self.assertTrue(any("负时间" in f.what for f in report.errors))

    def test_interval_start_after_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = build(Path(tmp))
            a = np.load(d / "audio_intervals.npy")
            a[5] = [a[5, 1], a[5, 0]]
            np.save(d / "audio_intervals.npy", a)
            report = check_sample(d, duration=DURATION)
            self.assertTrue(any("开始时间 > 结束时间" in f.what for f in report.errors))

    def test_valid_mask_length_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = build(Path(tmp))
            np.save(d / "vision_valid.npy", np.ones(5, dtype=np.uint8))
            report = check_sample(d, duration=DURATION)
            self.assertTrue(any("vision_valid" in f.what for f in report.errors))

    def test_bad_words_csv_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = build(Path(tmp))
            with (d / "text_words.csv").open("w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["idx", "token"])
                w.writerow([0, "a"])
            report = check_sample(d, duration=DURATION)
            message = " ".join(f.what for f in report.errors)
            self.assertIn("word_idx", message)
            self.assertIn("word", message)

    def test_word_count_mismatch_is_warning_with_two_options(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = build(Path(tmp))
            np.save(d / "text_features.npy", np.random.randn(20, 768).astype(np.float32))
            # text_valid.npy travels with text_features, so it must grow too;
            # otherwise the length mismatch is a genuine error and masks the
            # word-count warning this test is about.
            np.save(d / "text_valid.npy", np.ones(20, dtype=np.uint8))
            report = check_sample(d, duration=DURATION)
            self.assertTrue(report.ok, msg=[f.what for f in report.errors])
            match = [f for f in report.findings if "不一致" in f.what]
            self.assertTrue(match)
            self.assertIn("子词", match[0].how)

    def test_text_valid_length_mismatch_is_an_error(self):
        """The mask must track its own modality's feature rows."""
        with tempfile.TemporaryDirectory() as tmp:
            d = build(Path(tmp))
            np.save(d / "text_valid.npy", np.ones(99, dtype=np.uint8))
            report = check_sample(d, duration=DURATION)
            self.assertFalse(report.ok)
            self.assertTrue(any("text_valid" in f.what for f in report.errors),
                            f"errors: {[f.what for f in report.errors]}")

    def test_one_dimensional_features_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = build(Path(tmp))
            np.save(d / "text_features.npy", np.random.randn(6).astype(np.float32))
            report = check_sample(d, duration=DURATION)
            self.assertTrue(any("2 维" in f.what for f in report.errors))


class PaddingMisunderstandingTests(unittest.TestCase):
    """50-position layout is the aligner's job; producers must send raw rows."""

    def test_flags_manually_padded_array(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = build(Path(tmp))
            padded = np.vstack([np.random.randn(20, 768), np.zeros((30, 768))])
            np.save(d / "text_features.npy", padded.astype(np.float32))
            report = check_sample(d, duration=DURATION)
            self.assertTrue(any("疑似已手动补齐到 50" in f.what for f in report.findings))
            match = [f for f in report.findings if "手动补齐" in f.what][0]
            self.assertIn("原始粒度", match.how)

    def test_does_not_flag_genuine_50_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = build(Path(tmp))
            np.save(d / "text_features.npy",
                    np.random.randn(50, 768).astype(np.float32))
            report = check_sample(d, duration=DURATION)
            self.assertFalse(any("补齐" in f.what for f in report.findings))

    def test_does_not_flag_short_tail_zeros(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = build(Path(tmp))
            a = np.vstack([np.random.randn(47, 768), np.zeros((3, 768))])
            np.save(d / "text_features.npy", a.astype(np.float32))
            report = check_sample(d, duration=DURATION)
            self.assertFalse(any("补齐" in f.what for f in report.findings))

    def test_does_not_flag_variable_length(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = build(Path(tmp))
            np.save(d / "text_features.npy",
                    np.random.randn(15, 768).astype(np.float32))
            report = check_sample(d, duration=DURATION)
            self.assertFalse(any("补齐" in f.what for f in report.findings))


if __name__ == "__main__":
    unittest.main()
