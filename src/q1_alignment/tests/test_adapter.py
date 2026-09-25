"""Tests for the collaboration-document handoff adapter.

A mock delivery is synthesised exactly as the collaboration document specifies
(one directory per sample with the nine named files) so the adapter is verified
against the documented contract rather than against this project's own format.
"""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from q1_alignment.adapter import (
    import_handoff,
    load_sample,
    validate_sample,
)
from q1_alignment.common import write_jsonl


def make_sample_dir(
    samples_root: Path,
    sample_id: str,
    text_rows: int = 5,
    audio_rows: int = 100,
    vision_rows: int = 60,
    duration: float = 2.0,
    with_optional: bool = True,
) -> Path:
    """Write one sample in the exact document-format layout.

    ``with_optional=False`` omits only the *genuinely optional* extras
    (vision_confidence, metadata, and the two time-mask files).  The required
    set now includes text_words.csv and vision_valid.npy, so those are always
    written -- the contract and the self-check agree on them.
    """

    directory = samples_root / sample_id
    directory.mkdir(parents=True, exist_ok=True)

    np.save(directory / "text_features.npy",
            np.random.randn(text_rows, 768).astype(np.float32))
    np.save(directory / "audio_features.npy",
            np.random.randn(audio_rows, 74).astype(np.float32))
    intervals = np.column_stack([
        np.linspace(0, duration, audio_rows, endpoint=False),
        np.linspace(duration / audio_rows, duration, audio_rows),
    ])
    np.save(directory / "audio_intervals.npy", intervals)
    np.save(directory / "vision_features.npy",
            np.random.randn(vision_rows, 35).astype(np.float32))
    np.save(directory / "vision_timestamps.npy",
            np.linspace(0, duration, vision_rows, endpoint=False))

    # ---- always required alongside the arrays ----
    np.save(directory / "vision_valid.npy", np.ones(vision_rows, dtype=np.uint8))
    with (directory / "text_words.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["word_idx", "word"])
        for index in range(text_rows):
            writer.writerow([index, f"w{index}"])

    if with_optional:
        np.save(directory / "vision_confidence.npy",
                np.random.rand(vision_rows).astype(np.float32) * 0.5 + 0.5)
        np.save(directory / "audio_valid.npy", np.ones(audio_rows, dtype=np.uint8))
        np.save(directory / "text_valid.npy", np.ones(text_rows, dtype=np.uint8))
        (directory / "metadata.json").write_text(
            json.dumps({"sample_id": sample_id, "note": "ok"}), encoding="utf-8"
        )
    return directory


class ValidateSampleTests(unittest.TestCase):
    def test_accepts_document_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "samples"
            make_sample_dir(root, "video$_$1")
            issues = validate_sample(root / "video$_$1", "video$_$1", duration=2.0)
            self.assertTrue(issues.ok, msg=issues.errors)

    def test_flags_missing_required_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "samples"
            directory = make_sample_dir(root, "video$_$1")
            (directory / "audio_intervals.npy").unlink()
            issues = validate_sample(directory, "video$_$1", duration=2.0)
            self.assertFalse(issues.ok)
            self.assertTrue(any("audio_intervals" in e for e in issues.errors))

    def test_flags_row_count_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "samples"
            directory = make_sample_dir(root, "video$_$1", audio_rows=100)
            np.save(directory / "audio_intervals.npy",
                    np.zeros((50, 2), dtype=np.float64))
            issues = validate_sample(directory, "video$_$1", duration=2.0)
            self.assertFalse(issues.ok)
            self.assertTrue(any("mismatch" in e for e in issues.errors))

    def test_flags_non_monotonic_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "samples"
            directory = make_sample_dir(root, "video$_$1", vision_rows=10)
            bad = np.linspace(0, 2.0, 10)
            bad[5] = bad[3]  # break monotonicity
            np.save(directory / "vision_timestamps.npy", bad)
            issues = validate_sample(directory, "video$_$1", duration=2.0)
            self.assertFalse(issues.ok)
            self.assertTrue(any("monotonic" in e for e in issues.errors))

    def test_flags_nan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "samples"
            directory = make_sample_dir(root, "video$_$1", text_rows=4)
            features = np.random.randn(4, 768).astype(np.float32)
            features[0, 0] = np.nan
            np.save(directory / "text_features.npy", features)
            issues = validate_sample(directory, "video$_$1", duration=2.0)
            self.assertFalse(issues.ok)
            self.assertTrue(any("NaN" in e for e in issues.errors))

    def test_warns_on_missing_optional(self):
        """Omitting a genuinely optional file warns; it must not fail."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "samples"
            make_sample_dir(root, "video$_$1", with_optional=False)
            issues = validate_sample(root / "video$_$1", "video$_$1", duration=2.0)
            self.assertEqual(issues.errors, [], f"errors: {issues.errors}")
            self.assertTrue(issues.ok)
            self.assertTrue(any("optional" in w for w in issues.warnings),
                            f"warnings: {issues.warnings}")

    def test_missing_now_required_file_is_an_error(self):
        """text_words.csv and vision_valid.npy are required after the
        contract/self-check reconciliation; their absence must fail loudly."""
        for name in ("text_words.csv", "vision_valid.npy"):
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp) / "samples"
                    directory = make_sample_dir(root, "video$_$1")
                    (directory / name).unlink()
                    issues = validate_sample(directory, "video$_$1", duration=2.0)
                    self.assertFalse(issues.ok)
                    self.assertTrue(any(name in e for e in issues.errors),
                                    f"errors: {issues.errors}")


class LoadSampleTests(unittest.TestCase):
    def test_renames_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "samples"
            make_sample_dir(root, "video$_$1")
            issues = validate_sample(root / "video$_$1", "video$_$1", duration=2.0)
            arrays = load_sample(root / "video$_$1", issues)
            self.assertIsNotNone(arrays)
            # document names -> internal names
            self.assertIn("audio_times", arrays)
            self.assertIn("vision_times", arrays)
            self.assertNotIn("audio_intervals", arrays)
            self.assertNotIn("vision_timestamps", arrays)
            self.assertEqual(arrays["text_features"].dtype, np.float32)
            self.assertEqual(arrays["vision_valid"].dtype, np.uint8)

    def test_preserves_word_index_and_confidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "samples"
            make_sample_dir(root, "video$_$1", text_rows=7)
            issues = validate_sample(root / "video$_$1", "video$_$1", duration=2.0)
            arrays = load_sample(root / "video$_$1", issues)
            self.assertEqual(arrays["text_word_index"].shape[0], 7)
            self.assertIn("vision_confidence", arrays)


class ImportHandoffTests(unittest.TestCase):
    def _manifest(self, root: Path, ids: list[str]) -> Path:
        path = root / "manifest.jsonl"
        write_jsonl(
            path,
            [{"id": i, "safe_id": i.replace("$_$", "__"), "duration_sec": 2.0} for i in ids],
        )
        return path

    def test_imports_matching_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            samples = root / "handoff" / "samples"
            make_sample_dir(samples, "video$_$1")
            make_sample_dir(samples, "video$_$2")
            manifest = self._manifest(root, ["video$_$1", "video$_$2"])
            out = root / "raw_features"

            report = import_handoff(root / "handoff", manifest, out)
            self.assertTrue(report.ok, msg=report.errors)
            self.assertEqual(report.imported, 2)
            self.assertTrue((out / "video__1.npz").exists())
            self.assertTrue((out / "video__2.npz").exists())

    def test_imported_bundle_is_readable_by_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            samples = root / "handoff" / "samples"
            make_sample_dir(samples, "video$_$1")
            manifest = self._manifest(root, ["video$_$1"])
            out = root / "raw_features"
            import_handoff(root / "handoff", manifest, out)

            from q1_alignment.pipeline import load_bundle

            bundle = load_bundle(out / "video__1.npz")
            for key in ("text_features", "audio_features", "audio_times",
                        "vision_features", "vision_times"):
                self.assertIn(key, bundle)

    def test_records_unknown_sample_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            samples = root / "handoff" / "samples"
            make_sample_dir(samples, "video$_$1")
            make_sample_dir(samples, "ghost$_$9")
            manifest = self._manifest(root, ["video$_$1"])
            report = import_handoff(root / "handoff", manifest, root / "out")
            self.assertEqual(report.imported, 1)
            self.assertEqual(report.failed, 1)
            self.assertFalse(report.ok)

    def test_records_invalid_sample_without_aborting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            samples = root / "handoff" / "samples"
            make_sample_dir(samples, "video$_$1")
            broken = make_sample_dir(samples, "video$_$2")
            (broken / "vision_features.npy").unlink()
            manifest = self._manifest(root, ["video$_$1", "video$_$2"])

            report = import_handoff(root / "handoff", manifest, root / "out")
            self.assertEqual(report.imported, 1)
            self.assertEqual(report.failed, 1)

    def test_strict_mode_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            samples = root / "handoff" / "samples"
            broken = make_sample_dir(samples, "video$_$1")
            (broken / "text_features.npy").unlink()
            manifest = self._manifest(root, ["video$_$1"])
            with self.assertRaises(ValueError):
                import_handoff(root / "handoff", manifest, root / "out", strict=True)

    def test_missing_samples_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self._manifest(root, ["video$_$1"])
            report = import_handoff(root / "nowhere", manifest, root / "out")
            self.assertFalse(report.ok)
            self.assertEqual(report.imported, 0)


if __name__ == "__main__":
    unittest.main()
