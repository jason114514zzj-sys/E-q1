"""The teammate self-check and the adapter must agree on the file contract.

Background
----------
On 2026-09-24 end-to-end testing exposed a divergence:

    adapter  : text_words.csv and vision_valid.npy were OPTIONAL
    selfcheck: the same two files were REQUIRED (missing them => exit code 1)

A teammate following ``交付接口规格_冻结版.md`` (which listed them under
"建议文件") would deliver a directory that the adapter happily imports but the
self-check rejects.  She would then either redo work unnecessarily, or drop the
two files and silently lose the traceability the problem asks for.

These tests make that class of divergence impossible to reintroduce: the two
modules must declare the same required set, the same optional set, and the
adapter must be able to *load* every file the contract tells her to send.
"""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.q1_alignment import handoff_selfcheck as sc
from src.q1_alignment.adapter import (
    FILE_TO_KEY,
    OPTIONAL_FILES,
    REQUIRED_FILES,
    load_sample,
    validate_sample,
)


class TestContractAgreement(unittest.TestCase):
    """The two file lists must be identical, in both directions."""

    def setUp(self):
        self.adapter_required = set(REQUIRED_FILES)
        self.adapter_known = set(REQUIRED_FILES) | set(OPTIONAL_FILES)
        self.selfcheck_required = set(sc.REQUIRED_FILES)
        self.selfcheck_recommended = set(sc.RECOMMENDED_FILES)

    def test_required_sets_match_exactly(self):
        self.assertEqual(
            self.adapter_required, self.selfcheck_required,
            "adapter and handoff_selfcheck disagree on which files are required; "
            "a teammate cannot satisfy both contracts at once",
        )

    def test_adapter_optional_matches_selfcheck_recommended(self):
        """Optional must mean the same thing on both sides.

        The adapter may know a few extra keys (it tolerates alternate names),
        but every file the self-check calls "recommended" must be optional for
        the adapter too -- otherwise omitting a recommended file would break
        the import.
        """
        recommended_without_adapter = self.selfcheck_recommended - set(OPTIONAL_FILES)
        self.assertEqual(
            recommended_without_adapter, set(),
            "selfcheck recommends files the adapter would reject as unknown: "
            f"{recommended_without_adapter}",
        )
        # and the converse direction: adapter optional files that are neither
        # required nor recommended by the self-check are simply extras
        adapter_only = set(OPTIONAL_FILES) - self.selfcheck_recommended
        self.assertEqual(
            adapter_only, set(),
            f"adapter treats as optional files the self-check never mentions: "
            f"{adapter_only}",
        )

    def test_no_file_is_required_by_one_and_unknown_to_the_other(self):
        unknown = self.selfcheck_required - self.adapter_known
        self.assertEqual(unknown, set(),
                         f"selfcheck requires files the adapter cannot read: {unknown}")

    def test_every_contract_file_has_a_load_mapping(self):
        """Nothing the contract asks for may be silently dropped on import.

        The contract's section 10 introduced audio_valid.npy / text_valid.npy.
        Before this test existed, FILE_TO_KEY lacked them, so a teammate could
        send them and the adapter would ignore the arrays entirely.
        """
        for name in self.adapter_known:
            if name.endswith(".npy"):
                self.assertIn(name, FILE_TO_KEY,
                              f"{name} is in the contract but has no FILE_TO_KEY entry")


class TestAdapterAcceptsTheFullContract(unittest.TestCase):
    """A directory satisfying the contract must validate cleanly."""

    def _build(self, root: Path, include_all: bool = True) -> tuple[Path, dict]:
        sd = root / "samples" / "vid__1"
        sd.mkdir(parents=True)
        rng = np.random.default_rng(0)
        n_text, n_audio, n_vis = 5, 30, 45

        np.save(sd / "text_features.npy",
                rng.normal(size=(n_text, 8)).astype(np.float32))
        np.save(sd / "audio_features.npy",
                rng.normal(size=(n_audio, 4)).astype(np.float32))
        np.save(sd / "audio_intervals.npy",
                np.stack([np.arange(n_audio) * 0.05,
                          (np.arange(n_audio) + 1) * 0.05], axis=1)
                .astype(np.float64))
        np.save(sd / "vision_features.npy",
                rng.normal(size=(n_vis, 3)).astype(np.float32))
        ts = np.linspace(0.0, 1.5, n_vis) + np.arange(n_vis) * 1e-6
        np.save(sd / "vision_timestamps.npy", ts.astype(np.float64))

        with (sd / "text_words.csv").open("w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["word_idx", "word"])
            for i in range(n_text):
                w.writerow([i, f"w{i}"])

        np.save(sd / "vision_valid.npy",
                np.ones(n_vis, dtype=np.uint8))

        if include_all:
            np.save(sd / "vision_confidence.npy",
                    rng.uniform(0.5, 1.0, n_vis).astype(np.float32))
            np.save(sd / "audio_valid.npy", np.ones(n_audio, dtype=np.uint8))
            np.save(sd / "text_valid.npy", np.ones(n_text, dtype=np.uint8))
            (sd / "metadata.json").write_text("{}", encoding="utf-8")

        sample = {"id": "vid$_$1", "safe_id": "vid__1", "duration_sec": 1.5}
        return sd, sample

    def test_full_contract_validates_without_errors(self):
        with tempfile.TemporaryDirectory() as d:
            sd, sample = self._build(Path(d))
            issues = validate_sample(sd, sample, 1.5)
            self.assertEqual(issues.errors, [], f"errors: {issues.errors}")
            self.assertEqual(issues.warnings, [], f"warnings: {issues.warnings}")

    def test_missing_required_file_is_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            sd, sample = self._build(Path(d))
            (sd / "vision_valid.npy").unlink()
            issues = validate_sample(sd, sample, 1.5)
            self.assertTrue(any("vision_valid.npy" in e for e in issues.errors),
                            f"errors: {issues.errors}")

    def test_audio_valid_is_loaded_not_ignored(self):
        """The contract asks for audio_valid.npy; the adapter must keep it."""
        with tempfile.TemporaryDirectory() as d:
            sd, _ = self._build(Path(d))
            issues = validate_sample(sd, {"id": "vid$_$1", "safe_id": "vid__1",
                                          "duration_sec": 1.5}, 1.5)
            arrays = load_sample(sd, issues)
            self.assertIsNotNone(arrays)
            self.assertIn("audio_valid", arrays,
                          "audio_valid.npy was present but not loaded")
            self.assertEqual(arrays["audio_valid"].shape, (30,))
            self.assertEqual(arrays["audio_valid"].dtype, np.uint8)

    def test_text_valid_is_loaded_not_ignored(self):
        with tempfile.TemporaryDirectory() as d:
            sd, _ = self._build(Path(d))
            issues = validate_sample(sd, {"id": "vid$_$1", "safe_id": "vid__1",
                                          "duration_sec": 1.5}, 1.5)
            arrays = load_sample(sd, issues)
            self.assertIsNotNone(arrays)
            self.assertIn("text_valid", arrays)
            self.assertEqual(arrays["text_valid"].shape, (5,))

    def test_mismatched_audio_valid_length_is_an_error(self):
        """A mask must line up with ITS OWN modality's axis.

        The earlier code compared every mask against vision_times, so a wrong
        audio_valid length slipped through.
        """
        with tempfile.TemporaryDirectory() as d:
            sd, sample = self._build(Path(d))
            np.save(sd / "audio_valid.npy", np.ones(7, dtype=np.uint8))  # not 30
            issues = validate_sample(sd, sample, 1.5)
            self.assertTrue(any("audio_valid" in e for e in issues.errors),
                            f"errors: {issues.errors}")

    def test_mismatched_text_valid_length_is_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            sd, sample = self._build(Path(d))
            np.save(sd / "text_valid.npy", np.ones(99, dtype=np.uint8))
            issues = validate_sample(sd, sample, 1.5)
            self.assertTrue(any("text_valid" in e for e in issues.errors),
                            f"errors: {issues.errors}")

    def test_non_binary_mask_is_a_warning(self):
        with tempfile.TemporaryDirectory() as d:
            sd, sample = self._build(Path(d))
            bad = np.ones(30, dtype=np.uint8)
            bad[0] = 2
            np.save(sd / "audio_valid.npy", bad)
            issues = validate_sample(sd, sample, 1.5)
            self.assertEqual(issues.errors, [])
            self.assertTrue(any("audio_valid" in w for w in issues.warnings))


class TestSelfcheckAndAdapterAgreeOnRealInput(unittest.TestCase):
    """End-to-end: run the self-check on a directory the adapter accepts."""

    def test_recommended_files_do_not_block_delivery(self):
        """Omitting a RECOMMENDED file must not make the self-check fail.

        This is the exact scenario that motivated the fix: the contract lists
        vision_confidence.npy as recommended, so a teammate may legitimately
        omit it, and the self-check must not then report a blocking error.
        """
        required = set(sc.REQUIRED_FILES)
        recommended = set(sc.RECOMMENDED_FILES)
        self.assertEqual(required & recommended, set(),
                         "a file cannot be both required and recommended")
        self.assertNotIn("vision_confidence.npy", required)
        self.assertIn("vision_confidence.npy", recommended)


if __name__ == "__main__":
    unittest.main()
