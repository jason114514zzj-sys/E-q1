"""A spec-conformant delivery must survive the two digitally silent clips.

Why this file exists
--------------------
On 2026-09-25 an interface dry-run fed a *spec-conformant* 100-sample delivery
through the published entry points.  ``validate-handoff`` said 100/100 clean and
``import-handoff`` imported 100/100, but ``align`` **crashed**:

    ValueError: timeline is not marked final_usable: -mJ2ud6oKI8$_$1

Two of the 100 clips have a digitally silent audio track, so their word
timelines are deliberately marked ``final_usable=False`` (rule R8).  The
pipeline read that as a defect and aborted the whole run.  Nothing in the
existing 216 tests covered it, because the rehearsal handoff had never included
a silent sample.

These tests pin the three behaviours that matter:

1. an ``audio_absent`` timeline produces a complete 50-position record with a
   real text channel and audio/vision masked out everywhere;
2. ``run_alignment`` does not raise for it, and marks it in the report;
3. ``run_alignment`` **still raises** for a ``final_usable=False`` timeline whose
   audio is present -- the safety net must not be widened into silence.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from q1_alignment.alignment import align_sample
from q1_alignment.pipeline import run_alignment

SILENT_ID = "SIL$_$1"


def _sample() -> dict:
    return {
        "id": SILENT_ID,
        "safe_id": "SIL___1",
        "duration_sec": 6.0,
        "text": "alpha beta gamma",
    }


def _timeline(audio_present: bool, final_usable: bool) -> dict:
    if audio_present:
        words = [
            {"word": "alpha", "start": 0.0, "end": 2.0, "confidence": None},
            {"word": "beta", "start": 2.0, "end": 4.0, "confidence": None},
            {"word": "gamma", "start": 4.0, "end": 6.0, "confidence": None},
        ]
        method = "monotonic_constrained_pause_aware"
    else:
        words = [
            {"word": "alpha", "start": -1.0, "end": -1.0, "confidence": None},
            {"word": "beta", "start": -1.0, "end": -1.0, "confidence": None},
            {"word": "gamma", "start": -1.0, "end": -1.0, "confidence": None},
        ]
        method = "audio_absent"
    return {
        "id": SILENT_ID,
        "method": "monotonic",
        "alignment_method": method,
        "final_usable": final_usable,
        "words": words,
        "diagnostics": {
            "audio_present": audio_present,
            "energy_peak": 0.31 if audio_present else 1e-6,
        },
    }


def _bundle() -> dict:
    rng = np.random.default_rng(7)
    audio_times = np.linspace(0.0, 6.0, 60, dtype=np.float64)
    vision_times = np.linspace(0.0, 6.0, 30, dtype=np.float64)
    return {
        "text_features": rng.standard_normal((3, 8)).astype(np.float32),
        "audio_features": rng.standard_normal((60, 4)).astype(np.float32),
        "audio_times": audio_times,
        "vision_features": rng.standard_normal((30, 5)).astype(np.float32),
        "vision_times": vision_times,
        "vision_valid": np.ones(30, dtype=np.int64),
    }


class TestAlignSampleAudioAbsent(unittest.TestCase):
    def test_text_channel_stays_real(self) -> None:
        arrays, prov = align_sample(_sample(), _timeline(False, False), _bundle())
        # three words -> three content positions at 1,2,3 then [SEP] at 4
        self.assertEqual(int(arrays["effective_length"]), 5)
        self.assertEqual(list(arrays["text_mask"][1:4]), [1, 1, 1])
        self.assertTrue(prov["audio_absent"])

    def test_audio_and_vision_are_unusable_everywhere(self) -> None:
        arrays, _ = align_sample(_sample(), _timeline(False, False), _bundle())
        self.assertFalse(arrays["audio_mask"].any())
        self.assertFalse(arrays["vision_mask"].any())
        self.assertFalse(arrays["vision_valid_ratio"].any())
        self.assertFalse(arrays["audio"].any())
        self.assertFalse(arrays["vision"].any())

    def test_interval_arrays_keep_the_sentinel(self) -> None:
        arrays, _ = align_sample(_sample(), _timeline(False, False), _bundle())
        s = arrays["interval_start"]
        e = arrays["interval_end"]
        # R6: the sentinel means "no time evidence", and it must match the
        # audio mask exactly -- which is what guard C9 checks.
        self.assertTrue(np.all(s == -1.0))
        self.assertTrue(np.all(e == -1.0))
        self.assertTrue(np.array_equal(s == -1.0, arrays["audio_mask"] == 0))

    def test_sequence_mask_still_describes_a_real_sequence(self) -> None:
        arrays, _ = align_sample(_sample(), _timeline(False, False), _bundle())
        # CLS + 3 content + SEP are real sequence positions even though no
        # modality has time evidence there.
        self.assertEqual(arrays["sequence_mask"][0], 1)
        self.assertEqual(arrays["sequence_mask"][4], 1)
        self.assertEqual(int(arrays["sequence_mask"].sum()), 5)


class TestRunAlignmentAudioAbsent(unittest.TestCase):
    def _write_inputs(self, root: Path, timeline: dict) -> tuple[Path, Path, Path]:
        manifest = root / "manifest.jsonl"
        manifest.write_text(json.dumps(_sample()) + "\n", encoding="utf-8")
        timelines = root / "word_timelines.jsonl"
        timelines.write_text(json.dumps(timeline) + "\n", encoding="utf-8")
        features = root / "raw"
        features.mkdir()
        np.savez(
            features / "SIL___1.npz",
            **{k: np.asarray(v) for k, v in _bundle().items()},
        )
        return manifest, timelines, features

    def test_silent_sample_is_aligned_not_aborted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, timelines, features = self._write_inputs(
                root, _timeline(False, False)
            )
            out = root / "aligned"
            report = run_alignment(manifest, timelines, features, out)
            self.assertEqual(len(report), 1)
            self.assertTrue(report[0]["audio_absent"])
            self.assertTrue((out / "SIL___1.npz").exists())
            provenance = json.loads((out / "SIL___1.json").read_text(encoding="utf-8"))
            self.assertTrue(provenance["audio_absent"])
            self.assertEqual(provenance["alignment_method"], "audio_absent")

    def test_audible_but_unusable_timeline_still_raises(self) -> None:
        """The safety net must not be widened into a silent skip."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, timelines, features = self._write_inputs(
                root, _timeline(True, False)
            )
            with self.assertRaises(ValueError) as ctx:
                run_alignment(manifest, timelines, features, root / "aligned")
            self.assertIn("final_usable", str(ctx.exception))

    def test_audible_valid_timeline_is_unaffected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, timelines, features = self._write_inputs(
                root, _timeline(True, True)
            )
            out = root / "aligned"
            report = run_alignment(manifest, timelines, features, out)
            self.assertFalse(report[0]["audio_absent"])
            with np.load(out / "SIL___1.npz", allow_pickle=False) as z:
                self.assertTrue(z["audio_mask"][1:4].all())
                self.assertTrue(z["vision_mask"][1:4].all())
                self.assertTrue((z["interval_start"][1:4] >= 0).all())
                self.assertFalse(np.any(z["interval_start"][1:4] == -1.0))


if __name__ == "__main__":
    unittest.main()
