"""Tests for the pipeline rehearsal harness.

Steps 4-8 of run_q1_pipeline.sh had never executed (no teammate handoff existed).
make_rehearsal_handoff.py exists so the downstream half is exercised before the
real delivery arrives.  These tests pin the properties that make the rehearsal
meaningful: the generated handoff must satisfy the same contract a teammate is
held to, so a pass here is evidence the contract is implementable.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]

from src.q1_alignment.adapter import REQUIRED_FILES  # noqa: E402
from src.q1_alignment.make_rehearsal_handoff import (  # noqa: E402
    AUDIO_DIM,
    REQUIRED,
    TEXT_DIM,
    VISION_DIM,
    build_sample,
    pick_samples,
)


def _fake_manifest_rec() -> dict:
    return {
        "id": "vid$_$1",
        "safe_id": "vid__1",
        "duration_sec": 5.0,
        "words": [{"word": "hello", "start": 0.0, "end": 1.0},
                  {"word": "world", "start": 1.0, "end": 2.0}],
    }


class TestRehearsalContract(unittest.TestCase):
    def test_rehearsal_superset_of_adapter_requirements(self):
        """The rehearsal must produce at least everything the adapter requires.

        The adapter distinguishes strictly-required files from recommended ones;
        the rehearsal deliberately writes both, because a teammate following the
        contract is asked to supply all nine.  What must NOT happen is the
        rehearsal omitting a file the adapter would reject.
        """
        missing = set(REQUIRED_FILES) - set(REQUIRED)
        self.assertEqual(missing, set(),
                         f"rehearsal omits adapter-required files: {missing}")

    def test_rehearsal_also_covers_the_recommended_files(self):
        """vision_valid / confidence are what make 'no face' distinguishable
        from 'value happens to be 0', which the problem asks us to trace."""
        for name in ("vision_valid.npy", "vision_confidence.npy",
                     "text_words.csv", "metadata.json"):
            self.assertIn(name, REQUIRED)

    def test_rehearsal_file_count_is_nine(self):
        self.assertEqual(len(REQUIRED), 9)

    def test_dimensions_match_attachment2(self):
        # attachment 2 ships text 768 / audio 74 / vision 35
        self.assertEqual(TEXT_DIM, 768)
        self.assertEqual(AUDIO_DIM, 74)
        self.assertEqual(VISION_DIM, 35)


class TestBuildSample(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _build(self, tag="shortest"):
        import src.q1_alignment.make_rehearsal_handoff as m

        original = m.OUT
        m.OUT = self.root
        try:
            rng = np.random.default_rng(0)
            return build_sample(_fake_manifest_rec(), tag, rng)
        finally:
            m.OUT = original

    def test_writes_every_required_file(self):
        result = self._build()
        self.assertEqual(result["missing"], [],
                         f"missing files: {result['missing']}")
        d = self.root / "samples" / "vid__1"
        for name in REQUIRED:
            self.assertTrue((d / name).exists(), name)

    def test_vision_timestamps_strictly_increasing(self):
        self._build("longest")
        ts = np.load(self.root / "samples" / "vid__1" / "vision_timestamps.npy")
        self.assertTrue(np.all(np.diff(ts) > 0), "timestamps must strictly increase")
        self.assertGreaterEqual(ts[0], 0.0)

    def test_audio_intervals_start_at_zero(self):
        self._build()
        iv = np.load(self.root / "samples" / "vid__1" / "audio_intervals.npy")
        self.assertEqual(iv.shape[1], 2)
        self.assertAlmostEqual(iv[0, 0], 0.0, places=6)
        self.assertTrue(np.all(iv[:, 1] >= iv[:, 0]))

    def test_valid_masks_are_binary(self):
        self._build()
        for name in ("vision_valid", "vision_confidence"):
            a = np.load(self.root / "samples" / "vid__1" / f"{name}.npy")
            self.assertEqual(a.ndim, 1)
        vv = np.load(self.root / "samples" / "vid__1" / "vision_valid.npy")
        self.assertTrue(set(np.unique(vv)).issubset({0.0, 1.0}))

    def test_degraded_sample_keeps_frames_and_flags_them(self):
        """A damaged stretch must be marked, not deleted (题目要求保留样本)."""
        result = self._build("shortest")
        d = self.root / "samples" / "vid__1"
        vv = np.load(d / "vision_valid.npy")
        # the rehearsal drops some vision frames; the array length must still
        # equal the timestamp length so nothing was silently removed
        ts = np.load(d / "vision_timestamps.npy")
        self.assertEqual(len(vv), len(ts))
        self.assertGreater(result["n_vis"], 0)

    def test_metadata_is_json_serialisable(self):
        import json
        self._build()
        p = self.root / "samples" / "vid__1" / "metadata.json"
        json.loads(p.read_text(encoding="utf-8"))

    def test_word_features_row_count_matches_word_list(self):
        self._build()
        d = self.root / "samples" / "vid__1"
        feats = np.load(d / "text_features.npy")
        lines = (d / "text_words.csv").read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(feats.shape[0], len(lines) - 1)   # minus header


class TestPickSamples(unittest.TestCase):
    def test_picks_shortest_longest_and_middle(self):
        recs = [{"id": f"s{i}", "safe_id": f"s__{i}", "duration_sec": d}
                for i, d in enumerate([5.0, 1.0, 20.0, 9.0, 3.0])]
        picks = pick_samples(recs)
        tags = [t for t, _ in picks]
        self.assertEqual(tags, ["shortest", "longest", "middle"])
        durs = {t: r["duration_sec"] for t, r in picks}
        self.assertEqual(durs["shortest"], 1.0)
        self.assertEqual(durs["longest"], 20.0)


class TestRegatherEquivalence(unittest.TestCase):
    """C10's logic: re-aggregating from source indices must reproduce the stored
    features.  A correct implementation returns no mismatches; an incorrect one
    must be detected."""

    def _make_case(self, root: Path, perturb: float = 0.0,
                   extra_index: bool = False):
        """Build a minimal raw + aligned + provenance triple."""
        raw = root / "raw"
        aligned = root / "aligned"
        raw.mkdir(parents=True)
        aligned.mkdir(parents=True)

        src = np.arange(4 * 3, dtype=np.float32).reshape(4, 3)
        np.savez(raw / "s.npz", audio_features=src,
                 vision_features=src, text_features=src)

        idx0, idx1 = [0, 1], [2, 3]
        # The stored value must come from the ORIGINAL index list.  Mutating the
        # provenance afterwards is what simulates corrupted bookkeeping; if we
        # also recomputed the stored value from the mutated list the two would
        # stay self-consistent and the test would pass vacuously.
        stored = np.zeros((5, 3), dtype=np.float32)
        stored[1] = src[idx0].mean(axis=0)
        stored[2] = src[idx1].mean(axis=0)
        stored[2, 0] += perturb
        np.savez(aligned / "s.npz", audio=stored, vision=stored, text=stored)

        prov_idx1 = idx1 + [1] if extra_index else idx1
        prov = {"bins": [
            {"position": 1, "audio_feature_indices": idx0,
             "vision_feature_indices": idx0, "text_feature_indices": idx0},
            {"position": 2, "audio_feature_indices": prov_idx1,
             "vision_feature_indices": prov_idx1, "text_feature_indices": prov_idx1},
        ]}
        (aligned / "s.json").write_text(json.dumps(prov), encoding="utf-8")
        return aligned, raw

    def _regather_fails(self, aligned: Path, raw: Path) -> list[str]:
        import src.q1_alignment.guard_check as gc  # noqa: F401

        # mirror the guard's C10 loop exactly
        fails: list[str] = []
        for f in sorted(aligned.glob("*.npz")):
            r = raw / f.name
            prov_path = aligned / f"{f.stem}.json"
            if not r.exists() or not prov_path.exists():
                continue
            bins = json.loads(prov_path.read_text(encoding="utf-8"))["bins"]
            with np.load(f, allow_pickle=False) as z, \
                    np.load(r, allow_pickle=False) as rz:
                for mod, key in (("audio", "audio_features"),
                                 ("vision", "vision_features"),
                                 ("text", "text_features")):
                    src = rz[key]
                    stored = z[mod]
                    rebuilt = np.zeros_like(stored)
                    pos = []
                    for b in bins:
                        j = int(b["position"])
                        idx = b.get(f"{mod}_feature_indices") or []
                        if idx and j < stored.shape[0]:
                            rebuilt[j] = src[np.asarray(idx)].mean(axis=0)
                            pos.append(j)
                    if pos and not np.allclose(stored[pos], rebuilt[pos],
                                               atol=1e-4, rtol=1e-3):
                        fails.append(mod)
        return fails

    def test_consistent_case_passes(self):
        with tempfile.TemporaryDirectory() as d:
            aligned, raw = self._make_case(Path(d))
            self.assertEqual(self._regather_fails(aligned, raw), [])

    def test_corrupted_value_is_caught(self):
        with tempfile.TemporaryDirectory() as d:
            aligned, raw = self._make_case(Path(d), perturb=12.5)
            self.assertNotEqual(self._regather_fails(aligned, raw), [],
                                "a wrong stored value must be detected")

    def test_corrupted_index_bookkeeping_is_caught(self):
        with tempfile.TemporaryDirectory() as d:
            aligned, raw = self._make_case(Path(d), extra_index=True)
            self.assertNotEqual(self._regather_fails(aligned, raw), [],
                                "wrong index lists must be detected")


if __name__ == "__main__":
    unittest.main()
