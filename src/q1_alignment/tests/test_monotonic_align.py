"""Regression tests for the constraint-based monotonic aligner.

These encode the defect found in the first implementation: an alignment that
was monotone, finite and inside the clip while covering only a fraction of it
and implying an impossible speaking rate.
"""

from __future__ import annotations

import unittest

import numpy as np

from q1_alignment.monotonic_align import (
    MAX_RATE,
    _syllable_weight,
    align_words_monotonic,
)


class SyllableWeightTests(unittest.TestCase):
    def test_monosyllable_beats_multisyllable(self):
        self.assertLess(_syllable_weight("cat"), _syllable_weight("assistance"))

    def test_function_words_are_damped(self):
        self.assertLess(_syllable_weight("the"), _syllable_weight("cat"))

    def test_never_zero(self):
        for word in ("a", "I", "", "'''"):
            self.assertGreater(_syllable_weight(word), 0.0)


class CoverageTests(unittest.TestCase):
    """The core regression: full coverage is guaranteed, not hoped for."""

    def test_full_coverage_simple(self):
        words = ["it", "comes", "with", "the", "assistance"]
        entries, quality = align_words_monotonic(words, 29.288)
        self.assertAlmostEqual(entries[0]["start"], 0.0, places=6)
        self.assertAlmostEqual(entries[-1]["end"], 29.288, places=6)
        self.assertAlmostEqual(quality.coverage, 1.0, places=6)

    def test_full_coverage_many_spans_scenario(self):
        # the exact failure shape: long clip, few words, previously 19.4 w/s
        entries, quality = align_words_monotonic(["a", "b", "c", "d", "e"], 3.416)
        span = entries[-1]["end"] - entries[0]["start"]
        self.assertAlmostEqual(span, 3.416, places=6)
        self.assertGreater(quality.coverage, 0.999)

    def test_monotonic_and_non_overlapping(self):
        entries, _ = align_words_monotonic(["one", "two", "three", "four"], 5.0)
        for previous, current in zip(entries, entries[1:]):
            self.assertLessEqual(previous["end"], current["start"] + 1e-9)

    def test_speaking_rate_is_plausible(self):
        words = ["we", "are", "a", "huge", "user", "of", "adhesives"] * 4
        entries, quality = align_words_monotonic(words, 20.0)
        self.assertLessEqual(quality.max_rate, MAX_RATE + 1e-6)
        self.assertEqual(quality.rate_outliers, 0)

    def test_long_word_gets_more_time_than_short(self):
        entries, _ = align_words_monotonic(["a", "extraordinary"], 10.0)
        short = entries[0]["end"] - entries[0]["start"]
        long = entries[1]["end"] - entries[1]["start"]
        self.assertGreater(long, short)

    def test_single_word_covers_whole_clip(self):
        entries, quality = align_words_monotonic(["hello"], 4.0)
        self.assertAlmostEqual(entries[0]["start"], 0.0, places=6)
        self.assertAlmostEqual(entries[0]["end"], 4.0, places=6)
        self.assertAlmostEqual(quality.coverage, 1.0, places=6)

    def test_many_words_in_short_clip_stays_feasible(self):
        words = [f"word{i}" for i in range(60)]
        entries, quality = align_words_monotonic(words, 3.0)
        self.assertAlmostEqual(quality.coverage, 1.0, places=6)
        self.assertGreater(quality.min_span, 0.0)
        for previous, current in zip(entries, entries[1:]):
            self.assertLessEqual(previous["end"], current["start"] + 1e-9)

    def test_rejects_empty_or_bad_duration(self):
        with self.assertRaises(ValueError):
            align_words_monotonic([], 5.0)
        with self.assertRaises(ValueError):
            align_words_monotonic(["a"], 0.0)


class PauseSnappingTests(unittest.TestCase):
    def test_boundaries_move_towards_energy_valleys(self):
        # a synthetic silence in the middle should attract the boundary
        centers = np.arange(0.0, 4.0, 0.01)
        rms = np.full_like(centers, 0.5)
        rms[(centers > 1.9) & (centers < 2.1)] = 0.001
        entries, quality = align_words_monotonic(
            ["alpha", "beta"], 4.0, rms=rms, energy_centers=centers
        )
        self.assertAlmostEqual(entries[0]["end"], 2.0, delta=0.15)
        self.assertGreater(quality.pause_snapped, 0)

    def test_snapping_preserves_feasibility(self):
        centers = np.arange(0.0, 2.0, 0.01)
        rms = np.full_like(centers, 0.4)
        rms[(centers > 0.9) & (centers < 1.1)] = 0.0
        entries, quality = align_words_monotonic(
            ["a", "b", "c"], 2.0, rms=rms, energy_centers=centers
        )
        self.assertAlmostEqual(quality.coverage, 1.0, places=6)
        for previous, current in zip(entries, entries[1:]):
            self.assertLessEqual(previous["end"], current["start"] + 1e-9)


if __name__ == "__main__":
    unittest.main()
