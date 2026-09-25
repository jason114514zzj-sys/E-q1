"""Tests for per-dimension feature scaling.

The regression these tests defend against: someone switching back to a single
global mean/std, which leaves audio dim 0 (std 3.33) roughly 394x larger than
the median of dims 1..73 (std 0.0085).
"""

from __future__ import annotations

import unittest

import numpy as np

from src.q1_alignment.feature_scaling import (
    MODALITY_DIMS,
    audio_valid_mask,
    fit_scaler,
    split_scale_report,
    vision_valid_mask,
)


class TestValidMask(unittest.TestCase):
    def test_all_zero_step_is_missing(self):
        x = np.zeros((1, 3, 4))
        x[0, 1] = 1.0
        np.testing.assert_array_equal(audio_valid_mask(x), [[False, True, False]])

    def test_negative_signal_counts_as_present(self):
        x = np.zeros((1, 2, 3))
        x[0, 0] = -5.0
        np.testing.assert_array_equal(audio_valid_mask(x), [[True, False]])

    def test_partial_zero_row_counts_as_present(self):
        # a step is missing only when EVERY dimension is zero
        x = np.zeros((1, 2, 3))
        x[0, 0, 1] = 2.0
        np.testing.assert_array_equal(audio_valid_mask(x), [[True, False]])

    def test_vision_mask_matches_audio_convention(self):
        x = np.zeros((2, 2, 3))
        x[1, 1] = 1.0
        np.testing.assert_array_equal(vision_valid_mask(x), audio_valid_mask(x))

    def test_rejects_wrong_rank(self):
        with self.assertRaises(ValueError):
            audio_valid_mask(np.zeros((3, 4)))


class TestFitScaler(unittest.TestCase):
    def test_padding_steps_excluded_from_statistics(self):
        x = np.zeros((1, 4, 2))
        x[0, 0] = [10.0, 20.0]
        x[0, 1] = [30.0, 40.0]
        # steps 2,3 are padding and must not drag the mean down
        sc = fit_scaler(x, "audio")
        np.testing.assert_allclose(sc.mean, [20.0, 30.0])
        self.assertEqual(sc.n_steps_used, 2)

    def test_per_dimension_is_not_global(self):
        x = np.zeros((1, 4, 2))
        x[0, 0] = [0.0, 1000.0]
        x[0, 1] = [10.0, 2000.0]
        x[0, 2] = [20.0, 3000.0]
        x[0, 3] = [30.0, 4000.0]
        sc = fit_scaler(x, "audio")
        z = sc.transform(x)
        present = audio_valid_mask(x)
        # both dimensions end up unit variance -> neither dominates
        for dim in range(2):
            self.assertAlmostEqual(z[present][:, dim].std(), 1.0, places=5)

    def test_degenerate_dimension_is_clamped_not_nan(self):
        x = np.zeros((1, 3, 2))
        x[0, :, 0] = 5.0       # constant -> std 0
        x[0, :, 1] = [1.0, 2.0, 3.0]
        sc = fit_scaler(x, "audio")
        z = sc.transform(x)
        self.assertTrue(np.isfinite(z).all())

    def test_padding_stays_exactly_zero_after_transform(self):
        x = np.zeros((1, 4, 2))
        x[0, 0] = [1.0, 1.0]
        sc = fit_scaler(x, "audio")
        z = sc.transform(x)
        np.testing.assert_array_equal(z[0, 1:], 0.0)

    def test_roundtrip_inverse(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(5, 6, 3)) + 10.0
        sc = fit_scaler(x, "audio")
        np.testing.assert_allclose(sc.inverse_transform(sc.transform(x)), x, rtol=1e-4)

    def test_wrong_dim_raises(self):
        sc = fit_scaler(np.ones((1, 3, 4)), "audio")
        with self.assertRaises(ValueError):
            sc.transform(np.ones((1, 3, 5)))

    def test_all_padding_raises(self):
        with self.assertRaises(ValueError):
            fit_scaler(np.zeros((2, 3, 4)), "audio")

    def test_rejects_wrong_rank(self):
        with self.assertRaises(ValueError):
            fit_scaler(np.zeros((3, 4)), "audio")

    def test_audio_note_mentions_per_dimension_requirement(self):
        x = np.zeros((1, 3, 4))
        x[0] = 1.0
        sc = fit_scaler(x, "audio")
        self.assertTrue(any("per-dimension" in n for n in sc.notes))

    def test_text_can_skip_zero_convention(self):
        # text has no zeros; padding must not be inferred from zeros
        x = np.full((1, 3, 4), 2.0)
        sc = fit_scaler(x, "text", missing_is_zero=False)
        self.assertEqual(sc.n_steps_used, 3)
        self.assertFalse(sc.missing_is_zero)

    def test_dim_mismatch_recorded_in_notes(self):
        x = np.zeros((1, 3, 7))
        x[0] = 1.0
        sc = fit_scaler(x, "audio")   # audio expects 74
        self.assertTrue(any("expected 74" in n for n in sc.notes))

    def test_modality_dims_constant(self):
        self.assertEqual(MODALITY_DIMS, {"audio": 74, "vision": 35, "text": 768})

    def test_to_dict_is_serialisable(self):
        import json
        x = np.zeros((1, 3, 4))
        x[0] = 1.0
        sc = fit_scaler(x, "audio")
        json.dumps(sc.to_dict())   # must not raise


class TestSplitScaleReport(unittest.TestCase):
    def test_reports_dim0_dominance(self):
        splits = {}
        for name in ("train", "valid"):
            x = np.zeros((2, 3, 4))
            x[:, 0, 0] = 500.0     # dim0 huge
            x[:, 1, :] = 2.0       # other dims small
            splits[name] = x
        rep = split_scale_report(splits, "audio")
        self.assertGreater(rep["train"]["scale_ratio"], 10.0)
        self.assertEqual(rep["train"]["dim0_max"], 500.0)


if __name__ == "__main__":
    unittest.main()
