"""Tests for the mask-aware text pooling module.

These tests encode the verified dataset facts so that a regression (e.g.
someone reintroducing naive averaging) fails the suite instead of silently
degrading the features.
"""

from __future__ import annotations

import unittest
import warnings

import numpy as np

from src.q1_alignment.text_pooling import (
    MASK_CHANNEL,
    check_text_has_no_padding_zeros,
    extract_attention_mask,
    masked_mean_pool,
    pool_text_sequence,
)


def _make_bert(n: int, valid: list[int], t: int = 5) -> np.ndarray:
    """Build a (n, 3, t) text_bert where channel 1 is tail padding."""
    bert = np.zeros((n, 3, t), dtype=np.int64)
    for i, k in enumerate(valid):
        bert[i, 0, :k] = np.arange(100, 100 + k)  # word ids
        bert[i, 1, :k] = 1                        # attention mask
    return bert


class TestExtractAttentionMask(unittest.TestCase):
    def test_finds_the_mask_channel(self):
        bert = _make_bert(3, [5, 3, 1])
        mask = extract_attention_mask(bert)
        np.testing.assert_array_equal(mask.sum(axis=1), [5, 3, 1])

    def test_mask_channel_constant_is_1(self):
        self.assertEqual(MASK_CHANNEL, 1)

    def test_raises_when_no_mask_channel(self):
        bad = np.zeros((2, 3, 4), dtype=np.int64)
        bad[:, 0, :] = 7
        bad[:, 1, :] = 9
        bad[:, 2, :] = 11
        with self.assertRaises(ValueError):
            extract_attention_mask(bad)

    def test_raises_on_wrong_rank(self):
        with self.assertRaises(ValueError):
            extract_attention_mask(np.zeros((3, 5)))


class TestMaskedMeanPool(unittest.TestCase):
    def test_ignores_padded_positions(self):
        values = np.array([[[1.0, 1.0], [3.0, 3.0], [99.0, 99.0]]])
        mask = np.array([[1.0, 1.0, 0.0]])
        out = masked_mean_pool(values, mask)
        np.testing.assert_allclose(out, [[2.0, 2.0]])

    def test_all_valid_matches_plain_mean(self):
        rng = np.random.default_rng(0)
        v = rng.normal(size=(4, 6, 3))
        m = np.ones((4, 6))
        np.testing.assert_allclose(masked_mean_pool(v, m), v.mean(axis=1))

    def test_no_valid_steps_yields_zeros_not_nan(self):
        v = np.ones((1, 4, 2))
        m = np.zeros((1, 4))
        out = masked_mean_pool(v, m)
        self.assertFalse(np.isnan(out).any())
        np.testing.assert_allclose(out, 0.0)

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            masked_mean_pool(np.ones((2, 5, 3)), np.ones((2, 4)))

    def test_matches_hand_computed_mean(self):
        rng = np.random.default_rng(7)
        v = rng.normal(size=(3, 8, 4))
        m = np.zeros((3, 8))
        m[0, :8] = 1
        m[1, :3] = 1
        m[2, :1] = 1
        out = masked_mean_pool(v, m)
        for i in range(3):
            k = int(m[i].sum())
            np.testing.assert_allclose(out[i], v[i, :k].mean(axis=0))


class TestPoolTextSequence(unittest.TestCase):
    def test_masked_route_uses_mask(self):
        text = np.zeros((2, 4, 3), dtype=np.float32)
        text[0, :2] = 5.0
        text[0, 2:] = -100.0   # non-zero padding, the real dataset's behaviour
        text[1, :4] = 1.0
        bert = _make_bert(2, [2, 4], t=4)
        out = pool_text_sequence(text, bert, method="masked")
        np.testing.assert_allclose(out[0], [5.0, 5.0, 5.0])
        np.testing.assert_allclose(out[1], [1.0, 1.0, 1.0])

    def test_naive_route_is_contaminated(self):
        text = np.zeros((1, 4, 2), dtype=np.float32)
        text[0, :2] = 5.0
        text[0, 2:] = -100.0
        bert = _make_bert(1, [2], t=4)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            naive = pool_text_sequence(text, bert, method="naive")
        masked = pool_text_sequence(text, bert, method="masked")
        self.assertGreater(np.abs(naive - masked).max(), 40.0)

    def test_naive_emits_warning(self):
        text = np.ones((1, 4, 2), dtype=np.float32)
        bert = _make_bert(1, [4], t=4)
        with self.assertWarns(RuntimeWarning):
            pool_text_sequence(text, bert, method="naive")

    def test_missing_text_raises_for_masked_route(self):
        # attachment 3: text_bert present, text absent
        bert = _make_bert(1, [3], t=5)
        with self.assertRaises(ValueError) as ctx:
            pool_text_sequence(None, bert, method="masked")
        self.assertIn("text_bert", str(ctx.exception))

    def test_missing_bert_raises(self):
        with self.assertRaises(ValueError):
            pool_text_sequence(np.ones((1, 5, 2)), None, method="masked")

    def test_unknown_method_raises(self):
        with self.assertRaises(ValueError):
            pool_text_sequence(np.ones((1, 5, 2)), _make_bert(1, [5]), method="magic")


class TestPaddingZeroTrap(unittest.TestCase):
    def test_detects_padding_zero_trap(self):
        # mimic the real dataset: no zeros anywhere
        rng = np.random.default_rng(3)
        text = rng.normal(size=(10, 6, 4)).astype(np.float32)
        report = check_text_has_no_padding_zeros(text)
        self.assertTrue(report["trap_present"])
        self.assertEqual(report["all_zero_step_fraction"], 0.0)

    def test_reports_false_when_zero_padding_present(self):
        text = np.zeros((10, 6, 4), dtype=np.float32)
        text[:, :3] = 2.0
        report = check_text_has_no_padding_zeros(text)
        self.assertFalse(report["trap_present"])


if __name__ == "__main__":
    unittest.main()
