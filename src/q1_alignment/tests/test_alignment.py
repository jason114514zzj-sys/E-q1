import unittest

import numpy as np

from q1_alignment.alignment import align_sample, build_word_groups, temporal_pool


class TemporalPoolTests(unittest.TestCase):
    def test_pool_and_empty_interval(self):
        features = np.array([[1.0], [3.0], [9.0]], dtype=np.float32)
        times = np.array([[0.0, 0.2], [0.2, 0.4], [0.8, 1.0]])
        intervals = np.array([[0.0, 0.5], [0.5, 0.7]])
        result = temporal_pool(features, times, intervals)
        np.testing.assert_allclose(result.values[:, 0], [2.0, 0.0])
        np.testing.assert_array_equal(result.mask, [1, 0])
        self.assertEqual(result.source_indices, [[0, 1], []])

    def test_invalid_frame_is_excluded(self):
        result = temporal_pool(
            np.array([[2.0], [8.0]]),
            np.array([0.1, 0.2]),
            np.array([[0.0, 0.3]]),
            np.array([1, 0]),
        )
        self.assertEqual(float(result.values[0, 0]), 2.0)

    def test_overlapping_intervals_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            temporal_pool(
                np.array([[1.0], [2.0]]),
                np.array([0.1, 0.2]),
                np.array([[0.0, 0.2], [0.1, 0.3]]),
            )


class GroupTests(unittest.TestCase):
    def test_long_text_is_covered_by_48_groups(self):
        groups = build_word_groups(67, 48)
        self.assertEqual(len(groups), 48)
        np.testing.assert_array_equal(np.concatenate(groups), np.arange(67))


class AlignSampleTests(unittest.TestCase):
    def test_shapes_masks_and_provenance(self):
        sample = {"id": "video$_$1", "safe_id": "video__1"}
        timeline = {
            "id": "video$_$1",
            "method": "test",
            "final_usable": True,
            "words": [
                {"word": "hello", "start": 0.0, "end": 0.5},
                {"word": "world", "start": 0.5, "end": 1.0},
            ],
        }
        bundle = {
            "text_features": np.array([[1.0, 2.0], [3.0, 4.0]]),
            "audio_features": np.array([[1.0], [2.0], [3.0], [4.0]]),
            "audio_times": np.array([0.1, 0.4, 0.6, 0.9]),
            "vision_features": np.array([[5.0], [7.0]]),
            "vision_times": np.array([0.25, 0.75]),
        }
        arrays, provenance = align_sample(sample, timeline, bundle, max_positions=50)
        self.assertEqual(arrays["text"].shape, (50, 2))
        self.assertEqual(arrays["audio"].shape, (50, 1))
        self.assertEqual(arrays["vision"].shape, (50, 1))
        np.testing.assert_array_equal(arrays["attention_mask"][:5], [1, 1, 1, 1, 0])
        np.testing.assert_allclose(arrays["audio"][1:3, 0], [1.5, 3.5])
        self.assertEqual(provenance["bins"][0]["audio_feature_indices"], [0, 1])


if __name__ == "__main__":
    unittest.main()
