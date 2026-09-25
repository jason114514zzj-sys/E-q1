import unittest

from q1_alignment.timeline import tokenize_transcript, validate_timeline


class TimelineTests(unittest.TestCase):
    def test_tokenizer_keeps_contractions(self):
        self.assertEqual(tokenize_transcript("They've been here."), ["They've", "been", "here"])

    def test_validation(self):
        timeline = {
            "words": [
                {"word": "a", "start": 0.0, "end": 0.2},
                {"word": "b", "start": 0.2, "end": 0.5},
            ]
        }
        self.assertEqual(validate_timeline(timeline, 1.0), [])

    def test_overlap_is_reported(self):
        timeline = {
            "words": [
                {"word": "a", "start": 0.0, "end": 0.3},
                {"word": "b", "start": 0.25, "end": 0.5},
            ]
        }
        self.assertTrue(any("overlap" in issue for issue in validate_timeline(timeline, 1.0)))


if __name__ == "__main__":
    unittest.main()
