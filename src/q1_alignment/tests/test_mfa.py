import json
import tempfile
import unittest
from pathlib import Path

from q1_alignment.common import write_jsonl
from q1_alignment.mfa import import_mfa_json


class MfaJsonTests(unittest.TestCase):
    def test_import_words_tier(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.jsonl"
            write_jsonl(
                manifest,
                [
                    {
                        "id": "video$_$1",
                        "safe_id": "video__1",
                        "text": "hello world",
                        "duration_sec": 1.0,
                    }
                ],
            )
            payload = {
                "start": 0,
                "end": 1.0,
                "tiers": {
                    "words": {
                        "type": "interval",
                        "entries": [[0.1, 0.4, "hello"], [0.4, 0.8, "world"]],
                    },
                    "phones": {"type": "interval", "entries": []},
                },
            }
            (root / "video__1.json").write_text(json.dumps(payload), encoding="utf-8")
            output = root / "timelines.jsonl"
            records = import_mfa_json(root, manifest, output)
            self.assertEqual(records[0]["words"][1]["word"], "world")
            self.assertTrue(records[0]["final_usable"])


if __name__ == "__main__":
    unittest.main()
