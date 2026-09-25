"""Tests that exercise the CLI commands end to end.

The ``validate-handoff`` command once failed with ``NameError: Path`` because a
module-level import was missing; unit tests on the adapter alone could not catch
that, so these tests invoke the commands through ``main``.
"""

from __future__ import annotations

import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from q1_alignment.cli import main
from q1_alignment.common import write_jsonl


def make_handoff(root: Path, sample_id: str, duration: float = 2.0) -> Path:
    """A handoff that satisfies the reconciled contract in full.

    text_words.csv is required (the paper needs it to trace words back to the
    transcript), so a fixture omitting it no longer represents a valid delivery.
    """
    directory = root / "samples" / sample_id
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "text_features.npy", np.random.randn(4, 768).astype(np.float32))
    np.save(directory / "audio_features.npy", np.random.randn(50, 74).astype(np.float32))
    np.save(directory / "audio_intervals.npy",
            np.column_stack([np.linspace(0, duration, 50, endpoint=False),
                             np.linspace(duration / 50, duration, 50)]))
    np.save(directory / "vision_features.npy", np.random.randn(30, 35).astype(np.float32))
    np.save(directory / "vision_timestamps.npy", np.linspace(0, duration, 30, endpoint=False))
    np.save(directory / "vision_valid.npy", np.ones(30, dtype=np.uint8))
    with (directory / "text_words.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["word_idx", "word"])
        for index in range(4):
            writer.writerow([index, f"w{index}"])
    return root


def run_cli(argv: list[str]) -> tuple[int, str]:
    """Invoke the CLI, capturing stdout. Returns (exit_code, output)."""

    buffer = io.StringIO()
    code = 0
    with contextlib.redirect_stdout(buffer):
        try:
            main.__wrapped__(argv) if hasattr(main, "__wrapped__") else _call(argv)
        except SystemExit as exc:
            code = int(exc.code or 0)
    return code, buffer.getvalue()


def _call(argv: list[str]) -> None:
    import sys

    original = sys.argv
    sys.argv = ["q1_alignment"] + argv
    try:
        main()
    finally:
        sys.argv = original


class CliValidateHandoffTests(unittest.TestCase):
    def test_validate_handoff_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_handoff(root / "handoff", "video$_$1")
            manifest = root / "manifest.jsonl"
            write_jsonl(manifest, [{"id": "video$_$1", "safe_id": "video__1",
                                    "duration_sec": 2.0}])
            report = root / "validation.json"

            code, output = run_cli([
                "validate-handoff",
                "--handoff", str(root / "handoff"),
                "--manifest", str(manifest),
                "--report", str(report),
            ])
            self.assertEqual(code, 0, msg=output)
            self.assertTrue(report.exists())
            payload = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(payload["checked"], 1)
            self.assertEqual(payload["failed"], 0)

    def test_import_handoff_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_handoff(root / "handoff", "video$_$1")
            manifest = root / "manifest.jsonl"
            write_jsonl(manifest, [{"id": "video$_$1", "safe_id": "video__1",
                                    "duration_sec": 2.0}])
            out = root / "raw"

            code, output = run_cli([
                "import-handoff",
                "--handoff", str(root / "handoff"),
                "--manifest", str(manifest),
                "--output-dir", str(out),
            ])
            self.assertEqual(code, 0, msg=output)
            self.assertTrue((out / "video__1.npz").exists())


if __name__ == "__main__":
    unittest.main()
