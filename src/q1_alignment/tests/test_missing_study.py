"""Verify the missing-modality masks are truly contiguous, as the problem requires.

题目段24: "模态局部缺失指文本、语音或视觉中的部分连续时间段不可用"

A mask that removes scattered frames would satisfy "30% missing" but violate the
definition.  These tests assert contiguity explicitly, plus the other design
invariants the study relies on:

* the removed span is one consecutive run;
* the span lies entirely inside the previously-valid region;
* nothing outside the span changes;
* text removal also clears the attention mask (otherwise the model is told the
  text is present when it is not);
* removal never changes sequence length (the problem requires keeping samples).
"""

from __future__ import annotations

import unittest

import numpy as np

from src.q1_alignment.missing_study import (
    MODALITIES,
    MissingSpec,
    _valid_mask,
    apply_missing,
    build_conditions,
)


def make_bundle(n_valid: int = 20, total: int = 50, seed: int = 0) -> dict:
    """A bundle whose first n_valid steps are valid and the rest is padding."""
    rng = np.random.default_rng(seed)
    text = rng.normal(size=(total, 8)).astype(np.float32)
    audio = rng.normal(size=(total, 4)).astype(np.float32)
    vision = rng.normal(size=(total, 3)).astype(np.float32)
    # padding: zero everything past n_valid
    text[n_valid:] = 0.0
    audio[n_valid:] = 0.0
    vision[n_valid:] = 0.0
    bert = np.zeros((3, total), dtype=np.int64)
    bert[0, :n_valid] = np.arange(1000, 1000 + n_valid)
    bert[1, :n_valid] = 1
    return {"text": text, "audio": audio, "vision": vision, "text_bert": bert}


def removed_positions(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """Indices that were valid before and are not after."""
    return np.flatnonzero(before & ~after)


class TestContiguity(unittest.TestCase):
    def test_removed_span_is_one_contiguous_run(self):
        for ratio in (0.1, 0.3, 0.5):
            for mod in MODALITIES:
                b = make_bundle()
                spec = MissingSpec(f"t_{mod}_{ratio}", (mod,), ratio, seed=1)
                out, rec = apply_missing(b, spec, np.random.default_rng(5))
                before = _valid_mask(b[mod], mod, b.get("text_bert"))
                after = _valid_mask(out[mod], mod, out.get("text_bert"))
                gone = removed_positions(before, after)
                if gone.size == 0:
                    continue
                gaps = np.diff(gone)
                self.assertTrue(np.all(gaps == 1),
                                f"{mod}@{ratio}: removed {gone.tolist()} is not contiguous")

    def test_removed_count_matches_requested_ratio(self):
        b = make_bundle(n_valid=20)
        for ratio in (0.2, 0.4, 0.6):
            spec = MissingSpec(f"r{ratio}", ("audio",), ratio)
            out, rec = apply_missing(b, spec, np.random.default_rng(3))
            before = _valid_mask(b["audio"], "audio", None)
            after = _valid_mask(out["audio"], "audio", None)
            gone = removed_positions(before, after).size
            expected = max(1, int(round(ratio * before.sum())))
            self.assertEqual(gone, expected,
                             f"ratio {ratio}: removed {gone}, expected {expected}")

    def test_span_inside_valid_region_only(self):
        b = make_bundle(n_valid=15)
        spec = MissingSpec("inside", ("audio",), 0.5)
        out, rec = apply_missing(b, spec, np.random.default_rng(9))
        before = _valid_mask(b["audio"], "audio", None)
        gone = removed_positions(before, _valid_mask(out["audio"], "audio", None))
        self.assertTrue(np.all(before[gone]),
                        "removed a step that was already padding")

    def test_leading_and_trailing_positions(self):
        b = make_bundle(n_valid=20)
        for kind in ("leading", "trailing"):
            spec = MissingSpec(kind, ("audio",), 0.25, kind=kind)
            out, rec = apply_missing(b, spec, np.random.default_rng(1))
            before = _valid_mask(b["audio"], "audio", None)
            gone = removed_positions(before, _valid_mask(out["audio"], "audio", None))
            self.assertTrue(np.all(np.diff(gone) == 1), f"{kind}: not contiguous")
            if kind == "leading":
                self.assertEqual(gone[0], np.flatnonzero(before)[0])
            else:
                self.assertEqual(gone[-1], np.flatnonzero(before)[-1])


class TestLengthPreserved(unittest.TestCase):
    def test_sequence_length_unchanged(self):
        """The problem requires keeping samples; deletion would be wrong."""
        b = make_bundle()
        spec = MissingSpec("nodelete", ("audio", "vision"), 0.4)
        out, _ = apply_missing(b, spec, np.random.default_rng(0))
        for k in ("text", "audio", "vision"):
            self.assertEqual(out[k].shape, b[k].shape, f"{k} shape changed")
        self.assertEqual(out["text_bert"].shape, b["text_bert"].shape)

    def test_untouched_modalities_are_identical(self):
        b = make_bundle()
        spec = MissingSpec("only_audio", ("audio",), 0.3)
        out, _ = apply_missing(b, spec, np.random.default_rng(0))
        np.testing.assert_array_equal(out["vision"], b["vision"])
        np.testing.assert_array_equal(out["text"], b["text"])
        np.testing.assert_array_equal(out["text_bert"], b["text_bert"])

    def test_target_modality_only_changed_inside_the_span(self):
        b = make_bundle()
        spec = MissingSpec("span_only", ("audio",), 0.3)
        out, rec = apply_missing(b, spec, np.random.default_rng(2))
        s, e = rec["spans"]["start_pos"], rec["spans"]["end_pos"]
        changed = np.flatnonzero(np.any(out["audio"] != b["audio"], axis=1))
        self.assertTrue(np.all(changed >= s) and np.all(changed <= e),
                        f"changed outside span: {changed.tolist()}")


class TestTextAttentionCoupling(unittest.TestCase):
    def test_text_removal_clears_attention_mask(self):
        b = make_bundle()
        spec = MissingSpec("textmask", ("text",), 0.3)
        out, _ = apply_missing(b, spec, np.random.default_rng(0))
        gone = removed_positions(_valid_mask(b["text"], "text", b["text_bert"]),
                                 _valid_mask(out["text"], "text", out["text_bert"]))
        self.assertGreater(gone.size, 0)
        # every removed position must have its attention bit cleared
        self.assertTrue(np.all(out["text_bert"][1, gone] == 0),
                        "attention mask still claims the text is present")
        self.assertTrue(np.all(out["text"][gone] == 0))

    def test_non_text_removal_leaves_attention_intact(self):
        b = make_bundle()
        spec = MissingSpec("audiomask", ("audio",), 0.3)
        out, _ = apply_missing(b, spec, np.random.default_rng(0))
        np.testing.assert_array_equal(out["text_bert"], b["text_bert"])


class TestConditionGrid(unittest.TestCase):
    def test_grid_covers_the_axes_the_problem_asks_for(self):
        names = [c.name for c in build_conditions()]
        joined = " ".join(names)
        # modality type axis
        for mod in MODALITIES:
            self.assertIn(f"仅缺{mod}", joined, f"missing type axis for {mod}")
        # ratio axis
        self.assertIn("_30", joined)
        self.assertIn("_50", joined)
        # position axis
        for kind in ("leading", "trailing", "contiguous"):
            self.assertIn(kind, joined, f"missing position axis {kind}")

    def test_grid_has_a_baseline(self):
        conds = build_conditions()
        self.assertTrue(any(not c.modalities for c in conds),
                        "grid needs an unmodified baseline")

    def test_audio_vision_combination_matches_attachment3(self):
        """Attachment 3 removes audio+vision together in 29/30 samples, so the
        grid must include that exact combination to be comparable."""
        conds = build_conditions()
        combos = {tuple(sorted(c.modalities)) for c in conds}
        self.assertIn(("audio", "vision"), combos)

    def test_baseline_returns_bundle_unchanged(self):
        b = make_bundle()
        spec = MissingSpec("base", (), 0.0)
        out, rec = apply_missing(b, spec, np.random.default_rng(0))
        for k in b:
            np.testing.assert_array_equal(out[k], b[k])
        self.assertEqual(rec["spans"]["n_steps"], 0)


if __name__ == "__main__":
    unittest.main()
