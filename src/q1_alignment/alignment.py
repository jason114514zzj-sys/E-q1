from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class PoolResult:
    values: np.ndarray
    mask: np.ndarray
    source_indices: list[list[int]]


def _centers(times: np.ndarray) -> np.ndarray:
    times = np.asarray(times, dtype=np.float64)
    if times.ndim == 1:
        return times
    if times.ndim == 2 and times.shape[1] == 2:
        if np.any(times[:, 0] > times[:, 1]):
            raise ValueError("time interval start exceeds end")
        return times.mean(axis=1)
    raise ValueError(f"times must have shape [T] or [T,2], got {times.shape}")


def temporal_pool(
    features: np.ndarray,
    times: np.ndarray,
    intervals: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> PoolResult:
    features = np.asarray(features)
    if features.ndim != 2:
        raise ValueError(f"features must have shape [T,D], got {features.shape}")
    if not np.issubdtype(features.dtype, np.number):
        raise ValueError("features must be numeric")
    if not np.isfinite(features).all():
        raise ValueError("features contain NaN or Inf")

    centers = _centers(times)
    if len(centers) != features.shape[0]:
        raise ValueError("feature/time length mismatch")
    if np.any(np.diff(centers) < 0):
        raise ValueError("feature times are not monotonic")

    intervals = np.asarray(intervals, dtype=np.float64)
    if intervals.ndim != 2 or intervals.shape[1] != 2:
        raise ValueError("intervals must have shape [L,2]")
    if np.any(intervals[:, 0] >= intervals[:, 1]):
        raise ValueError("alignment interval start must be smaller than end")
    if len(intervals) > 1 and np.any(intervals[1:, 0] < intervals[:-1, 1]):
        raise ValueError("alignment intervals overlap")

    if valid_mask is None:
        valid = np.ones(features.shape[0], dtype=bool)
    else:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.shape != (features.shape[0],):
            raise ValueError("valid mask length mismatch")

    output = np.zeros((len(intervals), features.shape[1]), dtype=np.float32)
    mask = np.zeros(len(intervals), dtype=np.uint8)
    sources: list[list[int]] = []
    for position, (start, end) in enumerate(intervals):
        selected = np.flatnonzero(valid & (centers >= start) & (centers < end))
        if len(selected):
            output[position] = features[selected].mean(axis=0, dtype=np.float64)
            mask[position] = 1
        sources.append(selected.astype(int).tolist())
    return PoolResult(output, mask, sources)


def build_word_groups(word_count: int, content_slots: int = 48) -> list[np.ndarray]:
    if word_count <= 0:
        raise ValueError("word_count must be positive")
    if content_slots <= 0:
        raise ValueError("content_slots must be positive")
    indices = np.arange(word_count, dtype=np.int64)
    if word_count <= content_slots:
        return [indices[i : i + 1] for i in range(word_count)]
    return [group for group in np.array_split(indices, content_slots) if len(group)]


def _group_text_features(
    features: np.ndarray,
    word_count: int,
    groups: Sequence[np.ndarray],
    word_index: np.ndarray | None,
    valid_mask: np.ndarray | None,
) -> PoolResult:
    features = np.asarray(features)
    if features.ndim != 2 or not np.issubdtype(features.dtype, np.number):
        raise ValueError("text_features must be a numeric [T,D] array")
    if not np.isfinite(features).all():
        raise ValueError("text_features contain NaN or Inf")
    if valid_mask is None:
        valid = np.ones(features.shape[0], dtype=bool)
    else:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.shape != (features.shape[0],):
            raise ValueError("text_valid length mismatch")

    if word_index is None:
        if features.shape[0] != word_count:
            raise ValueError(
                "without text_word_index, text_features rows must equal timeline word count"
            )
        mapping = np.arange(word_count, dtype=np.int64)
    else:
        mapping = np.asarray(word_index, dtype=np.int64)
        if mapping.shape != (features.shape[0],):
            raise ValueError("text_word_index length mismatch")
        if np.any(mapping >= word_count):
            raise ValueError("text_word_index references a non-existent timeline word")

    output = np.zeros((len(groups), features.shape[1]), dtype=np.float32)
    mask = np.zeros(len(groups), dtype=np.uint8)
    sources: list[list[int]] = []
    for position, group in enumerate(groups):
        selected = np.flatnonzero(valid & np.isin(mapping, group))
        if len(selected):
            output[position] = features[selected].mean(axis=0, dtype=np.float64)
            mask[position] = 1
        sources.append(selected.astype(int).tolist())
    return PoolResult(output, mask, sources)


def align_sample(
    sample: dict[str, Any],
    timeline: dict[str, Any],
    bundle: dict[str, np.ndarray],
    max_positions: int = 50,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if sample["id"] != timeline["id"]:
        raise ValueError("manifest/timeline id mismatch")
    if max_positions < 3:
        raise ValueError("max_positions must leave room for CLS, content and SEP")
    words = timeline["words"]
    word_count = len(words)
    groups = build_word_groups(word_count, max_positions - 2)

    # A clip whose audio track is digitally silent has **no measurable word
    # boundary**: the timeline carries the -1.0 sentinel everywhere (R8).  That
    # must be a *supported* path, not an exception -- otherwise the whole
    # 100-sample run dies on a property of the source material.
    #
    # What still works: text grouping is driven by word *indices*, not by time,
    # so the text channel is real.  What cannot work: time pooling, so audio and
    # vision are unusable at every position and say so through their masks.
    # The interval arrays keep the sentinel at every content position, which is
    # exactly the R6 contract (sentinel <-> no time evidence) rather than a
    # special case bolted on top of it.
    audio_absent = not bool(
        (timeline.get("diagnostics") or {}).get("audio_present", True)
    )

    if audio_absent:
        intervals = np.full((len(groups), 2), -1.0, dtype=np.float64)
        text = _group_text_features(
            bundle["text_features"],
            word_count,
            groups,
            bundle.get("text_word_index"),
            bundle.get("text_valid"),
        )
        audio = PoolResult(
            np.zeros((len(groups), bundle["audio_features"].shape[1]), dtype=np.float32),
            np.zeros(len(groups), dtype=np.uint8),
            [[] for _ in groups],
        )
        vision = PoolResult(
            np.zeros((len(groups), bundle["vision_features"].shape[1]), dtype=np.float32),
            np.zeros(len(groups), dtype=np.uint8),
            [[] for _ in groups],
        )
        vision_valid_ratio = np.zeros(len(groups), dtype=np.float32)
    else:
        intervals = np.array(
            [
                [float(words[int(group[0])]["start"]), float(words[int(group[-1])]["end"])]
                for group in groups
            ],
            dtype=np.float64,
        )

        audio = temporal_pool(
            bundle["audio_features"],
            bundle["audio_times"],
            intervals,
            bundle.get("audio_valid"),
        )
        vision = temporal_pool(
            bundle["vision_features"],
            bundle["vision_times"],
            intervals,
            bundle.get("vision_valid"),
        )
        text = _group_text_features(
            bundle["text_features"],
            word_count,
            groups,
            bundle.get("text_word_index"),
            bundle.get("text_valid"),
        )

        # ratio of source-level valid vision frames inside each interval; the
        # collaboration document requires face-detection failures to stay visible
        # after alignment instead of collapsing into numeric zeros.
        vision_valid_ratio = np.zeros(len(intervals), dtype=np.float32)
        if "vision_valid" in bundle:
            raw_valid = np.asarray(bundle["vision_valid"], dtype=bool)
            vision_times = np.asarray(bundle["vision_times"], dtype=np.float64)
            if raw_valid.shape[0] == vision_times.shape[0]:
                centers = _centers(vision_times)
                for index, (start, end) in enumerate(intervals):
                    inside = (centers >= start) & (centers < end)
                    total = int(np.count_nonzero(inside))
                    if total:
                        vision_valid_ratio[index] = float(
                            np.count_nonzero(raw_valid[inside])
                        ) / total

    content_count = len(groups)
    sep_position = 1 + content_count
    effective_length = sep_position + 1

    # ---- text word coverage ------------------------------------------------
    # ``align_sample`` maps the delivered per-word text features onto OUR word
    # indices.  If the delivery references fewer words than the timeline has
    # (e.g. a different tokenisation), the tail words used to lose their text
    # features **silently**: their mask simply came out 0 and no structural
    # check noticed.  ``handoff_selfcheck``/``validate-handoff`` reject the
    # common case, but this stays as a visible second line of defence because
    # silence here is indistinguishable from "the word really has no text".
    text_coverage = {
        "covered": word_count,
        "total": word_count,
        "first_uncovered": None,
    }
    word_index = bundle.get("text_word_index")
    if word_index is not None and word_count:
        mapping = np.asarray(word_index, dtype=np.int64).reshape(-1)
        mapping = mapping[(mapping >= 0) & (mapping < word_count)]
        covered = np.zeros(word_count, dtype=bool)
        if mapping.size:
            covered[np.unique(mapping)] = True
        if not covered.all():
            text_coverage = {
                "covered": int(covered.sum()),
                "total": int(word_count),
                "first_uncovered": int(np.flatnonzero(~covered)[0]),
            }

    def padded(values: np.ndarray) -> np.ndarray:
        result = np.zeros((max_positions, values.shape[1]), dtype=np.float32)
        result[1 : 1 + content_count] = values
        return result

    def padded_mask(mask: np.ndarray, include_special: bool = False) -> np.ndarray:
        result = np.zeros(max_positions, dtype=np.uint8)
        result[1 : 1 + content_count] = mask
        if include_special:
            result[0] = 1
            result[sep_position] = 1
        return result

    interval_start = np.full(max_positions, -1.0, dtype=np.float32)
    interval_end = np.full(max_positions, -1.0, dtype=np.float32)
    if not audio_absent:
        interval_start[1 : 1 + content_count] = intervals[:, 0]
        interval_end[1 : 1 + content_count] = intervals[:, 1]

    arrays = {
        "text": padded(text.values),
        "audio": padded(audio.values),
        "vision": padded(vision.values),
        # sequence_mask: real sequence positions (CLS + content + SEP)
        # text/audio/vision_mask: whether that modality has data there
        "sequence_mask": padded_mask(np.ones(content_count, dtype=np.uint8), True),
        "attention_mask": padded_mask(np.ones(content_count, dtype=np.uint8), True),
        "text_mask": padded_mask(text.mask),
        "audio_mask": padded_mask(audio.mask),
        "vision_mask": padded_mask(vision.mask),
        "vision_valid_ratio": np.concatenate(
            [np.zeros(1, dtype=np.float32), vision_valid_ratio,
             np.zeros(max_positions - 1 - content_count, dtype=np.float32)]
        ),
        "interval_start": interval_start,
        "interval_end": interval_end,
        "effective_length": np.array(effective_length, dtype=np.int64),
        "original_word_count": np.array(word_count, dtype=np.int64),
        "original_audio_length": np.array(bundle["audio_features"].shape[0], dtype=np.int64),
        "original_vision_length": np.array(bundle["vision_features"].shape[0], dtype=np.int64),
    }

    bins = []
    for index, group in enumerate(groups):
        start_word = int(group[0])
        end_word = int(group[-1])
        bins.append(
            {
                "position": index + 1,
                "word_start_index": start_word,
                "word_end_index": end_word,
                "text": " ".join(str(words[i]["word"]) for i in group),
                "start": float(intervals[index, 0]),
                "end": float(intervals[index, 1]),
                "text_feature_indices": text.source_indices[index],
                "audio_feature_indices": audio.source_indices[index],
                "vision_feature_indices": vision.source_indices[index],
            }
        )
    provenance = {
        "id": sample["id"],
        "safe_id": sample["safe_id"],
        "timeline_method": timeline.get("method"),
        "alignment_method": timeline.get("alignment_method") or timeline.get("method"),
        "timeline_final_usable": bool(timeline.get("final_usable", False)),
        # True when the source clip's audio track carried no signal at all.  The
        # text channel is still valid; audio and vision are unusable everywhere
        # and every content position keeps the -1.0 sentinel.
        "audio_absent": audio_absent,
        # How many of OUR transcript words were referenced by the delivered
        # per-word text features.  ``first_uncovered`` is None when complete.
        "text_word_coverage": text_coverage,
        "max_positions": max_positions,
        "content_positions": content_count,
        "effective_length": effective_length,
        "long_text_grouped": word_count > max_positions - 2,
        "bins": bins,
    }
    return arrays, provenance
