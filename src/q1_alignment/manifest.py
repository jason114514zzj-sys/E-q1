from __future__ import annotations

from pathlib import Path
from typing import Any

import openpyxl

from .common import make_sample_id, safe_id, write_jsonl
from .ffprobe_utils import FFProbeError, read_timing


def _video_metadata(path: Path, ffprobe: str | Path | None = None) -> dict[str, Any]:
    """Read authoritative timing facts for one clip.

    ``nb_frames`` from the MP4 header is unreliable on this dataset: 90 of the
    100 clips over-report it, so ``frame_count / fps`` would inflate every
    duration (by up to 3.3x). Durations therefore come from the decoded stream
    and the container. Declared values are still recorded, under ``declared_*``
    names, purely as diagnostics.
    """

    try:
        timing = read_timing(path, ffprobe)
    except FFProbeError as exc:
        raise ValueError(f"cannot probe video: {path}: {exc}") from exc

    if timing.duration_sec <= 0 or timing.frame_count_decoded <= 0:
        raise ValueError(f"unusable timing for {path}")

    return {
        "duration_sec": timing.duration_sec,
        "video_stream_duration": timing.video_stream_duration,
        "audio_stream_duration": timing.audio_stream_duration,
        # ``format.duration``, so that
        # duration_sec == max(container, video, audio, frame_end) is re-derivable
        "container_duration_sec": timing.container_duration_sec,
        "frame_count": timing.frame_count_decoded,
        "fps": timing.fps_declared,
        "last_frame_pts": timing.frame_times[-1] if timing.frame_times else 0.0,
        "frame_end_sec": round(timing.frame_end_sec, 6),
        # --- frame-axis bounds (added 2026-09-25) ---------------------------
        # The frame sequence does not always start at 0 or end at the container
        # duration.  Both ends are recorded so that "which part of [0, D] has no
        # frame at all" is answerable from the manifest alone.
        "first_frame_pts": timing.first_frame_pts,
        "head_gap_sec": round(timing.head_gap_sec, 6),
        "tail_gap_sec": round(timing.tail_gap_sec, 6),
        "frame_end_excess_sec": round(timing.frame_end_excess_sec, 6),
        # --- edit list (added 2026-09-25) -----------------------------------
        # All 100 attachment-1 clips carry one; it maps media time onto
        # presentation time, which is why pts_time and stream.duration are the
        # quantities to use.  Recorded so the choice is auditable.
        "has_edit_list": timing.has_edit_list,
        "edit_list_count": len(timing.edit_lists),
        "edit_list_entries": ";".join(
            "|".join(f"{segment}/{media}/{rate}" for segment, media, rate in entries)
            for entries in timing.edit_lists
        ),
        # ``elst.segment_duration`` is in movie-timescale units; divide by the
        # ``mvhd`` timescale to get seconds, which is what guard C16 compares
        # against the ffprobe stream durations.
        "movie_timescale": timing.movie_timescale,
        "movie_duration_sec": (
            round(timing.movie_duration / timing.movie_timescale, 6)
            if timing.movie_timescale
            else 0.0
        ),
        "edit_segment_sec": ";".join(
            f"{segment / timing.movie_timescale:.6f}"
            for entries in timing.edit_lists
            for segment, _media, _rate in entries
        ) if timing.movie_timescale else "",
        # diagnostics: from metadata, may be wrong
        "declared_frame_count": timing.frame_count_declared,
        "declared_duration_sec": (
            timing.frame_count_declared / timing.fps_declared
            if timing.fps_declared
            else 0.0
        ),
        "frames_match": timing.frame_count_reliable,
    }


def build_manifest(
    data_root: str | Path,
    output: str | Path,
    ffprobe: str | Path | None = None,
) -> list[dict[str, Any]]:
    root = Path(data_root).resolve()
    label_path = root / "label-100.xlsx"
    if not label_path.exists():
        raise FileNotFoundError(label_path)

    workbook = openpyxl.load_workbook(label_path, data_only=True, read_only=True)
    sheet = workbook["label"]
    rows = list(sheet.iter_rows(min_row=2, values_only=True))

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for video_id, clip_id, text, label, annotation in rows:
        video_id = str(video_id)
        clip_id = str(clip_id)
        sample_id = make_sample_id(video_id, clip_id)
        if sample_id in seen:
            raise ValueError(f"duplicate label id: {sample_id}")
        seen.add(sample_id)

        video_path = root / video_id / f"{clip_id}.mp4"
        if not video_path.exists():
            raise FileNotFoundError(f"label has no matching video: {sample_id}")
        metadata = _video_metadata(video_path, ffprobe)
        records.append(
            {
                "id": sample_id,
                "safe_id": safe_id(sample_id),
                "video_id": video_id,
                "clip_id": clip_id,
                "video_relpath": video_path.relative_to(root).as_posix(),
                "text": str(text),
                "label": float(label),
                "annotation": str(annotation),
                **metadata,
            }
        )

    video_ids = {
        make_sample_id(path.parent.name, path.stem) for path in root.rglob("*.mp4")
    }
    if video_ids != seen:
        raise ValueError(
            f"video/label mismatch: missing labels={sorted(video_ids-seen)}, "
            f"missing videos={sorted(seen-video_ids)}"
        )
    write_jsonl(output, records)
    return records

