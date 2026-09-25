from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


ID_SEPARATOR = "$_$"


def make_sample_id(video_id: str, clip_id: str) -> str:
    return f"{video_id}{ID_SEPARATOR}{clip_id}"


def safe_id(sample_id: str) -> str:
    return sample_id.replace(ID_SEPARATOR, "__")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
    return records


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def is_audible_timeline(record: dict[str, Any]) -> bool:
    """True when the clip's audio track carried a measurable signal.

    Two of the 100 attachment-1 files have a digitally silent audio track.  The
    aligner marks them with ``diagnostics.audio_present == False`` and writes the
    ``-1.0`` sentinel in every word time rather than inventing boundaries, so
    *every* statistic derived from those times -- coverage, speaking rate,
    zero-length count, "is the boundary inside the clip" -- is meaningless for
    them.  Consumers must filter on this flag, not on "does the sample have
    words", which is what they used to do.
    """

    return bool((record.get("diagnostics") or {}).get("audio_present", True))


def audible_records(
    records: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """The subset of ``records`` whose word times are measurable."""

    return [r for r in records if is_audible_timeline(r)]


def silent_ids(records: Iterable[dict[str, Any]]) -> set[str]:
    """Ids of the samples whose audio track carried no signal."""

    return {r["id"] for r in records if not is_audible_timeline(r)}

