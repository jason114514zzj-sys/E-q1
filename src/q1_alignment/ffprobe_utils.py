"""Reliable video timing read from ffprobe.

Why this module exists
----------------------
``cv2.VideoCapture.get(cv2.CAP_PROP_FRAME_COUNT)`` reads the MP4 ``nb_frames``
metadata field. On the competition clips that field is unreliable: for 90 of
the 100 clips it reports more frames than the decoder can actually produce
(e.g. 261 declared vs 163 decodable). Dividing that wrong frame count by the
frame rate inflates every duration and corrupts any timeline built from it.

Everything here is therefore derived from decoded frames (``pts_time``) and
from ``format.duration``, never from ``nb_frames``.
"""

from __future__ import annotations

import json
import shutil
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path


class FFProbeError(RuntimeError):
    """Raised when ffprobe is missing or returns unusable output."""


def find_ffprobe(explicit: str | Path | None = None) -> str:
    """Locate an ffprobe binary, preferring an explicit path."""

    if explicit is not None:
        candidate = Path(explicit)
        if candidate.is_file():
            return str(candidate)
        raise FFProbeError(f"ffprobe not found at {explicit}")
    found = shutil.which("ffprobe")
    if found:
        return found
    # common conda / ffmpeg install locations
    home = Path.home()
    for candidate in (
        home / "miniconda3/envs/mosei/bin/ffprobe",
        Path("/usr/bin/ffprobe"),
        Path("/usr/local/bin/ffprobe"),
    ):
        if candidate.is_file():
            return str(candidate)
    raise FFProbeError("ffprobe executable not found")


@dataclass(frozen=True)
class VideoTiming:
    """Authoritative timing facts for one media file."""

    duration_sec: float
    video_stream_duration: float
    audio_stream_duration: float
    frame_count_declared: int
    frame_count_decoded: int
    frame_times: list[float]
    fps_declared: float
    # added 2026-09-25: the frame axis does NOT always start at 0 and does not
    # always end at the container duration, so both ends are recorded explicitly
    # instead of being assumed.
    edit_lists: tuple[tuple[tuple[int, int, int], ...], ...] = ()
    # added 2026-09-25: the container's own ``format.duration``, kept so that
    # ``duration_sec == max(container_duration_sec, video, audio, frame_end)``
    # can be re-derived from the manifest without re-running ffprobe.
    container_duration_sec: float = 0.0
    # added 2026-09-25: the movie timescale, without which the recorded ``elst``
    # integers cannot be converted to seconds (guard C16 converts and checks).
    movie_timescale: int = 0
    movie_duration: int = 0

    @property
    def frame_end_sec(self) -> float:
        """End of the last frame's covering interval: ``last_pts + 1/fps``."""
        if not self.frame_times or not self.fps_declared:
            return 0.0
        return self.frame_times[-1] + self.frame_duration

    @property
    def frame_end_excess_sec(self) -> float:
        """How far the frame axis runs past every other duration bound.

        Positive on the clips where the container understates the media: the
        last frame ends after ``format.duration`` and both stream durations, so
        the master axis had to be extended to keep the tail frame covered.
        """
        if not self.frame_times or not self.fps_declared:
            return 0.0
        bound = max(
            self.container_duration_sec,
            self.video_stream_duration,
            self.audio_stream_duration,
        )
        return self.frame_end_sec - bound

    @property
    def frame_count_reliable(self) -> bool:
        return self.frame_count_declared == self.frame_count_decoded

    @property
    def duration_ratio(self) -> float:
        """Declared-driven duration over real duration (1.0 is healthy)."""
        if self.duration_sec <= 0 or self.fps_declared <= 0:
            return 0.0
        naive = self.frame_count_declared / self.fps_declared
        return naive / self.duration_sec

    @property
    def frame_duration(self) -> float:
        return 1.0 / self.fps_declared if self.fps_declared else 0.0

    @property
    def first_frame_pts(self) -> float:
        return self.frame_times[0] if self.frame_times else 0.0

    @property
    def head_gap_sec(self) -> float:
        """Container head-room before the first frame.

        A position whose interval falls entirely inside this head-room has no
        frame centre and therefore ``mask = 0``; the gap is recorded so that
        fact is auditable instead of looking like a coding error.
        """
        return self.first_frame_pts

    @property
    def tail_gap_sec(self) -> float:
        """``duration_sec - (last_pts + 1/fps)``.

        Negative means the frame sequence runs past the container duration, so
        the last frame's covering interval is clipped by the master axis.
        """
        if not self.frame_times or not self.fps_declared:
            return 0.0
        return self.duration_sec - (self.frame_times[-1] + self.frame_duration)

    @property
    def has_edit_list(self) -> bool:
        return bool(self.edit_lists)


# Container boxes that may hold an ``edts/elst`` in their subtree.
_EDIT_LIST_CONTAINERS = {
    b"moov", b"trak", b"mdia", b"minf", b"stbl", b"edts", b"dinf", b"udta",
}


def read_movie_header(path: str | Path) -> tuple[int, int]:
    """Return ``(movie_timescale, movie_duration)`` from the ``mvhd`` box.

    ``elst.segment_duration`` is expressed in the **movie** timescale, so without
    this value the recorded edit-list integers cannot be converted into seconds
    and therefore cannot be checked against anything.  Both fields are optional
    in principle, so a missing header yields ``(0, 0)`` rather than an error.
    """

    try:
        data = Path(path).read_bytes()
    except OSError:
        return (0, 0)

    _CONTAINERS = {b"moov", b"trak", b"mdia", b"minf", b"stbl", b"edts",
                   b"dinf", b"udta"}
    result: list[int] = []

    def walk(offset: int, end: int) -> None:
        while offset + 8 <= end:
            size = struct.unpack(">I", data[offset:offset + 4])[0]
            kind = data[offset + 4:offset + 8]
            header = 8
            if size == 1:
                if offset + 16 > end:
                    return
                size = struct.unpack(">Q", data[offset + 8:offset + 16])[0]
                header = 16
            elif size == 0:
                size = end - offset
            if size < header or offset + size > end:
                return
            if kind == b"mvhd" and not result:
                body = data[offset + header:offset + size]
                if len(body) >= 32:
                    version = body[0]
                    # version 0: 4-byte creation/modification; version 1: 8-byte.
                    # Layout after that is timescale (4 bytes) then duration (4).
                    base = 4 + (16 if version == 1 else 8)
                    if base + 8 <= len(body):
                        timescale = struct.unpack(">I", body[base:base + 4])[0]
                        duration = struct.unpack(">I", body[base + 4:base + 8])[0]
                        result.extend([int(timescale), int(duration)])
                        return
            elif kind in _CONTAINERS:
                walk(offset + header, offset + size)
                if result:
                    return
            offset += size

    try:
        walk(0, len(data))
    except Exception:  # pragma: no cover - a malformed box tree must not abort
        return (0, 0)
    if len(result) == 2:
        return (result[0], result[1])
    return (0, 0)


def read_edit_list(path: str | Path) -> tuple[tuple[tuple[int, int, int], ...], ...]:
    """Return every MP4 edit list as ``((segment_duration, media_time, media_rate), ...)``.

    Why this is recorded rather than ignored
    ---------------------------------------
    An ``elst`` maps the *media* timeline onto the *presentation* timeline.  When
    one is present, "how long the file is" and "where the frames start" are not
    the same question, and the container duration is the presentation one.
    Measured on attachment 1 (2026-09-25): **all 100 clips carry an edit list**,
    two per file (one video, one audio), so ffmpeg is already applying it and the
    ``pts_time`` we read is a presentation time.  Guard ``C16`` converts each
    ``segment_duration`` with the movie timescale and checks it against the
    ffprobe stream durations, which is what makes this claim machine-verifiable
    rather than a note.
    """

    try:
        data = Path(path).read_bytes()
    except OSError:
        return ()

    found: list[tuple[tuple[int, int, int], ...]] = []

    def walk(offset: int, end: int) -> None:
        while offset + 8 <= end:
            size = struct.unpack(">I", data[offset:offset + 4])[0]
            kind = data[offset + 4:offset + 8]
            header = 8
            if size == 1:
                if offset + 16 > end:
                    return
                size = struct.unpack(">Q", data[offset + 8:offset + 16])[0]
                header = 16
            elif size == 0:
                size = end - offset
            if size < header or offset + size > end:
                return
            if kind == b"elst":
                body = data[offset + header:offset + size]
                if len(body) >= 8:
                    count = struct.unpack(">I", body[4:8])[0]
                    entries = []
                    for index in range(count):
                        start = 8 + 12 * index
                        if start + 12 > len(body):
                            break
                        entries.append(struct.unpack(">III", body[start:start + 12]))
                    found.append(tuple(entries))
            elif kind in _EDIT_LIST_CONTAINERS:
                walk(offset + header, offset + size)
            offset += size

    try:
        walk(0, len(data))
    except Exception:  # pragma: no cover - a malformed box tree must not abort
        return ()
    return tuple(found)


def _run(command: list[str]) -> str:
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise FFProbeError(
            f"ffprobe failed ({completed.returncode}): {completed.stderr.strip()[:400]}"
        )
    return completed.stdout


def probe_streams(path: str | Path, ffprobe: str | Path | None = None) -> dict:
    """Return the raw ffprobe stream/format JSON."""

    executable = find_ffprobe(ffprobe)
    output = _run([
        executable, "-v", "error",
        "-show_entries",
        "format=duration:stream=codec_type,duration,nb_frames,r_frame_rate,avg_frame_rate",
        "-of", "json", str(path),
    ])
    return json.loads(output)


def probe_frame_times(path: str | Path, ffprobe: str | Path | None = None) -> list[float]:
    """Return the presentation timestamp of every decodable video frame.

    This is the ground truth frame count: it counts frames the decoder can
    actually deliver, unlike ``nb_frames``.
    """

    executable = find_ffprobe(ffprobe)
    output = _run([
        executable, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "frame=pts_time", "-of", "json", str(path),
    ])
    payload = json.loads(output)
    times: list[float] = []
    for frame in payload.get("frames", []):
        raw = frame.get("pts_time")
        if raw in (None, "N/A"):
            continue
        try:
            times.append(float(raw))
        except (TypeError, ValueError):
            continue
    times.sort()
    return times


def _parse_rate(value: str | None) -> float:
    """Parse ffprobe rational frame rates such as ``30000/1001``."""

    if not value or value == "N/A":
        return 0.0
    if "/" in value:
        numerator, _, denominator = value.partition("/")
        try:
            den = float(denominator)
            return float(numerator) / den if den else 0.0
        except ValueError:
            return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def read_timing(path: str | Path, ffprobe: str | Path | None = None) -> VideoTiming:
    """Collect every timing fact needed to build a trustworthy timeline."""

    info = probe_streams(path, ffprobe)
    fmt = info.get("format", {}) or {}
    streams = info.get("streams", []) or []

    try:
        duration = float(fmt.get("duration", 0.0) or 0.0)
    except (TypeError, ValueError):
        duration = 0.0
    duration_from_container = duration

    video_duration = 0.0
    audio_duration = 0.0
    declared_frames = 0
    fps_declared = 0.0
    for stream in streams:
        kind = stream.get("codec_type")
        raw_duration = stream.get("duration")
        try:
            stream_duration = float(raw_duration) if raw_duration not in (None, "N/A") else 0.0
        except (TypeError, ValueError):
            stream_duration = 0.0
        if kind == "video":
            video_duration = stream_duration or video_duration
            try:
                declared_frames = int(stream.get("nb_frames") or 0)
            except (TypeError, ValueError):
                declared_frames = 0
            fps_declared = _parse_rate(stream.get("r_frame_rate")) or _parse_rate(
                stream.get("avg_frame_rate")
            )
        elif kind == "audio":
            audio_duration = stream_duration or audio_duration

    frame_times = probe_frame_times(path, ffprobe)
    edit_lists = read_edit_list(path)
    movie_timescale, movie_duration = read_movie_header(path)

    # The master timeline must cover every decodable frame.  Measured on
    # attachment 1 (2026-09-25): on **9 of the 100** clips ``last_pts + 1/fps``
    # lies strictly beyond *every* other duration bound (container, video stream,
    # audio stream), by up to 100.0 ms, so taking the container duration alone
    # would leave the tail of the axis uncovered.  The extension is recorded per
    # sample as ``frame_end_excess_sec``.
    frame_end = 0.0
    if frame_times and fps_declared > 0:
        frame_end = frame_times[-1] + 1.0 / fps_declared

    duration = max(duration, video_duration, audio_duration, frame_end)
    if duration <= 0:
        # nothing usable anywhere: fall back to the frame span, else 1 second
        duration = frame_end or (frame_times[-1] if frame_times else 0.0) or 1.0

    return VideoTiming(
        duration_sec=duration,
        video_stream_duration=video_duration,
        audio_stream_duration=audio_duration,
        frame_count_declared=declared_frames,
        frame_count_decoded=len(frame_times),
        frame_times=frame_times,
        fps_declared=fps_declared,
        edit_lists=edit_lists,
        container_duration_sec=duration_from_container,
        movie_timescale=movie_timescale,
        movie_duration=movie_duration,
    )
