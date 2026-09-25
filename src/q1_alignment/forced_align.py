"""Word-level forced alignment for the 100 competition clips.

The competition transcript is authoritative: this module only estimates *when*
each transcript word is spoken. It never rewrites, reorders or drops words.

Three interchangeable back ends are provided so the pipeline can run with or
without a downloadable acoustic model:

``energy``
    Pure signal-processing baseline. Uses voice-activity detection plus
    character-length-proportional allocation inside each voiced span. No model
    download, fully reproducible, and adequate for coarse alignment.

``ctc``
    Frame-level CTC forced alignment driven by ``torchaudio``'s bundled
    Wav2Vec2 pipeline. Requires an acoustic model, hence a network download.

``external``
    Adapter for a previously produced alignment (MFA JSON/CTM, WhisperX, ...)
    loaded from disk.

Every back end returns the same record shape consumed by
``q1_alignment.timeline.validate_timeline``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .monotonic_align import align_words_monotonic
from .timeline import tokenize_transcript, validate_timeline

SUPPORTED_METHODS = ("monotonic", "energy", "ctc", "external")

# The competition clips carry an unreliable ``nb_frames`` header (90 of 100
# over-report it), so ``duration`` passed in here must already come from
# ffprobe-decoded timing -- see :mod:`q1_alignment.ffprobe_utils`.

# Energy floor under which a track is treated as carrying no observable speech.
# ``_frame_energy`` adds 1e-12 inside the square root, so a *digitally silent*
# track (every sample exactly 0) yields a peak RMS of 1e-6.  1e-5 therefore
# catches true digital silence and leaves quiet-but-real recordings alone.
# Measured on attachment 1 (2026-09-25): 2 of the 100 clips are digitally
# silent -- and one of them carries a 13-word transcript.  Alignment used to
# "succeed" on those clips, inventing a set of monotone boundaries that passed
# every structural check while resting on no acoustic evidence at all.
SILENCE_PEAK_FLOOR = 1e-5

# Time written for a word whose position on the timeline is *unknown* rather
# than estimated.  Kept equal to the interval sentinel used by the alignment
# output so downstream code sees one convention, not two.
SENTINEL_TIME = -1.0


def frame_energy_stats(rms: "np.ndarray", threshold_ratio: float = 0.12) -> dict[str, float]:
    """Peak RMS and the fraction of frames carrying speech-level energy.

    ``speech_active_ratio`` is reported for every sample, not as a pass/fail
    gate: a low value means the clip is mostly pause, which is a property of the
    material and not a defect.  It exists so that a coverage of 1.000 can never
    hide "the whole track is silent".
    """

    if rms.size == 0:
        return {"energy_peak": 0.0, "speech_active_ratio": 0.0}
    peak = float(rms.max())
    if peak <= SILENCE_PEAK_FLOOR:
        return {"energy_peak": peak, "speech_active_ratio": 0.0}
    threshold = max(float(np.percentile(rms, 95)) * threshold_ratio, 1e-5)
    return {
        "energy_peak": peak,
        "speech_active_ratio": round(float((rms >= threshold).mean()), 6),
    }


@dataclass
class AlignResult:
    """Word timings plus the diagnostics needed for QC reporting."""

    words: list[dict[str, Any]]
    method: str
    diagnostics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    final_usable: bool = True


# --------------------------------------------------------------------------
# audio loading
# --------------------------------------------------------------------------
def load_audio_mono(
    video_path: str | Path,
    sample_rate: int = 16000,
    ffmpeg_exe: str | None = None,
) -> tuple[np.ndarray, int]:
    """Decode the audio track of a video to mono float32 via ffmpeg.

    Decoding goes through ffmpeg directly so the source MP4 is never modified
    and no temporary WAV has to be written for the in-memory path.
    """

    import subprocess

    executable = ffmpeg_exe or "ffmpeg"
    command = [
        executable,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "-",
    ]
    completed = subprocess.run(command, capture_output=True)
    if completed.returncode != 0:
        message = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg audio decode failed for {video_path}: {message}")
    waveform = np.frombuffer(completed.stdout, dtype=np.float32)
    if waveform.size == 0:
        raise RuntimeError(f"decoded audio is empty: {video_path}")
    return waveform, sample_rate


# --------------------------------------------------------------------------
# energy / VAD back end
# --------------------------------------------------------------------------
def _frame_energy(
    waveform: np.ndarray,
    sample_rate: int,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    frame_length = max(1, int(round(sample_rate * frame_ms / 1000.0)))
    hop_length = max(1, int(round(sample_rate * hop_ms / 1000.0)))
    if waveform.size < frame_length:
        padded = np.zeros(frame_length, dtype=np.float32)
        padded[: waveform.size] = waveform
        waveform = padded
    frame_count = 1 + (waveform.size - frame_length) // hop_length
    strides = (waveform.strides[0] * hop_length, waveform.strides[0])
    frames = np.lib.stride_tricks.as_strided(
        waveform, shape=(frame_count, frame_length), strides=strides, writeable=False
    )
    rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1) + 1e-12)
    centers = (np.arange(frame_count) * hop_length + frame_length / 2.0) / sample_rate
    return rms, centers


def _voiced_spans(
    rms: np.ndarray,
    centers: np.ndarray,
    duration: float,
    threshold_ratio: float = 0.12,
    min_silence_sec: float = 0.12,
) -> list[tuple[float, float]]:
    """Return merged (start, end) spans where speech energy is present."""

    if rms.size == 0:
        return [(0.0, duration)]
    peak = float(np.percentile(rms, 95))
    if peak <= 0:
        return [(0.0, duration)]
    threshold = max(peak * threshold_ratio, 1e-5)
    active = rms >= threshold

    spans: list[list[float]] = []
    start: float | None = None
    hop = float(np.median(np.diff(centers))) if centers.size > 1 else 0.01
    for index, flag in enumerate(active):
        if flag and start is None:
            start = float(centers[index]) - hop / 2.0
        elif not flag and start is not None:
            spans.append([max(0.0, start), float(centers[index]) - hop / 2.0])
            start = None
    if start is not None:
        spans.append([max(0.0, start), float(centers[-1]) + hop / 2.0])

    if not spans:
        return [(0.0, duration)]

    # merge spans separated by a short pause
    merged: list[list[float]] = [spans[0]]
    for begin, end in spans[1:]:
        if begin - merged[-1][1] <= min_silence_sec:
            merged[-1][1] = end
        else:
            merged.append([begin, end])

    return [(max(0.0, a), min(duration, b)) for a, b in merged if b > a]


def _allocate(weights: np.ndarray, total: float) -> np.ndarray:
    """Split ``total`` into proportional pieces that sum exactly to ``total``."""

    weights = np.asarray(weights, dtype=np.float64)
    if weights.size == 0:
        return np.zeros(0, dtype=np.float64)
    weights = np.maximum(weights, 1e-9)
    pieces = weights / weights.sum() * total
    # correct drift on the last element so the span is covered exactly
    drift = total - pieces.sum()
    pieces[-1] += drift
    return np.maximum(pieces, 0.0)


def align_monotonic(
    video_path: str | Path,
    transcript: str,
    duration: float,
    sample_rate: int = 16000,
    ffmpeg_exe: str | None = None,
    pause_weight: float = 0.5,
    **kwargs: Any,
) -> AlignResult:
    """Constraint-based alignment with guaranteed full coverage.

    Replaces :func:`align_energy` as the default back end. The earlier
    per-span allocation could leave most of a clip unaligned (median coverage
    42%); here coverage, monotonicity and speaking-rate sanity are enforced by
    construction, and boundaries are refined towards acoustic pauses.
    """

    words = tokenize_transcript(transcript)
    if not words:
        raise ValueError("transcript contains no words")

    waveform, rate = load_audio_mono(video_path, sample_rate, ffmpeg_exe)
    rms, centers = _frame_energy(waveform, rate)

    stats = frame_energy_stats(rms)
    audio_present = stats["energy_peak"] > SILENCE_PEAK_FLOOR

    if not audio_present:
        # Digitally silent track.  The transcript cannot be placed on this
        # timeline, and *any* set of boundaries would be invented rather than
        # measured.  The word list is still emitted -- so that the text axis and
        # the "which words does this clip contain" record survive -- but every
        # time field carries the sentinel and the sample is not usable.
        entries = [
            {"word": word, "start": SENTINEL_TIME, "end": SENTINEL_TIME,
             "confidence": None}
            for word in words
        ]
        result = AlignResult(
            words=entries,
            method="audio_absent",
            diagnostics={
                **stats,
                "audio_present": False,
                "sentinel": SENTINEL_TIME,
                "sample_rate": int(rate),
                "audio_seconds": float(waveform.size / rate),
                "frame_count": int(rms.size),
                "coverage": 0.0,
                "word_count": len(words),
            },
            final_usable=False,
        )
        result.warnings.append(
            "audio track carries no observable speech "
            f"(peak RMS {stats['energy_peak']:.3e} <= {SILENCE_PEAK_FLOOR:g}); "
            "no per-word time emitted (sentinel -1.0)"
        )
        return result

    entries, quality = align_words_monotonic(
        words,
        duration,
        rms=rms,
        energy_centers=centers,
        pause_weight=pause_weight,
    )

    result = AlignResult(
        words=entries,
        method="monotonic_constrained_pause_aware",
        diagnostics={
            **stats,
            "audio_present": True,
            "sample_rate": int(rate),
            "audio_seconds": float(waveform.size / rate),
            "frame_count": int(rms.size),
            "coverage": round(quality.coverage, 6),
            "min_span_sec": round(quality.min_span, 6),
            "median_rate_wps": round(quality.median_rate, 4),
            "max_rate_wps": round(quality.max_rate, 4),
            "rate_outliers": quality.rate_outliers,
            "pause_snapped": quality.pause_snapped,
            "pause_gain": round(quality.boundary_energy_gain, 4),
        },
    )
    if quality.warnings:
        result.warnings.extend(quality.warnings)

    issues = validate_timeline(
        {"words": entries}, duration, tolerance=max(0.05, 0.02 * duration)
    )
    hard = [issue for issue in issues if "outside clip" not in issue]
    if hard:
        result.warnings.extend(hard)
        result.final_usable = False
    return result


def align_energy(
    video_path: str | Path,
    transcript: str,
    duration: float,
    sample_rate: int = 16000,
    ffmpeg_exe: str | None = None,
    **kwargs: Any,
) -> AlignResult:
    """Voice-activity based alignment; no acoustic model required."""

    words = tokenize_transcript(transcript)
    if not words:
        raise ValueError("transcript contains no words")

    waveform, rate = load_audio_mono(video_path, sample_rate, ffmpeg_exe)
    rms, centers = _frame_energy(waveform, rate)

    stats = frame_energy_stats(rms)
    if stats["energy_peak"] <= SILENCE_PEAK_FLOOR:
        # Same guard as align_monotonic: never place words on a silent track.
        entries = [
            {"word": word, "start": SENTINEL_TIME, "end": SENTINEL_TIME,
             "confidence": None}
            for word in words
        ]
        result = AlignResult(
            words=entries,
            method="audio_absent",
            diagnostics={**stats, "audio_present": False,
                         "sentinel": SENTINEL_TIME, "coverage": 0.0,
                         "word_count": len(words)},
            final_usable=False,
        )
        result.warnings.append(
            "audio track carries no observable speech "
            f"(peak RMS {stats['energy_peak']:.3e}); no per-word time emitted"
        )
        return result

    spans = _voiced_spans(rms, centers, duration)

    weights = np.array([len(word.replace("'", "")) + 1 for word in words], dtype=np.float64)

    # distribute words across voiced spans in proportion to span duration,
    # guaranteeing the full text is covered even if VAD is imperfect
    span_lengths = np.array([end - start for start, end in spans], dtype=np.float64)
    span_share = _allocate(span_lengths, float(weights.sum()))

    boundaries: list[float] = []
    cursor = 0
    for (start, end), share in zip(spans, span_share):
        count = int(round(share))
        count = max(1, min(count, len(words) - cursor)) if cursor < len(words) else 0
        if count <= 0:
            continue
        local = weights[cursor : cursor + count]
        pieces = _allocate(local, end - start)
        for piece in pieces:
            boundaries.append(start + piece)
            start += piece
        cursor += count
    while len(boundaries) < len(words):
        boundaries.append(duration)

    starts = np.concatenate([[0.0], np.asarray(boundaries, dtype=np.float64)[:-1]])
    ends = np.asarray(boundaries, dtype=np.float64)
    # strictly increasing, non-overlapping
    starts = np.maximum.accumulate(starts)
    ends = np.maximum.accumulate(ends)
    ends = np.maximum(ends, starts + 1e-3)
    ends = np.minimum(ends, max(duration, float(ends[-1])))

    entries = [
        {
            "word": word,
            "start": float(starts[index]),
            "end": float(ends[index]),
            "confidence": None,
        }
        for index, word in enumerate(words)
    ]
    result = AlignResult(
        words=entries,
        method="vad_energy_proportional",
        diagnostics={
            "sample_rate": int(rate),
            "audio_seconds": float(waveform.size / rate),
            "voiced_span_count": len(spans),
            "voiced_ratio": float(sum(b - a for a, b in spans) / max(duration, 1e-9)),
            "frame_count": int(rms.size),
            **stats,
            "audio_present": True,
        },
    )
    issues = validate_timeline(
        {"words": entries}, duration, tolerance=max(0.05, 0.02 * duration)
    )
    # monotonicity is enforced by construction; keep only hard duration issues
    hard = [issue for issue in issues if "outside clip" not in issue]
    if hard:
        result.warnings.extend(hard)
        result.final_usable = False
    return result


# --------------------------------------------------------------------------
# CTC back end
# --------------------------------------------------------------------------
def align_ctc(
    video_path: str | Path,
    transcript: str,
    duration: float,
    bundle: Any = None,
    sample_rate: int = 16000,
    ffmpeg_exe: str | None = None,
    device: str = "cuda",
    **kwargs: Any,
) -> AlignResult:
    """CTC forced alignment using torchaudio's Wav2Vec2 pipeline.

    ``bundle`` must expose ``get_model`` and ``get_tokenizer`` as provided by
    :func:`load_ctc_bundle`. Raises ``RuntimeError`` when no model is available
    so callers can fall back to the energy back end.
    """

    if bundle is None:
        raise RuntimeError("no CTC acoustic model bundle supplied")

    import torch
    import torchaudio

    words = tokenize_transcript(transcript)
    if not words:
        raise ValueError("transcript contains no words")

    waveform, rate = load_audio_mono(video_path, sample_rate, ffmpeg_exe)
    tensor = torch.from_numpy(waveform).unsqueeze(0)

    model = bundle.get_model()
    tokenizer = bundle.get_tokenizer()

    target = []
    for word in words:
        pieces = tokenizer(word.lower())
        if not pieces:
            pieces = tokenizer(word.lower().strip("'"))
        target.extend(pieces)
    if not target:
        raise RuntimeError("tokenizer produced no symbols for the transcript")

    with torch.inference_mode():
        emission, _ = model(tensor)
    log_probs = torch.log_softmax(emission, dim=-1)
    if log_probs.dim() == 3:
        log_probs = log_probs[0]

    targets = torch.tensor([target], dtype=torch.int32)
    try:
        aligned, scores = torchaudio.functional.forced_align(
            log_probs.unsqueeze(0), targets, blank=0
        )
    except Exception as exc:  # pragma: no cover - depends on model internals
        raise RuntimeError(f"forced_align failed: {exc}") from exc

    spans = torchaudio.functional.merge_tokens(aligned[0], scores[0])
    frame_seconds = duration / max(int(log_probs.shape[0]), 1)

    # map token spans back to transcript words
    entries: list[dict[str, Any]] = []
    cursor = 0
    for word in words:
        pieces = tokenizer(word.lower()) or tokenizer(word.lower().strip("'"))
        count = len(pieces)
        picked = spans[cursor : cursor + count]
        cursor += count
        if picked:
            start = float(picked[0].start) * frame_seconds
            end = float(picked[-1].end) * frame_seconds
            confidence = float(np.mean([float(item.score) for item in picked]))
        else:
            start = entries[-1]["end"] if entries else 0.0
            end = start
            confidence = None
        entries.append(
            {
                "word": word,
                "start": start,
                "end": end,
                "confidence": confidence,
            }
        )

    result = AlignResult(
        words=entries,
        method="torchaudio_ctc_forced_align",
        diagnostics={"target_token_count": len(target), "frame_count": int(log_probs.shape[0])},
    )
    issues = validate_timeline({"words": entries}, duration, tolerance=max(0.05, 0.02 * duration))
    hard = [issue for issue in issues if "outside clip" not in issue]
    if hard:
        result.warnings.extend(hard)
        result.final_usable = False
    return result


# --------------------------------------------------------------------------
# CTC model loading
# --------------------------------------------------------------------------
def load_ctc_bundle(
    bundle_name: str = "WAV2VEC2_ASR_BASE_960H",
    device: str = "cuda",
    cache_dir: str | Path | None = None,
) -> Any:
    """Load a torchaudio CTC bundle for the ``ctc`` back end.

    The acoustic weights are downloaded on first use. On an isolated network
    this raises, and the caller should fall back to the ``energy`` back end.
    """

    import torch
    import torchaudio

    if not hasattr(torchaudio.pipelines, bundle_name):
        raise RuntimeError(f"unknown torchaudio bundle: {bundle_name}")
    bundle = getattr(torchaudio.pipelines, bundle_name)
    kwargs: dict[str, Any] = {}
    if cache_dir is not None:
        kwargs["dl_kwargs"] = {"model_dir": str(cache_dir)}
    model = bundle.get_model(**kwargs)
    if device == "cuda" and torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return bundle


# --------------------------------------------------------------------------
# dispatcher
# --------------------------------------------------------------------------
def align_transcript(
    video_path: str | Path,
    transcript: str,
    duration: float,
    method: str = "energy",
    **kwargs: Any,
) -> AlignResult:
    """Dispatch to the requested back end."""

    if method == "monotonic":
        return align_monotonic(video_path, transcript, duration, **kwargs)
    if method == "energy":
        return align_energy(video_path, transcript, duration, **kwargs)
    if method == "ctc":
        return align_ctc(video_path, transcript, duration, **kwargs)
    raise ValueError(f"unsupported alignment method: {method!r}")
