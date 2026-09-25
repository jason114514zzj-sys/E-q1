"""Per-modality, per-dimension standardization for the E-problem features.

Why the default "subtract the global mean" is wrong here
-------------------------------------------------------
Verified on ``aligned_50.pkl`` (train, n=3395):

===================  ==========================  ====================
modality             value range                 note
===================  ==========================  ====================
``audio`` dim 0      ``[0, 500]``                artificially clipped
``audio`` dims 1..73 ``[-47.7, 36.0]`` (p99 ±3.1) scale ratio approx 118x
``vision``           35 dims, whole-sample zeros in 110/3395 train rows
``text``             768 dims, **no zeros at all** (padding is non-zero)
===================  ==========================  ====================

``audio`` dimension 0 is on a scale two orders of magnitude larger than the
other 73.  A single global mean/std therefore leaves dim 0 dominating every
distance and every fused representation, while the remaining 73 COVAREP
dimensions get squashed towards zero.

This module standardizes **per dimension**, which is the correct treatment
when channels have different physical units (as COVAREP does).

The statistics must be fitted on the **training split only** and then applied
unchanged to validation/test/attachment 3/attachment 4.  Fitting on the union
of all splits is a leakage error: attachment 3 and 4 carry no labels, so their
labels cannot leak, but their *feature distribution* would still leak into the
training preprocessing and make the reported validation numbers optimistic.

Missing positions are never allowed to influence the statistics.  A step is
"missing" when it is all-zero after the padding convention of the dataset, and
those steps are excluded from the mean/std accumulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

__all__ = [
    "FeatureScaler",
    "fit_scaler",
    "audio_valid_mask",
    "vision_valid_mask",
    "split_scale_report",
    "MODALITY_DIMS",
]

#: Expected feature width per modality, as measured on attachment 2.
MODALITY_DIMS = {"audio": 74, "vision": 35, "text": 768}


def audio_valid_mask(x: np.ndarray) -> np.ndarray:
    """(N, T) mask: True where the audio step carries real signal.

    ``aligned_50.pkl`` has no length field, so the mask must be derived.  A
    step counts as present when it is not entirely zero.  Note that step 0 is
    all-zero in **100%** of samples across all three splits, so it is padding
    by construction and this rule correctly drops it.
    """
    arr = np.asarray(x)
    if arr.ndim != 3:
        raise ValueError(f"expected (N, T, D), got {arr.shape}")
    return ~np.all(arr == 0, axis=-1)


def vision_valid_mask(x: np.ndarray) -> np.ndarray:
    """(N, T) mask for vision, identical convention to :func:`audio_valid_mask`."""
    return audio_valid_mask(x)


@dataclass
class FeatureScaler:
    """Per-dimension z-score statistics fitted on one split.

    ``missing_is_zero`` records whether the all-zero convention was used to
    exclude padding from the statistics.  Text must pass ``False``: it has no
    zeros and its padding is identified by the ``text_bert`` attention mask
    instead (see :mod:`src.q1_alignment.text_pooling`).
    """

    modality: str
    mean: np.ndarray
    std: np.ndarray
    n_steps_used: int
    missing_is_zero: bool = True
    source_split: str = "train"
    notes: list[str] = field(default_factory=list)

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Apply the fitted statistics; padding positions stay exactly zero."""
        arr = np.asarray(x, dtype=np.float32)
        if arr.shape[-1] != self.mean.shape[-1]:
            raise ValueError(
                f"{self.modality}: expected last dim {self.mean.shape[-1]}, "
                f"got {arr.shape[-1]}"
            )
        if self.missing_is_zero:
            mask = (~np.all(arr == 0, axis=-1))[..., None]
            out = (arr - self.mean) / self.std
            return np.where(mask, out, 0.0).astype(np.float32)
        return ((arr - self.mean) / self.std).astype(np.float32)

    def inverse_transform(self, z: np.ndarray) -> np.ndarray:
        return (np.asarray(z, dtype=np.float32) * self.std + self.mean)

    def to_dict(self) -> dict:
        return {
            "modality": self.modality,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "n_steps_used": int(self.n_steps_used),
            "missing_is_zero": bool(self.missing_is_zero),
            "source_split": self.source_split,
            "notes": list(self.notes),
        }


def fit_scaler(
    x: np.ndarray,
    modality: str,
    *,
    source_split: str = "train",
    missing_is_zero: bool = True,
    eps: float = 1e-6,
) -> FeatureScaler:
    """Fit per-dimension mean/std, excluding padding steps.

    ``eps`` guards dimensions that are constant (std 0) so the transform cannot
    produce inf/NaN.  Without this guard, a degenerate dimension would poison
    the entire feature matrix.
    """
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(f"expected (N, T, D), got {arr.shape}")

    notes: list[str] = []
    expected = MODALITY_DIMS.get(modality)
    if expected is not None and arr.shape[-1] != expected:
        notes.append(
            f"expected {expected} dims for {modality}, got {arr.shape[-1]}"
        )

    if missing_is_zero:
        mask = ~np.all(arr == 0, axis=-1)          # (N, T)
        valid = arr[mask]                          # (M, D)
        n_steps = int(mask.sum())
    else:
        valid = arr.reshape(-1, arr.shape[-1])
        n_steps = int(arr.shape[0] * arr.shape[1])

    if valid.size == 0:
        raise ValueError(f"{modality}: no valid steps found")

    mean = valid.mean(axis=0)
    std = valid.std(axis=0)
    degenerate = int((std < eps).sum())
    std = np.maximum(std, eps)
    if degenerate:
        notes.append(f"{degenerate} dimension(s) had std<{eps}; clamped")

    if modality == "audio":
        notes.append(
            "per-dimension scaling is mandatory: dim0 range [0,500] vs "
            "dims1..73 p99 +-3.1 (scale ratio approx 118x)"
        )

    return FeatureScaler(
        modality=modality,
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        n_steps_used=n_steps,
        missing_is_zero=missing_is_zero,
        source_split=source_split,
        notes=notes,
    )


def split_scale_report(splits: dict[str, np.ndarray], modality: str) -> dict:
    """Compare per-dimension statistics across splits.

    Used by the guard to prove that train/valid/test share a scale, which is
    the precondition for applying training statistics to the other splits.
    """
    out: dict[str, dict] = {}
    for name, x in splits.items():
        arr = np.asarray(x, dtype=np.float64)
        mask = ~np.all(arr == 0, axis=-1)
        valid = arr[mask] if mask.any() else arr.reshape(-1, arr.shape[-1])
        out[name] = {
            "n_valid_steps": int(mask.sum()),
            "dim0_min": float(valid[:, 0].min()),
            "dim0_max": float(valid[:, 0].max()),
            "rest_p99_abs": float(np.percentile(np.abs(valid[:, 1:]), 99)),
            "scale_ratio": float(
                np.percentile(np.abs(valid[:, 0]), 99)
                / max(np.percentile(np.abs(valid[:, 1:]), 99), 1e-9)
            ),
        }
    return out
