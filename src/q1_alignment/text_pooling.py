"""Mask-aware text feature handling for the E-problem.

Why this module exists
----------------------
Directly averaging the 50 aligned ``text`` steps is **wrong** for this dataset.

Verified facts (``work/check_text_pool.py``, train split, n=3395):

* ``text`` (N, 50, 768) contains **no zeros at all**.  The positions beyond the
  real utterance length are filled with *non-zero* values (mean -0.0102,
  std 0.4594).  So ``x == 0`` can never detect text padding.
* ``text_bert`` (N, 3, 50) carries the ground truth in three channels:
  channel 0 = BERT vocab ids, channel 1 = the 0/1 attention mask,
  channel 2 = all-zero placeholder.
* The mask is pure *tail* padding: interior zeros occur in 0/3395 samples.
* Valid-step counts are [min=3, p25=16, median=22, p75=32, max=50] -- i.e. over
  half the sequence is usually padding.

Quantified damage of naive averaging:

    relative difference ||naive - masked|| / ||masked|| : mean 0.4343, max 1.1189
    cosine(naive, masked)                              : mean 0.8769, min 0.4056

and the damage concentrates on short utterances (valid ratio < 0.2 -> relative
difference 0.78), which are exactly the low-information samples.

Attachment 3 exposes the same trap in a harsher form: it has ``text_bert`` but
**no** ``text``, so any pipeline keyed on ``text`` fails outright there.

This module is the single place where text is turned into a fixed-size vector,
so the mistake cannot silently reappear elsewhere.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

__all__ = [
    "MASK_CHANNEL",
    "WORD_ID_CHANNEL",
    "extract_attention_mask",
    "masked_mean_pool",
    "pool_text_sequence",
    "check_text_has_no_padding_zeros",
]

#: ``text_bert`` channel holding the 0/1 attention mask (verified on all splits).
MASK_CHANNEL = 1

#: ``text_bert`` channel holding BERT vocabulary ids.
WORD_ID_CHANNEL = 0


def extract_attention_mask(text_bert: np.ndarray) -> np.ndarray:
    """Return the (N, T) 0/1 attention mask from a ``text_bert`` array.

    Parameters
    ----------
    text_bert:
        Array of shape (N, C, T).  ``C`` is normally 3.

    Raises
    ------
    ValueError
        If no channel looks like a 0/1 mask, which would mean the dataset
        changed underneath us and every downstream assumption is suspect.
    """
    arr = np.asarray(text_bert)
    if arr.ndim != 3:
        raise ValueError(f"text_bert must be (N, C, T), got shape {arr.shape}")

    n, c, _ = arr.shape
    for ch in range(c):
        flat = arr[:, ch, :]
        uniq = np.unique(flat)
        if uniq.size <= 2 and set(np.round(uniq.astype(np.float64), 6)).issubset({0.0, 1.0}):
            return flat.astype(np.float32)

    raise ValueError(
        f"no 0/1 mask channel found in text_bert of shape {arr.shape}; "
        "expected one channel to hold the attention mask"
    )


def masked_mean_pool(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mean-pool ``values`` over the valid steps only.

    Parameters
    ----------
    values:
        (N, T, D) feature sequence.
    mask:
        (N, T) 0/1 validity mask.

    Returns
    -------
    (N, D) pooled vector.  A sample with zero valid steps yields zeros rather
    than NaN, so downstream code never silently propagates NaN.
    """
    v = np.asarray(values, dtype=np.float64)
    m = np.asarray(mask, dtype=np.float64)
    if v.ndim != 3:
        raise ValueError(f"values must be (N, T, D), got {v.shape}")
    if m.shape != v.shape[:2]:
        raise ValueError(f"mask {m.shape} does not match values {v.shape[:2]}")

    denom = np.maximum(m.sum(axis=1, keepdims=True), 1.0)
    pooled = (v * m[:, :, None]).sum(axis=1) / denom
    # samples with no valid steps: the denominator guard leaves zeros already
    return pooled.astype(np.float32)


def pool_text_sequence(
    text: np.ndarray | None,
    text_bert: np.ndarray | None,
    method: Literal["masked", "naive"] = "masked",
) -> np.ndarray:
    """Pool a text sequence to (N, D) using the correct route by default.

    ``method="naive"`` exists **only** so the ablation table can quantify the
    damage; it must never be used for the submitted pipeline.  Passing
    ``method="naive"`` emits a warning because the resulting features are known
    to differ from the correct ones by ~43% in relative L2 norm.

    Attachment 3 ships ``text_bert`` but no ``text``.  When ``text`` is missing
    this function raises rather than silently returning a wrong shape -- the
    caller must switch to the BERT encoder path instead.
    """
    if method == "masked":
        if text is None:
            raise ValueError(
                "text is None: attachment 3 provides only text_bert. "
                "Route this sample through the BERT encoder using the "
                "attention mask instead of pooling raw GloVe steps."
            )
        if text_bert is None:
            raise ValueError(
                "text_bert is required to build the attention mask; "
                "without it text padding cannot be detected (text has no zeros)"
            )
        mask = extract_attention_mask(text_bert)
        return masked_mean_pool(text, mask)

    if method == "naive":
        import warnings

        warnings.warn(
            "naive text pooling averages padded steps; verified to differ from "
            "masked pooling by ~43% relative L2 (max 112%). Use only for ablation.",
            RuntimeWarning,
            stacklevel=2,
        )
        if text is None:
            raise ValueError("naive pooling needs text, which is absent in attachment 3")
        arr = np.asarray(text, dtype=np.float64)
        return arr.mean(axis=1).astype(np.float32)

    raise ValueError(f"unknown method {method!r}; expected 'masked' or 'naive'")


def check_text_has_no_padding_zeros(text: np.ndarray) -> dict:
    """Assert the trap is still present, so the guard fails loudly if it changes.

    Returns a small dict for embedding in QC reports.
    """
    arr = np.asarray(text)
    zero_frac = float(np.all(arr == 0, axis=-1).mean())
    nz = arr[arr != 0]
    return {
        "all_zero_step_fraction": zero_frac,
        "nonzero_mean": float(nz.mean()) if nz.size else float("nan"),
        "nonzero_std": float(nz.std()) if nz.size else float("nan"),
        "trap_present": zero_frac == 0.0,
        "note": (
            "text contains no zeros, so padding is invisible to x==0 tests; "
            "the text_bert attention mask is mandatory"
        ),
    }
