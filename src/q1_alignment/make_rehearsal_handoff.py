"""Build a realistic 3-sample feature handoff to rehearse pipeline steps 4-8.

Why this matters
----------------
Steps 4-8 of run_q1_pipeline.sh (import -> align -> report -> summary -> figures)
have NEVER executed.  The only run so far stopped at step 3 because no teammate
handoff existed.  So the entire downstream half of the pipeline is untested
against real inputs -- if it has a bug, we discover it the moment the teammate
delivers, which is the worst possible time.

This script generates a handoff directory in the exact document format the
contract specifies (9 named files per sample), for three deliberately chosen
samples: the shortest clip, the longest clip, and a clip with a degraded
audio track.  That mirrors the "先交 3 条联调" advice in the interface spec.

Output goes to a scratch directory; the real handoff/ is untouched.
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/user/MathModel")
WORK = ROOT / "work"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "work" / "rehearsal_handoff"

REQUIRED = (
    "metadata.json",
    "text_words.csv",
    "text_features.npy",
    "audio_features.npy",
    "audio_intervals.npy",
    "vision_features.npy",
    "vision_timestamps.npy",
    "vision_valid.npy",
    "vision_confidence.npy",
)

TEXT_DIM = 768
AUDIO_DIM = 74
VISION_DIM = 35


def load_manifest() -> list[dict]:
    return [json.loads(l) for l in
            (WORK / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()]


def pick_samples(recs: list[dict]) -> list[tuple[str, dict]]:
    """Shortest, longest, and a middle clip -- the order the spec recommends."""
    by_dur = sorted(recs, key=lambda r: r["duration_sec"])
    picks = [
        ("shortest", by_dur[0]),
        ("longest", by_dur[-1]),
        ("middle", by_dur[len(by_dur) // 2]),
    ]
    return picks


def build_sample(sample: dict, tag: str, rng: np.random.Generator) -> dict:
    """Write the 9 contract files for one sample. Returns a status dict."""
    safe = sample["safe_id"]
    d = OUT / "samples" / safe
    d.mkdir(parents=True, exist_ok=True)

    duration = float(sample["duration_sec"])
    words = sample.get("words") or []
    if not words:
        # fall back to the manifest's transcript words if timelines lack them
        words = [{"word": w, "start": i * duration / 5, "end": (i + 1) * duration / 5}
                 for i, w in enumerate(sample.get("transcript", "a b c d e".split()))]
    n_words = max(len(words), 1)

    # ---- text: one row per word ----
    text_feat = rng.normal(size=(n_words, TEXT_DIM)).astype(np.float32)
    np.save(d / "text_features.npy", text_feat)
    with (d / "text_words.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["word_idx", "word"])
        for i, tok in enumerate(words):
            w.writerow([i, tok.get("word", f"w{i}")])

    # ---- audio: ~10 ms hop, but deliberately COARSE for the degraded sample ----
    hop = 0.05 if tag != "longest" else 0.02
    n_audio = max(int(duration / hop), 2)
    audio_feat = rng.normal(size=(n_audio, AUDIO_DIM)).astype(np.float32)
    audio_feat[:, 0] = np.abs(audio_feat[:, 0]) * 100.0   # mimic dim0's scale
    intervals = np.stack([
        np.arange(n_audio) * hop,
        np.minimum((np.arange(n_audio) + 1) * hop, duration),
    ], axis=1).astype(np.float64)

    # simulate a damaged stretch: zeros in the middle for the "shortest" sample
    audio_valid = np.ones(n_audio, dtype=np.float32)
    if tag == "shortest" and n_audio > 6:
        lo, hi = n_audio // 3, 2 * n_audio // 3
        audio_feat[lo:hi] = 0.0
        audio_valid[lo:hi] = 0.0

    np.save(d / "audio_features.npy", audio_feat)
    np.save(d / "audio_intervals.npy", intervals)

    # ---- vision: timestamps strictly increasing, some invalid frames kept ----
    n_vis = max(int(duration * 30), 2)
    ts = np.linspace(0.0, max(duration - 1e-3, 0.0), n_vis)
    ts = np.maximum.accumulate(ts)
    # enforce strict increase
    ts = ts + np.arange(n_vis) * 1e-6
    vision_feat = rng.normal(size=(n_vis, VISION_DIM)).astype(np.float32)
    vision_valid = np.ones(n_vis, dtype=np.float32)
    if n_vis > 10:
        vision_valid[::7] = 0.0            # frames with no detected face
    conf = np.where(vision_valid > 0, rng.uniform(0.5, 1.0, n_vis), 0.0).astype(np.float32)
    conf = conf.astype(np.float32)

    np.save(d / "vision_features.npy", vision_feat)
    np.save(d / "vision_timestamps.npy", ts.astype(np.float64))
    np.save(d / "vision_valid.npy", vision_valid)
    np.save(d / "vision_confidence.npy", conf)

    meta = {
        "sample_id": sample["id"],
        "safe_id": safe,
        "tag": tag,
        "duration_sec": duration,
        "n_words": n_words,
        "n_audio_frames": n_audio,
        "n_vision_frames": n_vis,
        "audio_hop_sec": hop,
        "audio_valid_frames": int(audio_valid.sum()),
        "vision_valid_frames": int(vision_valid.sum()),
        "notes": ("audio track degraded in the middle third"
                  if tag == "shortest" else "synthetic rehearsal data"),
    }
    (d / "metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    written = sorted(p.name for p in d.iterdir())
    return {
        "safe_id": safe, "tag": tag, "duration": duration,
        "n_words": n_words, "n_audio": n_audio, "n_vis": n_vis,
        "missing": [f for f in REQUIRED if f not in written],
    }


def main() -> int:
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "samples").mkdir(parents=True)

    recs = load_manifest()
    rng = np.random.default_rng(20260924)
    picks = pick_samples(recs)

    print("=" * 74)
    print(f"编排 3 条合成交付样本 -> {OUT}")
    print("=" * 74)
    rows = []
    for tag, sample in picks:
        rows.append(build_sample(sample, tag, rng))

    # top-level manifest.csv so import-handoff has something to read
    with (OUT / "manifest.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sample_id", "safe_id", "duration_sec"])
        for r in rows:
            w.writerow([r["safe_id"].replace("__", "$_$"), r["safe_id"], r["duration"]])

    for r in rows:
        status = "OK" if not r["missing"] else f"缺 {r['missing']}"
        print(f"  [{status:12s}] {r['safe_id']:22s} tag={r['tag']:9s} "
              f"dur={r['duration']:6.3f}s words={r['n_words']:3d} "
              f"audio={r['n_audio']:5d} vision={r['n_vis']:5d}")

    print(f"\n共 {len(rows)} 条样本，每条 9 个文件")
    total = sum(1 for _ in (OUT / "samples").rglob("*") if _.is_file())
    print(f"实际写入文件数: {total + 1}（含 manifest.csv）")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
