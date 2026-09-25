"""Probe: does the self-check tool actually validate all three modalities?

Constructs one handoff sample per modality failure mode and reports which
problems the tool catches, so we know whether it is genuinely tri-modal or
only effectively visual.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

TOOL = Path.home() / "MathModel" / "src" / "q1_alignment" / "handoff_selfcheck.py"

DURATION = 4.0


def base_sample(root: Path, sample_id: str = "v$_$1") -> Path:
    d = root / "samples" / sample_id
    d.mkdir(parents=True, exist_ok=True)
    n_words, n_audio, n_vision = 6, int(DURATION * 100), int(DURATION * 30)
    np.save(d / "text_features.npy", np.random.randn(n_words, 768).astype(np.float32))
    np.save(d / "audio_features.npy", np.random.randn(n_audio, 74).astype(np.float32))
    step = DURATION / n_audio
    np.save(d / "audio_intervals.npy",
            np.column_stack([np.arange(n_audio) * step, np.arange(n_audio) * step + step]))
    np.save(d / "vision_features.npy", np.random.randn(n_vision, 35).astype(np.float32))
    np.save(d / "vision_timestamps.npy", np.linspace(0, DURATION, n_vision, endpoint=False))
    np.save(d / "vision_valid.npy", np.ones(n_vision, dtype=np.uint8))
    np.save(d / "vision_confidence.npy", np.full(n_vision, 0.9, dtype=np.float32))
    with (d / "text_words.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["word_idx", "word"])
        for i in range(n_words):
            w.writerow([i, f"w{i}"])
    (d / "metadata.json").write_text(json.dumps({"sample_id": sample_id}), encoding="utf-8")
    return d


CASES = {
    "干净样本（三模态都正常）": lambda d: None,
    "文本-词数不符": lambda d: np.save(
        d / "text_features.npy", np.random.randn(99, 768).astype(np.float32)),
    "文本-含NaN": lambda d: _set_nan(d / "text_features.npy"),
    "文本-维度不是2D": lambda d: np.save(
        d / "text_features.npy", np.random.randn(6).astype(np.float32)),
    "语音-行数与时间轴不符": lambda d: np.save(
        d / "audio_intervals.npy", np.zeros((10, 2))),
    "语音-时间非递增": lambda d: _break_monotonic(
        d / "audio_intervals.npy", two_dim=True),
    "语音-含Inf": lambda d: _set_inf(d / "audio_features.npy"),
    "视觉-行数与时间戳不符": lambda d: np.save(
        d / "vision_timestamps.npy", np.linspace(0, DURATION, 7)),
    "视觉-时间非递增": lambda d: _break_monotonic(d / "vision_timestamps.npy"),
    "视觉-时间超出时长": lambda d: np.save(
        d / "vision_timestamps.npy", np.linspace(0, DURATION * 3, 120)),
    "视觉-含NaN": lambda d: _set_nan(d / "vision_features.npy"),
    "缺少语音时间文件": lambda d: (d / "audio_intervals.npy").unlink(),
    "缺少视觉特征文件": lambda d: (d / "vision_features.npy").unlink(),
    "缺少视觉valid": lambda d: (d / "vision_valid.npy").unlink(),
    "缺少text_words.csv": lambda d: (d / "text_words.csv").unlink(),
    "text_words.csv缺列": lambda d: _bad_words_csv(d),
}


def _set_nan(path):
    a = np.load(path)
    a.flat[0] = np.nan
    np.save(path, a)


def _set_inf(path):
    a = np.load(path)
    a.flat[0] = np.inf
    np.save(path, a)


def _break_monotonic(path, two_dim=False):
    a = np.load(path)
    if two_dim:
        a[5] = a[3]
    else:
        a[5] = a[3]
    np.save(path, a)


def _bad_words_csv(d):
    with (d / "text_words.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["idx", "token"])
        w.writerow([0, "a"])


def run_tool(root: Path) -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, str(TOOL), "--handoff", str(root)],
        capture_output=True, text=True,
    )
    return proc.returncode, proc.stdout + proc.stderr


def main() -> None:
    print("=" * 78)
    print("自检工具覆盖度实测：逐个体检查三模态")
    print("=" * 78)
    print()

    caught, missed = [], []
    for label, mutate in CASES.items():
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            d = base_sample(root)
            mutate(d)
            code, output = run_tool(root)
            detected = code != 0
            marker = "捕获" if detected else ("通过" if label.startswith("干净") else "!!! 漏检")
            print(f"[{marker:9s}] {label}")
            if label.startswith("干净"):
                if detected:
                    print("            -> 干净样本被误报为错误！")
                    print("            " + output.strip().replace("\n", "\n            ")[:400])
                else:
                    caught.append(label)
                continue
            if detected:
                message = [ln for ln in output.splitlines() if "✗" in ln]
                for m in message[:2]:
                    print(f"            -> {m.strip()}")
                caught.append(label)
            else:
                missed.append(label)

    print()
    print("=" * 78)
    print(f"覆盖: {len(caught)} / {len(CASES)}")
    if missed:
        print("漏检项:")
        for m in missed:
            print(f"  - {m}")
    else:
        print("全部用例均被正确处理")
    print()
    print("模态维度覆盖统计（从用例看）:")
    for mod in ("文本", "语音", "视觉", "文件完整性", "干净"):
        relevant = [c for c in CASES if c.startswith(mod) or (mod == "干净" and c.startswith("干净"))]
        ok = [c for c in relevant if c in caught]
        print(f"  {mod:8s} {len(ok)}/{len(relevant)}")


if __name__ == "__main__":
    main()
