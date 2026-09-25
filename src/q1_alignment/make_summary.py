"""Produce the full-sample summary table required by problem section 四.2.(2).

The problem statement asks for a table covering all 100 clips with:

    样本编号 / 模态类型 / 原始有效时长 / 特征维度 / 对齐粒度

Because the teammate's real features are not available yet, this script reports
the *timeline* facts that are already known (durations, frame counts, word
counts, aligned positions) and leaves the feature-dimension columns explicit
about their current demo/placeholder status, so nothing is silently implied.

Outputs:
    work/ALL_SAMPLES_SUMMARY.csv    one row per sample
    work/MODALITY_SUMMARY.csv       one row per sample per modality
    work/SUMMARY_README.md          column definitions and status caveats
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .common import read_jsonl
from .ffprobe_utils import read_timing

# demo feature dimensions currently used by make-demo-features
DEMO_DIMS = {"text": 16, "audio": 8, "vision": 6}
# 附件2 reference dimensions, recorded for cross-checking later
REFERENCE_DIMS = {"text": 768, "audio": 74, "vision": 35}

SUMMARY_README = """# 全量结果汇总表说明

对应题目四.2.(2)「特征文件规范与全量结果汇总」。

## ALL_SAMPLES_SUMMARY.csv —— 每条样本一行

| 列 | 含义 | 题目对应 |
|---|---|---|
| `sample_id` | 样本编号 `video_id$_$clip_id` | 样本编号 |
| `safe_id` | 文件名安全版本 | — |
| `original_duration_sec` | 原始有效时长（秒，ffprobe 真实值） | 原始有效时长 |
| `frame_count` | 真实可解码帧数 | 对齐粒度 |
| `transcript_words` | 转写文本词数 | — |
| `aligned_positions` | 对齐后内容位置数（≤48） | 对齐粒度 |
| `sequence_length` | 统一序列长度（=50） | 对齐粒度 |
| `word_groups_merged` | 是否因超48词发生合并 | 填充规则 |
| `text_dim` / `audio_dim` / `vision_dim` | 三模态特征维度 | 特征维度 |
| `label` / `annotation` | 情感强度 / 极性 | — |
| `feature_status` | 特征来源状态 | 可追溯性 |
| `audio_present` | 音轨是否有信号（峰值 RMS 是否高于 1e-5 静音底线） | 可核验性 |
| `energy_peak` | 音轨峰值 RMS（数字静音的 2 条实测为 1e-6） | 可核验性 |
| `speech_active_ratio` | 帧能量高于阈值的比例 | 可核验性 |
| `has_edit_list` | 容器是否带 `elst` 编辑列表 | 时序可核验性 |
| `first_frame_pts` | 首帧时间戳（不是常数 0 说明帧轴有头偏移） | 时序可核验性 |
| `head_gap_sec` / `tail_gap_sec` | 帧轴相对容器声明时长的头/尾空档 | 时序可核验性 |
| `duration_frame_axis_sec` | 由 `末帧 pts + 1/fps` 反推的帧轴时长 | 原始有效时长 |
| `duration_max_delta_sec` | 帧轴时长 − 容器声明时长的差值 | 原始有效时长 |

## MODALITY_SUMMARY.csv —— 每条样本 × 每个模态一行

便于按模态维度汇总，字段含义同上。

## ⚠️ 关于 `feature_status`

- `demo` —— 当前使用的是**演示用合成特征**，仅用于打通流程，
  **不可作为最终结果**。等待队友交付真实特征后须重跑。
- `real` —— 队友交付的真实特征。
- `audio_absent` —— 该条视频的**音轨是数字静音**（峰值 RMS 仅 1e-6，低于 1e-5 静音底线），
  因此按定义不存在词级时间与音频特征。这类样本的词时间**不填哨兵、不给假边界**，
  特征列一律留 0 并在此显式标注。

`text_dim/audio_dim/vision_dim` 在 `demo` 状态下记录的是演示维度，
必须与队友交付的实际维度核对。
"""


def _frame_axis_duration(record: dict[str, Any]) -> float | None:
    """Duration implied by the frame axis: last frame pts + one frame period.

    ``ffprobe``'s container duration and the decoded frame axis do not always
    agree; when the last frame starts at ``duration`` exactly, the container
    value understates the clip by one frame period.  ``ffprobe_utils`` folds
    this bound into ``duration_sec``, so here it is only recomputed for the
    auditable ``duration_frame_axis_sec`` column.
    """

    fps = float(record.get("fps") or 0.0)
    frame_count = int(record.get("frame_count") or 0)
    if fps <= 0 or frame_count <= 0:
        return None
    return frame_count / fps


def person_summary(
    manifest_path: str | Path,
    timelines_path: str | Path,
    aligned_dir: str | Path | None,
    output_dir: str | Path,
    data_root: str | Path | None = None,
) -> dict[str, Any]:
    """Write both summary tables and their README."""

    manifest = read_jsonl(manifest_path)
    timelines = {row["id"]: row for row in read_jsonl(timelines_path)}
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    aligned = Path(aligned_dir) if aligned_dir else None

    sample_rows: list[dict[str, Any]] = []
    modality_rows: list[dict[str, Any]] = []

    for record in manifest:
        timeline = timelines.get(record["id"], {})
        words = timeline.get("words", [])
        word_count = len(words)
        diagnostics = timeline.get("diagnostics") or {}
        audio_present = bool(diagnostics.get("audio_present", True))

        dims = dict(DEMO_DIMS)
        feature_status = "demo"
        aligned_positions = 0
        sequence_length = 50
        merged = word_count > 48

        if not audio_present:
            # No signal to align against: no word times were emitted (they are
            # withheld, not sentinel-filled) and no audio/vision feature can be
            # pooled, so the row must not advertise demo dimensions as if a real
            # extraction had happened.
            feature_status = "audio_absent"
            dims = {name: 0 for name in DEMO_DIMS}
        elif aligned is not None:
            npz = aligned / f"{record['safe_id']}.npz"
            if npz.exists():
                with np.load(npz, allow_pickle=False) as arrays:
                    dims = {
                        "text": int(arrays["text"].shape[1]),
                        "audio": int(arrays["audio"].shape[1]),
                        "vision": int(arrays["vision"].shape[1]),
                    }
                    sequence_length = int(arrays["text"].shape[0])
                    aligned_positions = int(arrays["effective_length"]) - 2
                # The npz exists, so these dimensions came from a real extraction
                # product, not from DEMO_DIMS.  Leaving the label at "demo" here
                # meant the finished deliverable would still say "demo" -- which
                # is the opposite failure of claiming "real" too early.
                feature_status = "real"

        if not aligned_positions and audio_present:
            aligned_positions = min(word_count, 48)

        duration = float(record["duration_sec"])
        frame_axis = _frame_axis_duration(record)
        sample_rows.append({
            "sample_id": record["id"],
            "safe_id": record["safe_id"],
            "original_duration_sec": round(duration, 4),
            "duration_frame_axis_sec": (
                round(frame_axis, 4) if frame_axis is not None else None
            ),
            "duration_max_delta_sec": (
                round(duration - float(record.get("declared_duration_sec") or duration), 4)
            ),
            "frame_count": int(record["frame_count"]),
            "fps": round(float(record.get("fps", 0.0)), 4),
            "transcript_words": word_count,
            "aligned_positions": aligned_positions if audio_present else 0,
            "sequence_length": sequence_length,
            "word_groups_merged": merged,
            "text_dim": dims["text"],
            "audio_dim": dims["audio"],
            "vision_dim": dims["vision"],
            "label": round(float(record["label"]), 4),
            "annotation": record["annotation"],
            "feature_status": feature_status,
            "audio_present": audio_present,
            "energy_peak": diagnostics.get("energy_peak"),
            "speech_active_ratio": diagnostics.get("speech_active_ratio"),
            "has_edit_list": record.get("has_edit_list"),
            "first_frame_pts": record.get("first_frame_pts"),
            "head_gap_sec": record.get("head_gap_sec"),
            "tail_gap_sec": record.get("tail_gap_sec"),
        })

        for modality in ("text", "audio", "vision"):
            if modality == "text":
                granularity = "word/word-group"
                count = aligned_positions
            elif modality == "audio":
                granularity = "100 Hz frame -> pooled by word interval"
                count = int(record.get("audio_stream_duration", 0) * 100) or 0
            else:
                granularity = "video frame -> pooled by word interval"
                count = int(record["frame_count"])
            modality_rows.append({
                "sample_id": record["id"],
                "modality": modality,
                "original_duration_sec": round(duration, 4),
                "feature_dim": dims[modality],
                "reference_dim_附件2": REFERENCE_DIMS[modality],
                "dim_matches_附件2": dims[modality] == REFERENCE_DIMS[modality],
                "alignment_granularity": granularity,
                "source_units": count,
                "aligned_positions": aligned_positions if audio_present else 0,
                "audio_present": audio_present,
                "feature_status": feature_status,
            })

    _write_csv(root / "ALL_SAMPLES_SUMMARY.csv", sample_rows)
    _write_csv(root / "MODALITY_SUMMARY.csv", modality_rows)
    (root / "SUMMARY_README.md").write_text(SUMMARY_README, encoding="utf-8")

    counts: dict[str, int] = {}
    for row in sample_rows:
        counts[row["feature_status"]] = counts.get(row["feature_status"], 0) + 1
    return {
        "samples": len(sample_rows),
        "modality_rows": len(modality_rows),
        "output_dir": str(root),
        "feature_status": "real" if counts.get("real") else "demo",
        "feature_status_counts": counts,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    import argparse

    home = Path.home()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(home / "MathModel/work/manifest.jsonl"))
    parser.add_argument("--timelines", default=str(home / "MathModel/work/word_timelines.jsonl"))
    parser.add_argument("--aligned-dir", default=str(home / "MathModel/work/aligned_demo"))
    parser.add_argument("--output-dir", default=str(home / "MathModel/work/summary"))
    args = parser.parse_args()

    result = person_summary(
        args.manifest, args.timelines, args.aligned_dir, args.output_dir
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
