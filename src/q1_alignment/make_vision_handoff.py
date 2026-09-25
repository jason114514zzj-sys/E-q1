"""Build the vision-timing handoff pack for the feature-extraction teammate.

Why this exists
---------------
``cv2.CAP_PROP_FRAME_COUNT`` reads the MP4 ``nb_frames`` metadata field, which
is wrong for 90 of the 100 competition clips: it reports more frames than the
decoder can actually deliver (e.g. 261 declared vs 163 decodable). Any
timestamp computed as ``frame_index / fps`` therefore lands too late, and the
error differs per clip (up to 3.3x), so it cannot be corrected afterwards.

This script exports the decoder's own per-frame timestamps so the vision
features can be stamped with real time instead of an estimate.

Output layout::

    <out>/_READ_ME_FIRST.md          prominent warning + usage
    <out>/frame_count_lookup.csv     per-clip declared vs real frame count
    <out>/vision_timing_manifest.csv one row per clip, all timing facts
    <out>/samples/<safe_id>/vision_timestamps.npy   [Tv] seconds
    <out>/samples/<safe_id>/vision_frame_meta.csv   frame_idx, pts_sec, ...
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

from .common import read_jsonl
from .ffprobe_utils import FFProbeError, find_ffprobe, read_timing

READ_ME = """# ⚠️ 视觉特征时间戳交付包 —— 请先读这一页

**面向：负责视觉特征提取的同学**
**来自：负责时序对齐的同学**
**日期：2026-09-23**

---

## 一、必须先知道的一件事

**这批视频不能用 `cv2.CAP_PROP_FRAME_COUNT` 取帧数，它是错的。**

实测 100 条：

| 项目 | 数值 |
|---|---|
| 帧数报错的样本 | **90 / 100 (90%)** |
| 时长报错的样本 | **87 / 100 (87%)** |
| 声明帧数 ÷ 真实帧数 | 最多差 **3.27 倍** |
| 平均偏差 | 约 **1.53 倍** |

### 具体例子

| 样本 | OpenCV 说 | 真实情况 | 差多少 |
|---|---|---|---|
| `-9y-fZ3swSY` clip 4 | 277 帧 / 9.23 秒 | **81 帧 / 2.82 秒** | 时间多 227% |
| `-s9qJ7ATP7w` clip 6 | 195 帧 / 6.50 秒 | **71 帧 / 2.48 秒** | 多 163% |
| `-3g5yACwYnA` clip 13 | 261 帧 / 8.70 秒 | **163 帧 / 5.40 秒** | 多 61% |

**注意：每条错的比例都不一样，事后没法用统一系数修正。**

---

## 二、为什么会这样

视频文件头部有个元数据字段 `nb_frames`，写着"我有 261 帧"。

- 这个字段是**制作视频时填写的**，属于**提示性信息**，不是准确值
- MOSEI 这些片段经过**剪辑和转码**，实际只保留了 163 帧，**但头部数字没更新**
- `cv2.CAP_PROP_FRAME_COUNT` **直接照抄这个字段**，所以被坑了

**不是你的错，也不是 OpenCV 的错 —— 是视频文件自己报的数字不准。**

### 最危险的地方

这个错误**不报错、不崩溃，数值看起来完全正常**：

```python
n = cap.get(cv2.CAP_PROP_FRAME_COUNT)   # 261
fps = cap.get(cv2.CAP_PROP_FPS)         # 30
duration = n / fps                      # 8.7 秒  ← 看着很合理，但是错的
```

只有**真的一帧一帧读**，才会发现读到第 163 帧就没了。

---

## 三、请这样使用本交付包

### 正确写法

```python
import numpy as np
import cv2

safe_id = "-3g5yACwYnA__13"
timestamps = np.load(f"samples/{safe_id}/vision_timestamps.npy")
# timestamps.shape == (163,)  ← 这就是真实帧数，直接用

cap = cv2.VideoCapture(video_path)
features, times, valid = [], [], []

i = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break                       # 读到没有为止，不要用 range(总帧数)
    if i >= len(timestamps):
        break                       # 保险：不该发生
    t = float(timestamps[i])        # ← 用交付包给的真实时间戳
    feat = extract_your_features(frame)
    features.append(feat)
    times.append(t)
    valid.append(1)                 # 没检测到人脸时置 0，但不要删这一帧
    i += 1
cap.release()

np.save("vision_features.npy",   np.asarray(features, dtype=np.float32))
np.save("vision_timestamps.npy", np.asarray(times,    dtype=np.float32))
np.save("vision_valid.npy",      np.asarray(valid,    dtype=np.uint8))
```

### ❌ 不要这样写

```python
# 错法 1：用声明的帧数
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))   # 261（错）
for i in range(total):                            # 循环到 164 就断了

# 错法 2：用帧号猜时间
t = frame_index / fps                             # 时间会偏大，最多偏 3.3 倍
```

---

## 四、文件说明

```
_READ_ME_FIRST.md              本文件
frame_count_lookup.csv         每条样本：声明帧数 vs 真实帧数（自查用）
vision_timing_manifest.csv     每条样本的全部时间信息
samples/<safe_id>/
    vision_timestamps.npy      [Tv] 每帧的真实时间（秒），float32  ← 主要交付物
    vision_frame_meta.csv      frame_idx, pts_sec, is_keyframe
```

### vision_timestamps.npy

- 形状 `[Tv]`，`Tv` 就是**真实可解码帧数**
- 单位秒，从 0.0 开始
- `timestamps[i]` 对应你按顺序 `read()` 出来的**第 i 帧**
- 直接当时间轴用，不要再用 `帧号 / fps` 算

### frame_count_lookup.csv 字段

| 字段 | 含义 |
|---|---|
| `id` | 样本编号 `video_id$_$clip_id` |
| `safe_id` | 文件名安全版本 `video_id__clip_id` |
| `declared_frames` | 视频头部声明的帧数（**不可信**） |
| `decoded_frames` | 实际能解码出的帧数（**用这个**） |
| `frames_match` | 两者是否一致 |
| `real_duration_sec` | 真实播放时长（秒） |

---

## 五、你需要做的自查

如果你**已经开始或完成了**视觉特征提取，请核对：

```python
# 你输出的视觉特征行数
your_rows = your_vision_features.shape[0]

# 交付包里的真实帧数
real_rows = np.load(f"samples/{safe_id}/vision_timestamps.npy").shape[0]

assert your_rows == real_rows, f"{safe_id}: {your_rows} vs {real_rows}"
```

**对不上就说明踩坑了，需要重跑。**

---

## 六、参考：协作技术文档原文

问题一协作技术文档第 34 行写着：

> 逐帧或按固定间隔提取表情、姿态或关键点特征，
> **并使用视频时间戳，而不是仅依赖帧号估算时间。**

**本交付包就是为此准备的真实时间戳。**
"""


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def build_vision_handoff(
    manifest_path: str | Path,
    data_root: str | Path,
    output_dir: str | Path,
    ffprobe: str | Path | None = None,
    limit: int | None = None,
) -> dict:
    """Export per-frame vision timestamps plus the warning documentation."""

    manifest = read_jsonl(manifest_path)
    if limit is not None:
        manifest = manifest[:limit]
    root = Path(data_root).resolve()
    out = Path(output_dir)
    samples_root = out / "samples"
    samples_root.mkdir(parents=True, exist_ok=True)

    try:
        ffprobe_exe = find_ffprobe(ffprobe)
    except FFProbeError as exc:
        raise SystemExit(f"ffprobe is required: {exc}") from exc

    lookup_rows: list[dict] = []
    manifest_rows: list[dict] = []

    for record in manifest:
        video_path = root / record["video_relpath"]
        safe = record["safe_id"]
        timing = read_timing(video_path, ffprobe_exe)

        sample_dir = samples_root / safe
        sample_dir.mkdir(parents=True, exist_ok=True)

        timestamps = np.asarray(timing.frame_times, dtype=np.float32)
        np.save(sample_dir / "vision_timestamps.npy", timestamps)

        _write_csv(
            sample_dir / "vision_frame_meta.csv",
            [
                {
                    "frame_idx": index,
                    "pts_sec": round(value, 6),
                    "frame_delta_sec": round(value - timestamps[index - 1], 6)
                    if index
                    else 0.0,
                }
                for index, value in enumerate(timing.frame_times)
            ],
        )

        lookup_rows.append({
            "id": record["id"],
            "safe_id": safe,
            "declared_frames": timing.frame_count_declared,
            "decoded_frames": timing.frame_count_decoded,
            "frames_match": timing.frame_count_declared == timing.frame_count_decoded,
            "frames_lost": timing.frame_count_declared - timing.frame_count_decoded,
            "real_duration_sec": round(timing.duration_sec, 4),
        })

        manifest_rows.append({
            "id": record["id"],
            "safe_id": safe,
            "real_duration_sec": round(timing.duration_sec, 4),
            "video_stream_duration": round(timing.video_stream_duration, 4),
            "audio_stream_duration": round(timing.audio_stream_duration, 4),
            "declared_frames": timing.frame_count_declared,
            "decoded_frames": timing.frame_count_decoded,
            "fps_declared": round(timing.fps_declared, 4),
            "last_frame_pts": round(timing.frame_times[-1], 4) if timing.frame_times else 0.0,
            "naive_duration": round(
                timing.frame_count_declared / timing.fps_declared, 4
            ) if timing.fps_declared else 0.0,
            "duration_ratio": round(timing.duration_ratio, 4),
        })

    _write_csv(out / "frame_count_lookup.csv", lookup_rows)
    _write_csv(out / "vision_timing_manifest.csv", manifest_rows)
    (out / "_READ_ME_FIRST.md").write_text(READ_ME, encoding="utf-8")

    mismatched = sum(1 for row in lookup_rows if not row["frames_match"])
    return {
        "samples": len(lookup_rows),
        "frame_mismatch": mismatched,
        "output_dir": str(out),
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ffprobe", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    result = build_vision_handoff(
        args.manifest, args.data_root, args.output_dir, args.ffprobe, args.limit
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
