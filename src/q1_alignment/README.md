# 问题一 时序对齐（q1_alignment）

数学建模竞赛 E 题 —— 多模态情感识别，问题一的时序对齐实现。

## 快速开始

```bash
# 服务器上
ssh user@example.invalid        # 已配置免密登录
conda activate mosei
cd ~/MathModel
export PYTHONPATH=~/MathModel/src
```

## ⚠️ 最重要的技术约束：视频时长必须用 ffprobe

本数据集（附件1 的 100 条 mp4）存在一个**隐蔽且严重**的问题：

> MP4 头部的 `nb_frames` 元数据字段**不可靠**。
> **100 条中 90 条虚报帧数**，合计虚报 9906 帧（30%）。

因此 `cv2.CAP_PROP_FRAME_COUNT ÷ FPS` 算出的时长是**错的**：

| 样本 | 声称 | 真实 | 偏差 |
|---|---|---|---|
| `-9y-fZ3swSY$_$4` | 9.23s / 277帧 | **2.82s / 81帧** | 时间虚高 227% |
| `-3g5yACwYnA$_$13` | 8.70s / 261帧 | **5.51s / 163帧** | 虚高 58% |
| `-3g5yACwYnA$_$3` | 19.27s / 578帧 | **14.39s / 431帧** | 虚高 34% |

**每条样本偏差比例都不同**（1.0 ~ 3.27 倍），事后无法用统一系数修正。

### 正确做法

```python
from q1_alignment.ffprobe_utils import read_timing

timing = read_timing(video_path)
timing.duration_sec        # 真实播放时长
timing.frame_times         # 逐帧真实时间戳（秒）
timing.frame_count_decoded # 真实帧数
timing.frame_count_declared # 元数据声称值（不可信，仅诊断用）
```

### ❌ 不要这样做

```python
cap = cv2.VideoCapture(video)
n = cap.get(cv2.CAP_PROP_FRAME_COUNT)   # 元数据，可能虚高 3 倍
fps = cap.get(cv2.CAP_PROP_FPS)
duration = n / fps                       # 错
```

> 这个错误**不报错、不崩溃、数值看着正常**，只有逐帧解码才会暴露。

## 目录结构

```
~/MathModel/
├── data/                     原始数据（只读）
├── src/q1_alignment/         本模块
│   ├── ffprobe_utils.py      ★ 真实时长/帧时间/逐帧PTS
│   ├── manifest.py           样本清单（已改用 ffprobe）
│   ├── forced_align.py       词级强制对齐（energy VAD / CTC）
│   ├── align_driver.py       批量对齐 + 容错 + 报告
│   ├── alignment.py          三模态汇总 + 50位置整理 + 四类掩码
│   ├── reports.py            ★ alignment_trace.csv / qc_report.csv
│   ├── make_vision_handoff.py 生成给队友的逐帧时间戳包
│   ├── audit_duration.py     100条时长审计
│   ├── timeline.py           时间轴校验 / CTM导入
│   ├── mfa.py                MFA 接口
│   └── tests/                216 个单元测试
├── work/                     中间产物
│   ├── manifest.jsonl        100条样本元数据（真实时长）
│   ├── word_timelines.jsonl  ★ 100条词级时间轴（98 条可发声 + 2 条 audio_absent）
│   ├── duration_audit.csv    时长审计结果
│   ├── alignment_trace.csv   ★ 逐位置追溯表
│   └── qc_report.csv         ★ 质量检查报告
├── handoff/vision_timing/    ★ 给队友的逐帧时间戳交付包
└── config.yaml / versions.txt
```

## 常用命令

```bash
cd ~/MathModel && export PYTHONPATH=~/MathModel/src

# 单元测试
python -m unittest discover -s src/q1_alignment/tests -t src

# 重建 manifest（真实时长）
python -m q1_alignment.cli build-manifest \
  --data-root "data/附件1-数据集原始多模态样本/MOSEI数据集部分原始视频-100条" \
  --output work/manifest.jsonl

# 词级强制对齐（100条约 9 秒）
python -m q1_alignment.cli align-transcripts \
  --manifest work/manifest.jsonl \
  --data-root "data/附件1-数据集原始多模态样本/MOSEI数据集部分原始视频-100条" \
  --output work/word_timelines.jsonl \
  --method energy \
  --report work/word_align_report.csv

# 时长审计（发现元数据问题）
python -m q1_alignment.audit_duration

# 生成队友交付包
python -m q1_alignment.make_vision_handoff \
  --manifest work/manifest.jsonl \
  --data-root "data/附件1-数据集原始多模态样本/MOSEI数据集部分原始视频-100条" \
  --output-dir handoff/vision_timing
```

## 对齐规则

- 以**逐词时间区间**为主时间轴
- 特征帧**中心**落入 `[start, end)` 时归入该位置
- 多帧取**均值**；空区间填零并将对应模态 mask 置 0
- 50 个位置中：位置 0 给 `[CLS]`，末位给 `[SEP]`，中间 48 个为内容位
- 不超过 48 词时每词一位；超过 48 词时按顺序**均匀合并**为 48 组（覆盖完整文本）
- 输出**四类掩码**，含义严格区分：
  - `sequence_mask` —— 是否属于真实序列位置（含 CLS/SEP）
  - `text_mask` / `audio_mask` / `vision_mask` —— 该模态在该位置是否有有效数据
- 额外输出 `vision_valid_ratio` —— 该位置内源级视觉有效帧占比，
  用于区分「没有人脸」与「数值恰好为0」

## 约束式单调对齐（v1.1.0）

### 为什么需要它

v1.0 的 `energy` 后端按「每个有声段内部独立分配词」实现。当有声段多而词数
相对少时，**早期段消耗完全部词，后续段分到 0 个词被跳过**，于是对齐只覆盖
了片段的前一小部分：

| 指标 | v1.0 energy | v1.1.0 monotonic |
|---|---|---|
| 覆盖率中位数 | 0.417 | **1.000** |
| 覆盖率最小值 | 0.075 | **1.000** |
| 覆盖率 <90% 的样本 | **83 / 100** | **0 / 100** |

最极端的例子 `-iRBcNs9oI8$_$9`：3.42 秒视频、5 个词，只覆盖 0.26 秒，
等价 **19.4 词/秒**——生理上不可能。

**这个缺陷通过了当时全部校验**（时间单调、数值有限、落在时长内），
说明「结构合法」不等于「语义正确」。

### 模型表述

在 `[0, D]` 上求词边界 `0 = b₀ < b₁ < … < b_T = D`，最小化

```
Σ_t  w_t · ( (b_t − b_{t−1}) / ŝ_t − 1 )²
  − λ · Σ_t  P(b_t)
```

- `ŝ_t` —— 词 `t` 的期望时长，由**音节数**（元音组计数，功能词衰减）加权分配
- `P(·)` —— 边界落在低能量（停顿）处的奖励项
- 硬约束：完整覆盖 `b₀=0, b_T=D`；严格单调 `b_t − b_{t−1} ≥ ε`；语速 ∈ [1, 6] 词/秒

### 停顿感知的边界吸附

边界在 ±80 ms 内寻找能量谷并吸附。**消融实验**（40 条样本）：

| 指标 | 关闭吸附 | 开启吸附 |
|---|---|---|
| 边界处归一化能量（越低越好） | 1.4687 | **1.1463** |
| 相对降低 | — | **21.95%** |
| 改善样本数 | — | **40 / 40（100%）** |

### 四级可核验 QC

`qc_report.csv` 对每条样本核验：

1. `shape_ok` —— 三模态序列长度一致
2. `time_monotonic` + `time_within_clip` —— 时间单调且不越界
3. **`coverage ≥ 0.90`** —— 新增，正对 v1.0 的失败模式
4. **`words_per_second ≤ 6.0`** —— 新增，生理合理性

## 队友交付接口

队友的视觉特征请配合 `handoff/vision_timing/` 使用：

```python
timestamps = np.load(f"samples/{safe_id}/vision_timestamps.npy")
# timestamps[i] 对应顺序 read() 出的第 i 帧，直接用作时间轴
```

详见该目录下的 `_READ_ME_FIRST.md`。

## 环境

- 服务器：server，conda 环境 `mosei`（Python 3.10.21）
- torch 2.4.1+cu121（V100 sm_70 上限版本）
- ffmpeg 7.1（conda），ffprobe 同目录

## 边界约定

- ❌ 不改动 `~/private_project`（另一个 IR 项目）
- ❌ 不改动 `REDACTED` 环境
- ❌ 原始数据只读，产物写入 `work/` 与 `output/`
