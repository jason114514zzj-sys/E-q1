"""交付格式自检 —— 请在把特征发给对齐同学之前运行。

设计原则
--------
每条问题都同时给出三样东西，让你不用猜：

1. **是什么** —— 具体哪个文件、哪一行、哪个数值
2. **为什么错** —— 为什么这样会出问题
3. **怎么改** —— 可直接照做的修复办法

问题分三级
----------
``错误``  必须修复，否则对齐程序会失败或结果不可信（退出码 1）
``缺失``  必需文件没交，同样必须补上（退出码 1）
``警告``  不阻断流程，但会影响论文中"可追溯性"，建议修复（退出码 0）

用法
----
    python handoff_selfcheck.py --handoff /path/to/q1_feature_handoff

    # 只看前 3 条（联调阶段用）
    python handoff_selfcheck.py --handoff ... --limit 3

    # 结果同时写入 JSON
    python handoff_selfcheck.py --handoff ... --json check.json

本脚本只依赖 NumPy 和标准库，可以拷到你自己的环境里直接运行。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# --------------------------------------------------------------------------
# 交付要求
# --------------------------------------------------------------------------
# 表格列：文件名 / 规格 / 用途说明 / 缺失后果
REQUIRED_FILES: dict[str, tuple[str, str, str]] = {
    "text_features.npy": (
        "(Tt, Dt) float32",
        "逐词文本特征；一个词拆成多个子词时对子向量取平均",
        "文本模态无法对齐",
    ),
    "audio_features.npy": (
        "(Ta, Da) float32",
        "逐帧低层声学特征（音高/能量/MFCC/频谱）",
        "语音模态无法对齐",
    ),
    "audio_intervals.npy": (
        "(Ta, 2) float64",
        "每帧 [开始秒, 结束秒]，从 0 开始",
        "无法知道每个音频帧的时间位置",
    ),
    "vision_features.npy": (
        "(Tv, Dv) float32",
        "逐帧视觉特征（表情/姿态/关键点）",
        "视觉模态无法对齐",
    ),
    "vision_timestamps.npy": (
        "(Tv,) float64",
        "每帧时间戳（秒），从 0 开始且严格递增",
        "无法知道每个视觉帧的时间位置",
    ),
    # 以下两项由"建议"升为"必需"：题目要求可回溯文本、可区分人脸检测失败
    "text_words.csv": (
        "表格，至少 word_idx 与 word 两列",
        "逐词记录，用于把对齐结果回溯到原始文本",
        "论文无法展示「文本片段」与特征位置的对应关系",
    ),
    "vision_valid.npy": (
        "(Tv,) 0/1",
        "源级有效标记；未检出人脸时置 0，但保留该帧",
        "无法区分「没有人脸」与「特征恰好为 0」",
    ),
}

RECOMMENDED_FILES: dict[str, tuple[str, str, str]] = {
    "vision_confidence.npy": (
        "(Tv,) float32",
        "人脸检测置信度",
        "无法按置信度加权，弱检测帧与强检测帧同等对待",
    ),
    "metadata.json": (
        "JSON",
        "该样本的文件信息与异常说明",
        "异常情况缺少书面记录",
    ),
    "audio_valid.npy": (
        "(Ta,) 0/1",
        "语音逐帧有效性；该帧不可用时置 0，但保留该帧与时间戳",
        "实测：不提供时，一段 80% 全零的区间会被当成正常数据平均，"
        "数值错 80% 且不报错、不像异常。题目要求说明「有效长度」与「填充规则」，"
        "缺了它这两条写不清楚",
    ),
    "text_valid.npy": (
        "(Tt,) 0/1",
        "文本逐词有效性；该词不可用时置 0",
        "同理，缺了它无法区分「这个词本来就不存在」与「特征恰好为 0」",
    ),
}

MANIFEST_FILES: dict[str, str] = {
    "manifest.csv": "样本总表：样本ID、时长、fps、三模态行数与维度、处理状态",
    "config.yaml": "本次提取使用的统一配置（模型、参数、特征列）",
    "versions.txt": "工具与模型的版本号",
    "run_log.csv": "逐样本处理日志，含失败样本与原因",
}


@dataclass
class Finding:
    """一条检查结果，自带纠正指导。"""

    level: str          # 错误 / 缺失 / 警告
    what: str           # 是什么
    why: str            # 为什么有问题
    how: str            # 怎么改


@dataclass
class SampleReport:
    sample_id: str
    findings: list[Finding] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.level in ("错误", "缺失")]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.level == "警告"]

    @property
    def ok(self) -> bool:
        return not self.errors


# --------------------------------------------------------------------------
# 单项检查
# --------------------------------------------------------------------------
def _load(directory: Path, name: str, report: SampleReport) -> np.ndarray | None:
    try:
        return np.load(directory / name, allow_pickle=False)
    except Exception as exc:
        report.findings.append(Finding(
            "错误",
            f"{name} 无法读取：{exc}",
            "文件损坏或不是有效的 .npy 格式。",
            f"用 numpy 重新保存：np.save('{name}', 你的数组)；"
            "确认保存时用的是 np.save 而不是 np.savez 或 pickle。",
        ))
        return None


def _check_features(name: str, array: np.ndarray, report: SampleReport) -> None:
    if array.ndim != 2:
        report.findings.append(Finding(
            "错误",
            f"{name} 应为 2 维 [T, D]，实际是 {array.shape}",
            "对齐程序按「行=时间步、列=特征维」组织数据，一维或三维都无法处理。",
            f"如果 {array.ndim} 维是因为只有一个时间步，用 arr.reshape(1, -1)；"
            "如果是把时间步和特征维弄反了，用 arr.reshape(T, D) 转成二维。",
        ))
        return
    if not np.issubdtype(array.dtype, np.number):
        report.findings.append(Finding(
            "错误",
            f"{name} 不是数值类型（{array.dtype}）",
            "对齐需要做均值聚合，非数值无法计算。",
            "用 astype(np.float32) 转成数值数组。",
        ))
        return
    if not np.isfinite(array).all():
        mask = ~np.isfinite(array)
        bad = int(mask.sum())
        rows = np.flatnonzero(mask.any(axis=1))[:5].tolist()
        where = f"，涉及第 {rows} 行" if rows else ""
        if len(rows) < bad:
            where += f" 等（共 {bad} 处）"
        report.findings.append(Finding(
            "错误",
            f"{name} 含 NaN 或 Inf{where}",
            "NaN/Inf 会污染后续的均值聚合，并让整条样本的对齐结果失效。",
            "把这些位置改成 0，并配合 valid 标记说明该处无效；"
            "不要用插值填充，也不要把这一帧删掉——"
            "题目要求保留源级失败位置。",
        ))
    if array.dtype != np.float32:
        report.findings.append(Finding(
            "警告",
            f"{name} 的 dtype 是 {array.dtype}，建议 float32",
            "统一 float32 可减小体积并避免精度不一致。",
            f"保存时转一下：np.save('{name}', arr.astype(np.float32))",
        ))
    _check_not_padded(name, array, report)


def _check_not_padded(name: str, array: np.ndarray, report: SampleReport) -> None:
    """Catch the likely misunderstanding of padding arrays to 50 rows.

    50-position layout and padding are the aligner's job, not the producer's.
    A producer that pads to 50 destroys the boundary between "padding" and
    "genuinely zero feature", which the problem statement requires us to keep
    distinguishable.
    """

    if array.ndim != 2 or array.shape[0] != 50:
        return
    # Heuristic: a large block of all-zero rows at the tail is the signature
    # of manual padding rather than 50 real time steps.
    zero_rows = np.all(array == 0, axis=1)
    if not zero_rows.any():
        return
    tail = int(np.count_nonzero(np.cumprod(zero_rows[::-1])))
    if tail >= 5:
        report.findings.append(Finding(
            "警告",
            f"{name} 恰好 50 行，且末尾 {tail} 行全为 0，疑似已手动补齐到 50 位",
            "50 位置整理与 Padding 由对齐阶段负责，不属于交付内容。"
            "提前补零会让「填充」与「特征本身就是 0」无法区分，"
            "题目要求这两种情况必须能分开核对。",
            "请改为交付原始粒度：这个词/帧有多少就存多少行。"
            "例如 15 个词就存 (15, D)。若该数组确实是 50 个真实时间步，"
            "可忽略本警告。",
        ))


def _check_time_axis(
    name: str,
    array: np.ndarray,
    report: SampleReport,
    duration: float | None,
) -> None:
    if array.ndim == 2:
        if array.shape[1] != 2:
            report.findings.append(Finding(
                "错误",
                f"{name} 应为 [T, 2]（开始、结束两列），实际是 {array.shape}",
                "时间区间需要成对给出，列数不对就无法解析。",
                "用 np.column_stack([starts, ends]) 组织成两列。",
            ))
            return
        if np.any(array[:, 0] > array[:, 1]):
            bad = int((array[:, 0] > array[:, 1]).sum())
            report.findings.append(Finding(
                "错误",
                f"{name} 有 {bad} 行「开始时间 > 结束时间」",
                "区间起点晚于终点，时间轴自相矛盾。",
                "检查这些行的生成逻辑，必要时用 np.minimum/np.maximum "
                "把两列纠正为升序对。",
            ))
        centers = array.mean(axis=1)
    elif array.ndim == 1:
        centers = array
    else:
        report.findings.append(Finding(
            "错误",
            f"{name} 应为 1 维或 [T, 2]，实际是 {array.shape}",
            "形状不符合约定的两种形式。",
            "时间戳用一维数组；时间区间用 [T, 2]。",
        ))
        return

    if not np.isfinite(centers).all():
        report.findings.append(Finding(
            "错误",
            f"{name} 的时间值含 NaN 或 Inf",
            "时间无法参与比较与排序。",
            "检查时间戳的计算过程，确保没有除以 0 或对空数组取值。",
        ))
        return

    if centers.size > 1 and np.any(np.diff(centers) < -1e-9):
        bad = int((np.diff(centers) < -1e-9).sum())
        report.findings.append(Finding(
            "错误",
            f"{name} 时间非递增（{bad} 处回退）",
            "对齐假设帧按时间先后排列，回退会导致区间归属错乱。",
            "若顺序被打乱过，用 order = np.argsort(centers) 同步重排"
            "特征数组与时间数组；若本应递增却出现回退，检查帧号是否重复。",
        ))

    if centers.size and centers.min() < -0.05:
        report.findings.append(Finding(
            "错误",
            f"{name} 存在负时间（最小 {centers.min():.3f} 秒）",
            "视频开头统一记为 0 秒，负值说明时间基准不对。",
            "把整个时间轴平移，使第一帧为 0：centers = centers - centers.min()",
        ))

    if duration and centers.size:
        overrun = float(centers.max()) - duration
        ratio = overrun / duration if duration > 0 else 0
        if ratio > 0.30:
            report.findings.append(Finding(
                "错误",
                f"{name} 最大时间 {centers.max():.3f}s 超出视频时长 {duration:.3f}s"
                f"（超出 {overrun:.3f}s，达 {ratio*100:.0f}%）",
                "严重越界通常意味着时间戳用了错误的帧率或错误的时长基准。"
                "本数据集 MP4 头部元数据不可靠，用它算时长会虚高很多。",
                "改用 ffprobe 的真实时间：\n"
                "            ffprobe -select_streams v:0 -show_entries frame=pts_time \\\n"
                "                    -of json 你的视频.mp4\n"
                "            若已有时间戳，检查是否误用了 cv2.CAP_PROP_FRAME_COUNT/FPS。",
            ))
        elif overrun > 0.05:
            report.findings.append(Finding(
                "警告",
                f"{name} 最大时间 {centers.max():.3f}s 略超时长 {duration:.3f}s"
                f"（超出 {overrun:.3f}s）",
                "小幅超出一般是帧时长取整造成，通常无碍。",
                "若超出超过一帧时长，建议核对时间戳的来源。",
            ))


def _check_text_words(directory: Path, report: SampleReport, text_rows: int | None) -> None:
    path = directory / "text_words.csv"
    if not path.exists():
        return  # 缺失已在必需文件检查中报告
    try:
        with path.open(encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError("文件为空")
            missing = {"word_idx", "word"} - set(reader.fieldnames)
            rows = list(reader)
    except Exception as exc:
        report.findings.append(Finding(
            "错误",
            f"text_words.csv 解析失败：{exc}",
            "文件不是有效的 CSV，或编码不对。",
            "用 UTF-8 编码保存，第一行为表头 word_idx,word。",
        ))
        return

    if missing:
        report.findings.append(Finding(
            "错误",
            f"text_words.csv 缺少列 {sorted(missing)}"
            f"（当前列：{reader.fieldnames}）",
            "对齐追溯需要 word_idx 定位、word 展示。",
            "表头改为：word_idx,word（可再加 start/end 等列）。",
        ))
        return

    if not rows:
        report.findings.append(Finding(
            "警告",
            "text_words.csv 没有数据行",
            "该样本疑似空转写。",
            "确认该视频是否真的没有语音；若是无效样本，"
            "请在 manifest 与 run_log 中记录，而不是静默留空。",
        ))
        return

    report.stats["word_count"] = len(rows)

    if text_rows is not None and len(rows) != text_rows:
        report.findings.append(Finding(
            "警告",
            f"text_words.csv 有 {len(rows)} 行，text_features 有 {text_rows} 行，不一致",
            "通常是「保留子词级向量」造成的：一个词被分词器拆成多个子词，"
            "于是特征行数多于词数。这在文档里是允许的，但需要明确。",
            "二选一：\n"
            "            (a) 按文档对子词向量取平均，使行数等于词数；\n"
            "            (b) 保留子词级，并在 text_words.csv 增加一列 "
            "subword_idx 说明每个子词属于哪个词。\n"
            "            若两者都不是，说明切词逻辑有 bug，请检查。",
        ))


def _check_valid_mask(name: str, directory: Path, report: SampleReport,
                      expected: int | None) -> None:
    path = directory / name
    if not path.exists():
        return
    array = _load(directory, name, report)
    if array is None:
        return
    if array.ndim != 1:
        report.findings.append(Finding(
            "错误",
            f"{name} 应为 1 维 [T]，实际是 {array.shape}",
            "有效标记必须与时间轴一一对应。",
            "用 array.reshape(-1) 展平。",
        ))
        return
    if expected is not None and array.shape[0] != expected:
        report.findings.append(Finding(
            "错误",
            f"{name} 长度 {array.shape[0]} 与对应时间轴长度 {expected} 不一致",
            "有效标记必须与时间轴逐帧对应，长度不同就无法正确配对。",
            "确保两者用同一个循环生成，或检查是否漏了某一帧。",
        ))
    if not np.isin(array, [0, 1]).all():
        report.findings.append(Finding(
            "警告",
            f"{name} 含 0/1 以外的取值",
            "下游按布尔语义解释该数组，非 0/1 会被当作 True。",
            f"若是概率，请改用 (array >= 0.5).astype(np.uint8)；"
            f"若确实是 0/1，忽略本警告。",
        ))

    zeros = int((array == 0).sum())
    if zeros:
        report.stats[f"{name}_zero_count"] = zeros


# --------------------------------------------------------------------------
# 单样本检查
# --------------------------------------------------------------------------
def check_sample(directory: Path, duration: float | None = None,
                 frame_count: int | None = None) -> SampleReport:
    report = SampleReport(sample_id=directory.name)

    # ---- 必需文件 ----
    for name, (spec, purpose, consequence) in REQUIRED_FILES.items():
        if not (directory / name).exists():
            report.findings.append(Finding(
                "缺失",
                f"缺少必需文件 {name}（规格 {spec}）",
                f"{purpose}。缺了它，{consequence}。",
                f"请补齐 {name}。如果这个模态在该样本确实不可用，"
                "不要删文件——保留文件并把对应的 valid 标记置 0。",
            ))

    # ---- 建议文件 ----
    for name, (spec, purpose, consequence) in RECOMMENDED_FILES.items():
        if not (directory / name).exists():
            report.findings.append(Finding(
                "警告",
                f"缺少建议文件 {name}（规格 {spec}）",
                f"{purpose}。缺了它，{consequence}。",
                f"建议补上 {name}。它不影响对齐能否跑通，"
                "但论文中的可解释性与可追溯性材料需要用到。",
            ))

    # ---- 目录级清单 ----
    missing_manifest = [n for n in MANIFEST_FILES if not (directory.parent.parent / n).exists()]
    for name in missing_manifest:
        report.findings.append(Finding(
            "警告",
            f"交付根目录缺少 {name}",
            f"{MANIFEST_FILES[name]}。这是全量结果可追溯的依据。",
            f"请在 q1_feature_handoff/ 下补上 {name}。",
        ))

    if report.errors:
        return report

    # ---- 内容检查 ----
    text = _load(directory, "text_features.npy", report)
    audio = _load(directory, "audio_features.npy", report)
    a_int = _load(directory, "audio_intervals.npy", report)
    vision = _load(directory, "vision_features.npy", report)
    v_ts = _load(directory, "vision_timestamps.npy", report)

    for name, array in (("text_features.npy", text),
                        ("audio_features.npy", audio),
                        ("vision_features.npy", vision)):
        if array is not None:
            _check_features(name, array, report)
            if array.ndim == 2:
                report.stats[f"{name}_shape"] = list(array.shape)

    if audio is not None and a_int is not None and audio.shape[0] != a_int.shape[0]:
        report.findings.append(Finding(
            "错误",
            f"audio_features 有 {audio.shape[0]} 行，audio_intervals 有 {a_int.shape[0]} 行",
            "每个音频帧必须对应一个时间区间，行数不同就无法配对。",
            "检查两者是否用同一个帧循环生成；常见原因是丢弃了静音帧"
            "但没同步丢弃对应的时间区间。不要删除静音帧——"
            "文档要求保留原始时序。",
        ))

    if vision is not None and v_ts is not None and vision.shape[0] != v_ts.shape[0]:
        report.findings.append(Finding(
            "错误",
            f"vision_features 有 {vision.shape[0]} 行，vision_timestamps 有 {v_ts.shape[0]} 行",
            "每个视觉帧必须对应一个时间戳，行数不同就无法配对。",
            "确认视觉特征提取时，是否对未检出人脸的帧做了删除或跳过。"
            "正确做法：保留该帧，置 vision_valid=0。",
        ))

    if a_int is not None:
        _check_time_axis("audio_intervals.npy", a_int, report, duration)
    if v_ts is not None:
        _check_time_axis("vision_timestamps.npy", v_ts, report, duration)

    _check_valid_mask("vision_valid.npy", directory, report,
                      vision.shape[0] if vision is not None and vision.ndim == 2 else None)
    _check_valid_mask("audio_valid.npy", directory, report,
                      audio.shape[0] if audio is not None and audio.ndim == 2 else None)
    _check_valid_mask("text_valid.npy", directory, report,
                      text.shape[0] if text is not None and text.ndim == 2 else None)

    _check_text_words(directory, report, text.shape[0] if text is not None and text.ndim == 2 else None)

    # ---- 维度一致性提示 ----
    if text is not None and text.ndim == 2:
        report.stats["text_dim"] = int(text.shape[1])
    if audio is not None and audio.ndim == 2:
        report.stats["audio_dim"] = int(audio.shape[1])
    if vision is not None and vision.ndim == 2:
        report.stats["vision_dim"] = int(vision.shape[1])

    return report


# --------------------------------------------------------------------------
# 输出
# --------------------------------------------------------------------------
LEVEL_ORDER = {"错误": 0, "缺失": 1, "警告": 2}


def print_report(reports: list[SampleReport], root: Path) -> tuple[int, int, int]:
    errors = warnings = 0

    for report in reports:
        report.findings.sort(key=lambda f: LEVEL_ORDER.get(f.level, 9))
        status = "通过" if report.ok else "未通过"
        stats = report.stats
        detail = ""
        if stats.get("text_features_shape"):
            detail = (f"  text={tuple(stats['text_features_shape'])}"
                      f" audio={tuple(stats.get('audio_features_shape', []))}"
                      f" vision={tuple(stats.get('vision_features_shape', []))}")
        print(f"[{status}] {report.sample_id}{detail}")

        if not report.ok:
            for finding in report.findings:
                if finding.level in ("错误", "缺失"):
                    errors += 1
                    print(f"    ✗ [{finding.level}] {finding.what}")
                    print(f"        原因：{finding.why}")
                    print(f"        改正：{finding.how}")
        for finding in report.findings:
            if finding.level == "警告":
                warnings += 1
                print(f"    ! [警告] {finding.what}")
                print(f"        原因：{finding.why}")
                print(f"        改正：{finding.how}")

    return errors, warnings, len(reports)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="交付格式自检（三模态）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--handoff", required=True, help="q1_feature_handoff 目录")
    parser.add_argument("--limit", type=int, default=None, help="只检查前 N 个样本")
    parser.add_argument("--json", help="把结果写入该 JSON 文件")
    parser.add_argument("--quiet", action="store_true", help="只输出汇总")
    args = parser.parse_args(argv)

    root = Path(args.handoff)
    samples = root / "samples"
    if not samples.is_dir():
        print(f"[错误] 找不到 {samples}")
        print("        原因：交付目录必须按文档组织，样本放在 samples/ 下。")
        print(f"        改正：确认路径正确，或建立 samples/ 并把每个样本"
              "建成 samples/<video_id>$_$<clip_id>/ 子目录。")
        return 1

    directories = sorted(p for p in samples.iterdir() if p.is_dir())
    if args.limit:
        directories = directories[: args.limit]
    if not directories:
        print(f"[错误] {samples} 下没有样本目录")
        print("        改正：每个样本应为 samples/<sample_id>/ 子目录。")
        return 1

    print("=" * 72)
    print("问题一 特征交付自检")
    print("=" * 72)
    print(f"交付目录: {root}")
    print(f"样本数量: {len(directories)}")
    print()

    # 尝试读取时长信息用于越界判断
    durations: dict[str, float] = {}
    manifest_csv = root / "manifest.csv"
    if manifest_csv.exists():
        try:
            with manifest_csv.open(encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    key = row.get("sample_id") or row.get("id")
                    value = row.get("duration") or row.get("video_duration")
                    if key and value:
                        durations[key] = float(value)
        except Exception:
            pass

    reports = []
    for directory in directories:
        reports.append(check_sample(directory, durations.get(directory.name)))

    errors, warnings, total = print_report(reports, root)

    passed = sum(1 for r in reports if r.ok)
    print()
    print("=" * 72)
    print(f"通过 {passed}/{total}    错误 {errors}    警告 {warnings}")
    print("=" * 72)

    if errors:
        print()
        print("必须修复后才能交付（错误与缺失项）：")
        seen: set[str] = set()
        for report in reports:
            for finding in report.errors:
                key = finding.what.split("（")[0]
                if key not in seen:
                    seen.add(key)
                    print(f"  - {finding.what}")
        print()
        print("修复后请重新运行本脚本，直到「错误 0」。")
    elif warnings:
        print()
        print("没有阻断性错误，可以交付。")
        print("建议一并处理上面的警告，它们会影响论文的可追溯性材料。")
    else:
        print()
        print("完全合规，可以直接交付。")

    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {
                    "samples": total,
                    "passed": passed,
                    "errors": errors,
                    "warnings": warnings,
                    "per_sample": [
                        {
                            "sample_id": r.sample_id,
                            "ok": r.ok,
                            "findings": [
                                {"level": f.level, "what": f.what,
                                 "why": f.why, "how": f.how}
                                for f in r.findings
                            ],
                        }
                        for r in reports
                    ],
                },
                ensure_ascii=False, indent=2,
            ),
            encoding="utf-8",
        )

    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
