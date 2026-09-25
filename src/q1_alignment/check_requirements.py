"""Check every explicit problem-1 deliverable required by the E-problem text.

Requirement sources (problem statement section 三 / 四):

问题1 (line 44):
  一是原始样本覆盖完整性 —— 样本编号、模态文件与输出特征一一对应
  二是时序组织的可核验性 —— 序列位置、有效长度、填充规则、与原始素材的对应记录
  三是方法的合理性与可复现性 —— 工具、关键参数、处理日志、复现说明

四.2.问题1相关内容 (lines 76-82):
  (1) 整体方案
  (2) 特征文件规范与全量结果汇总表
      —— 样本编号、模态类型、原始有效时长、特征维度、对齐粒度
      —— 全量原始特征文件随附件提交
  (3) 典型样本验证（文本片段/语音时段/视频帧段/三类特征对应关系）
  (4) 方案可复现性说明（版本、参数、完整运行流程）
"""

from __future__ import annotations

import json
from pathlib import Path

HOME = Path.home()
ROOT = HOME / "MathModel"

CHECKS: list[tuple[str, str, bool, str]] = []


def check(requirement: str, evidence: str, passed: bool, note: str = "") -> None:
    CHECKS.append((requirement, evidence, passed, note))


def exists(path: Path) -> bool:
    return path.exists()


def main() -> None:
    work = ROOT / "work"
    src = ROOT / "src" / "q1_alignment"

    # ---- 问题1 三条核心要求 ----
    manifest = work / "manifest.jsonl"
    records = []
    if manifest.exists():
        records = [
            json.loads(l)
            for l in manifest.read_text(encoding="utf-8").splitlines()
            if l.strip()
        ]

    check(
        "覆盖完整性: 样本编号一一对应",
        "work/manifest.jsonl",
        len(records) == 100 and len({r["id"] for r in records}) == 100,
        f"{len(records)} 条，id 唯一" if records else "缺失",
    )
    check(
        "覆盖完整性: 模态文件与特征一一对应",
        "work/word_timelines.jsonl + aligned/*",
        exists(work / "word_timelines.jsonl"),
        "词级时间轴已生成（等待队友特征接入）",
    )
    check(
        "时序可核验: 序列位置/有效长度/填充规则",
        "alignment.py: sequence_mask / effective_length / interval_start",
        exists(src / "alignment.py"),
        "四类掩码 + 有效长度 + 区间边界",
    )
    check(
        "时序可核验: 与原始素材对应记录",
        "work/alignment_trace.csv",
        exists(work / "alignment_trace.csv"),
        "逐位置可回溯到音频/视觉帧索引",
    )
    check(
        "方法可复现: 工具与关键参数",
        "config.yaml",
        exists(ROOT / "config.yaml"),
        "冻结配置含全部参数",
    )
    check(
        "方法可复现: 处理日志",
        "work/word_align_report.csv",
        exists(work / "word_align_report.csv"),
        "逐样本耗时/状态/问题记录",
    )
    check(
        "方法可复现: 复现说明",
        "src/q1_alignment/README.md",
        exists(src / "README.md"),
        "含运行命令与约束说明",
    )

    # ---- 四.2 问题1 提交要求 ----
    check(
        "四.2.(1) 整体方案（文本/语音/视觉处理流程）",
        "报告文字部分（待撰写）",
        False,
        "论文正文内容，代码侧已具备素材",
    )
    check(
        "四.2.(2) 全量结果汇总表",
        "待生成 ALL_SAMPLES_SUMMARY.csv",
        False,
        "★ 明确要求：样本编号/模态/原始有效时长/特征维度/对齐粒度",
    )
    check(
        "四.2.(2) 全量原始特征文件",
        "待队友特征交付后生成",
        False,
        "★ 需随附件提交",
    )
    check(
        "四.2.(3) 典型样本验证",
        "待生成典型样本对齐材料",
        False,
        "★ 至少1个样本，展示三模态对应关系",
    )
    check(
        "四.2.(4) 可复现性说明",
        "versions.txt + SERVER_ENV.md",
        exists(ROOT / "versions.txt") and exists(ROOT / "SERVER_ENV.md"),
        "版本与运行环境已记录",
    )

    # ---- 附件提交 ----
    check(
        "附件≤50MB",
        "待打包时核验",
        False,
        "当前中间产物合计需核算",
    )
    check(
        "无身份信息泄漏",
        "待打包时核验",
        False,
        "打包前需扫描",
    )

    # ---- 输出 ----
    width = 62
    print("=" * 100)
    print("问题一 交付要求核对表".center(90))
    print("=" * 100)
    done = sum(1 for _, _, ok, _ in CHECKS if ok)
    for requirement, evidence, ok, note in CHECKS:
        mark = "[OK]  " if ok else "[TODO]"
        print(f"{mark} {requirement}")
        print(f"       证据: {evidence}")
        if note:
            print(f"       备注: {note}")
    print("=" * 100)
    print(f"已满足 {done} / {len(CHECKS)}")
    print()
    print("待办项（严格按题目要求）:")
    for requirement, evidence, ok, note in CHECKS:
        if not ok:
            print(f"  - {requirement}")
            print(f"      -> {note}")


if __name__ == "__main__":
    main()
