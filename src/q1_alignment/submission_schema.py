"""Submission-schema definitions and validator for the E-problem deliverables.

Why this exists
---------------
The problem states (段55):

    专项测试结果文件：附件3测试集预测结果CSV文件、附件4测试集预测与解释结果CSV文件，
    文件命名清晰规范。

and (段30):

    分类任务（情感极性）采用准确率（Accuracy）、F1值评价，
    回归任务（情感强度）采用平均绝对误差（MAE）、皮尔逊相关系数评价。

"文件命名清晰规范" and "预测结果" are prose, not a schema.  Left at that level
each teammate invents their own column names, and the three sections of the paper
end up describing three incompatible file formats -- which the guard's F/G checks
would then flag late, after the modelling is done.

This module turns the prose into an explicit, testable contract so a teammate can
validate their own CSV before sending it, exactly as `handoff_selfcheck.py` does
for the problem-1 feature handoff.

Two schemas are defined:

* attachment 3 -- prediction only (polarity + intensity)
* attachment 4 -- prediction + interpretability (dominant modality, per-modality
  contribution, and a locatable evidence span)
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

__all__ = [
    "Column",
    "Schema",
    "SCHEMAS",
    "validate_csv",
    "Finding",
    "ATT3_SAMPLE_COUNT",
    "ATT4_SAMPLE_COUNT",
]

#: Measured on the real attachments (30 pkl files / 20 pkl files, 1 sample each).
ATT3_SAMPLE_COUNT = 30
ATT4_SAMPLE_COUNT = 20

#: Label semantics verified from attachment 2: 0 is Neutral ONLY, and there is a
#: gap between the positive and negative ranges (negative max -3.0, positive min
#: +0.1667).  So polarity must be a 3-class label, never a thresholded sign.
POLARITY_DOMAIN = ("negative", "neutral", "positive")


@dataclass(frozen=True)
class Column:
    """One required CSV column.

    ``aliases`` lets a teammate use a reasonable synonym without failing, while
    still requiring that *some* column carries the information.
    """

    name: str
    aliases: tuple[str, ...]
    kind: str                      # str | float | int | polarity | bool
    why: str                       # which problem sentence demands it
    lo: float | None = None
    hi: float | None = None


@dataclass
class Schema:
    id: str
    filename_hint: str
    expected_rows: int
    columns: list[Column] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


COMMON_PREDICTION = [
    Column("sample_id", ("id", "file", "name", "sample", "文件名"),
           "str",
           "附件3/4 无 id 字段，只能用文件名标识样本；缺了就无法对应回附件"),
    Column("polarity", ("classification", "label", "class", "极性", "情感极性"),
           "polarity",
           "题目段24/28：模型需输出「情感极性」"),
    Column("intensity", ("regression", "score", "value", "强度", "情感强度"),
           "float", "题目段24/28：模型需输出「情感强度」", -3.0, 3.0),
]

SCHEMAS: dict[str, Schema] = {
    "att3": Schema(
        id="att3",
        filename_hint="附件3_预测结果.csv",
        expected_rows=ATT3_SAMPLE_COUNT,
        columns=list(COMMON_PREDICTION),
        notes=[
            "附件3 为无标签专项测试集，禁止出现任何真值列",
            "30 个 pkl 对应 30 行，逐一核对不得漏样本",
            "polarity 必须是 negative/neutral/positive 三分类："
            "标签中 0 只属中性且正负之间有间隙，不可用回归值直接取符号",
        ],
    ),
    "att4": Schema(
        id="att4",
        filename_hint="附件4_预测与解释结果.csv",
        expected_rows=ATT4_SAMPLE_COUNT,
        columns=list(COMMON_PREDICTION) + [
            Column("dominant_modality",
                   ("main_modality", "dominant", "main_mod", "主要参考模态"),
                   "str",
                   "题目段28：需输出「对预测起主要作用的模态」"),
            Column("modality_weights",
                   ("weights", "contribution", "importance", "模态作用程度"),
                   "str",
                   "题目段27/28：需输出「不同模态在当前样本中的作用差异」"
                   "「模态作用程度」"),
            Column("evidence_start", ("evidence_start_sec", "start", "关键证据起点"),
                   "float", "题目段28：关键证据需可对应至原始时间位置", 0.0, None),
            Column("evidence_end", ("evidence_end_sec", "end", "关键证据终点"),
                   "float", "题目段28：关键证据需可对应至原始时间位置", 0.0, None),
            Column("evidence_modality", ("evidence_mod", "证据模态"),
                   "str", "题目段28：关键证据需指明落在文本/语音/视觉哪一模态"),
        ],
        notes=[
            "附件4 无标签，同样禁止真值列",
            "evidence_start/end 必须落在该样本真实时长内，"
            "否则无法「回看至原始视频」（题目段16）",
            "modality_weights 建议写成 JSON 或 'text:0.5,audio:0.3,vision:0.2' 形式",
        ],
    ),
}


@dataclass
class Finding:
    level: str      # 错误 | 缺失 | 警告
    what: str
    why: str
    how: str


def _pick(fields: Iterable[str], col: Column) -> str | None:
    lowered = {f.lower(): f for f in fields}
    for cand in (col.name, *col.aliases):
        if cand.lower() in lowered:
            return lowered[cand.lower()]
    return None


def _check_value(raw: str, col: Column) -> str | None:
    """Return a problem description, or None when the value is acceptable."""
    text = (raw or "").strip()
    if text == "":
        return "空值"
    if col.kind in ("float", "int"):
        try:
            val = float(text)
        except ValueError:
            return f"无法解析为数值: {text!r}"
        if col.lo is not None and val < col.lo - 1e-9:
            return f"{val} 低于下限 {col.lo}"
        if col.hi is not None and val > col.hi + 1e-9:
            return f"{val} 高于上限 {col.hi}"
    if col.kind == "polarity":
        if text.lower() not in POLARITY_DOMAIN and text not in ("-1", "0", "1"):
            return (f"{text!r} 不是合法极性；应为 "
                    f"{POLARITY_DOMAIN} 之一或 -1/0/1")
    return None


def validate_csv(path: str | Path, schema_key: str) -> list[Finding]:
    """Validate one submission CSV.  Returns every problem found, not just the first."""
    schema = SCHEMAS[schema_key]
    path = Path(path)
    findings: list[Finding] = []

    if not path.exists():
        return [Finding("缺失", f"文件不存在: {path}",
                        f"{schema.id} 的提交文件缺失",
                        f"应由 {schema.filename_hint} 提供")]

    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames or []
        rows = list(reader)

    if not rows:
        findings.append(Finding("错误", "CSV 无数据行", "空结果无法评分",
                                "确认推理是否真的跑过附件"))
        return findings

    # ---- columns ----
    resolved: dict[str, str] = {}
    for col in schema.columns:
        hit = _pick(fields, col)
        if hit is None:
            findings.append(Finding(
                "缺失",
                f"缺少列 {col.name}（别名 {list(col.aliases)} 均未命中）",
                col.why,
                f"加一列 {col.name}；现有列: {fields}",
            ))
        else:
            resolved[col.name] = hit

    # ---- forbidden ground-truth columns ----
    forbidden = ("regression_labels", "classification_labels", "annotation",
                 "gt", "ground_truth", "真值", "label_true")
    leaked = [f for f in fields if f.lower() in forbidden]
    if leaked:
        findings.append(Finding(
            "错误", f"出现疑似真值列: {leaked}",
            "附件3/4 均为无标签专项测试集；出现真值列说明数据来源有误或标签泄漏",
            "删除这些列；若确实需要参考真值，说明用途并确认未参与任何调参",
        ))

    # ---- row count ----
    if len(rows) != schema.expected_rows:
        findings.append(Finding(
            "错误", f"行数 {len(rows)} != 期望 {schema.expected_rows}",
            "题目要求覆盖全量专项测试样本",
            f"{schema.expected_rows} 个 pkl 应产生 {schema.expected_rows} 行",
        ))

    # ---- per-value checks ----
    bad: dict[str, list[str]] = {}
    for i, row in enumerate(rows):
        for col in schema.columns:
            field_name = resolved.get(col.name)
            if field_name is None:
                continue
            problem = _check_value(row.get(field_name, ""), col)
            if problem:
                bad.setdefault(col.name, []).append(f"第{i+2}行: {problem}")
    for name, problems in bad.items():
        shown = problems[:3]
        findings.append(Finding(
            "错误",
            f"列 {name} 有 {len(problems)} 个非法值（示例 {shown}）",
            "非法值会让该样本无法计入指标",
            "修正后再提交",
        ))

    # ---- duplicate ids ----
    id_field = resolved.get("sample_id")
    if id_field:
        ids = [r.get(id_field, "") for r in rows]
        dupes = {x for x in ids if ids.count(x) > 1}
        if dupes:
            findings.append(Finding(
                "错误", f"sample_id 重复: {sorted(dupes)[:5]}",
                "重复标识会让样本计数虚高",
                "每个 pkl 恰好一行",
            ))

    # ---- evidence span sanity (att4) ----
    if schema_key == "att4":
        s_field = resolved.get("evidence_start")
        e_field = resolved.get("evidence_end")
        if s_field and e_field:
            bad_span = []
            for i, row in enumerate(rows):
                try:
                    s = float(row[s_field]); e = float(row[e_field])
                except (TypeError, ValueError):
                    continue
                if e <= s:
                    bad_span.append(f"第{i+2}行: end({e}) <= start({s})")
            if bad_span:
                findings.append(Finding(
                    "错误", f"证据区间无效: {len(bad_span)} 行",
                    "题目段28 要求关键证据可对应至原文片段/语音时段/视觉关键帧，"
                    "区间倒置或零长度无法回看",
                    "保证 start < end 且落在样本真实时长内",
                ))
    return findings


def format_findings(findings: list[Finding], title: str) -> str:
    if not findings:
        return f"[通过] {title}: 未发现问题"
    order = {"错误": 0, "缺失": 1, "警告": 2}
    lines = [f"[{len(findings)} 个问题] {title}"]
    for f in sorted(findings, key=lambda x: order.get(x.level, 9)):
        lines.append(f"\n  ({f.level}) {f.what}")
        lines.append(f"      原因: {f.why}")
        lines.append(f"      处理: {f.how}")
    return "\n".join(lines)
