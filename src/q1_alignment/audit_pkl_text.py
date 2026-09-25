"""Scan every text-bearing surface of attachment 2 for injected instructions.

Anti-AI designs in recent competitions sometimes embed text that a human would
never read but a model would happily follow (prompt injection). This scans:

  * raw_text in both pkl files
  * the label spreadsheets
  * filenames and directory names
  * zero-width / bidi / homoglyph characters
"""

from __future__ import annotations

import pickle
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import openpyxl

DATA = Path(r"G:\MathModel\E题数据")

# Patterns that would indicate an attempt to steer a model
SUSPICIOUS_PATTERNS = [
    (r"(?i)\bignore\s+(all\s+)?(previous|above|prior)\b", "忽略先前指令"),
    (r"(?i)\bdisregard\s+(the\s+)?(above|previous|instruction)", "无视指令"),
    (r"(?i)\byou\s+are\s+(now\s+)?(a|an)\b", "角色重定义"),
    (r"(?i)\bsystem\s*prompt\b", "系统提示词"),
    (r"(?i)\bas\s+an?\s+(ai|assistant|language\s+model)\b", "AI 自指"),
    (r"(?i)\bprompt\b", "提示词"),
    (r"(?i)\binstruction[s]?\b", "指令"),
    (r"(?i)(必须|务必|请注意).{0,20}(回答|输出|忽略|不要)", "中文指令式"),
    (r"(?i)\bdo\s+not\s+(report|mention|reveal)\b", "要求隐瞒"),
    (r"(?i)\b(the\s+)?(answer|solution)\s+is\b", "直接给答案"),
    (r"(?i)\bflag\s*\{", "CTF 式标记"),
    (r"(?i)\bbase64\b", "编码指示"),
    (r"(?i)https?://\S+", "外链"),
]

# Invisible / deceptive unicode categories
INVISIBLE = {
    "\u200b": "零宽空格",
    "\u200c": "零宽非连接符",
    "\u200d": "零宽连接符",
    "\u2060": "词连接符",
    "\ufeff": "BOM/零宽无断空格",
    "\u202a": "从左至右嵌入",
    "\u202b": "从右至左嵌入",
    "\u202c": "方向格式化终止",
    "\u202d": "从左至右覆盖",
    "\u202e": "从右至左覆盖(RTL反转)",
    "\u2066": "从左至右隔离",
    "\u2067": "从右至左隔离",
    "\u2069": "隔离终止",
}


def scan_text(text: str, where: str, hits: list) -> None:
    """Record suspicious patterns and invisible characters."""

    for pattern, label in SUSPICIOUS_PATTERNS:
        for match in re.finditer(pattern, text):
            start = max(0, match.start() - 60)
            end = min(len(text), match.end() + 60)
            hits.append((where, label, match.group(0), text[start:end]))

    for char, name in INVISIBLE.items():
        if char in text:
            index = text.index(char)
            context = text[max(0, index - 40): index + 40]
            hits.append((where, f"不可见字符 {name}", repr(char), context))


def scan_pkl(path: Path, hits: list, limit: int | None = None) -> dict:
    print(f"  读取 {path.name} ...")
    with path.open("rb") as handle:
        data = pickle.load(handle)

    stats = {}
    scanned = 0
    for split in data:
        block = data[split]
        if not isinstance(block, dict):
            continue
        raw_text = block.get("raw_text")
        ids = block.get("id")
        stats[split] = len(ids) if ids is not None else 0
        if raw_text is None:
            continue
        for index, text in enumerate(raw_text):
            if limit and scanned >= limit:
                break
            scanned += 1
            scan_text(str(text), f"{path.name}/{split}/raw_text[{index}]", hits)
            if ids is not None:
                scan_text(str(ids[index]), f"{path.name}/{split}/id[{index}]", hits)
    print(f"    扫描 {scanned} 条文本")
    return stats


def main() -> None:
    hits: list = []

    print("=" * 78)
    print("1. 扫描 pkl 中的 raw_text 与 id")
    print("=" * 78)
    for name in ("aligned_50.pkl", "unaligned_50.pkl"):
        path = DATA / "附件2-数据集特征文件" / name
        if path.exists():
            scan_pkl(path, hits)

    print()
    print("=" * 78)
    print("2. 扫描标签表")
    print("=" * 78)
    for path in (
        DATA / "附件1-数据集原始多模态样本/MOSEI数据集部分原始视频-100条/label-100.xlsx",
        DATA / "附件2-数据集特征文件/label.xlsx",
    ):
        if not path.exists():
            continue
        wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
        count = 0
        for sheet in wb.sheetnames:
            for row in wb[sheet].iter_rows(values_only=True):
                for cell in row:
                    if isinstance(cell, str):
                        count += 1
                        scan_text(cell, f"{path.name}/{sheet}", hits)
        print(f"  {path.name}: 扫描 {count} 个字符串单元格")

    print()
    print("=" * 78)
    print("3. 扫描文件名与目录名")
    print("=" * 78)
    names = 0
    for path in DATA.rglob("*"):
        names += 1
        scan_text(path.name, f"文件名:{path.parent.name}", hits)
    print(f"  扫描 {names} 个路径")

    print()
    print("=" * 78)
    print("扫描结果")
    print("=" * 78)
    if not hits:
        print("  未发现可疑指令、外链或不可见字符。")
        return

    grouped: Counter = Counter()
    for where, label, matched, context in hits:
        grouped[label] += 1
    print(f"  共 {len(hits)} 处命中，分类：")
    for label, count in grouped.most_common():
        print(f"    {label}: {count}")

    print()
    for label in grouped:
        print(f"--- {label} ---")
        shown = 0
        for where, lb, matched, context in hits:
            if lb != label or shown >= 3:
                continue
            shown += 1
            print(f"  位置: {where}")
            print(f"  匹配: {matched!r}")
            print(f"  上下文: ...{context}...")
            print()


if __name__ == "__main__":
    sys.exit(main())
