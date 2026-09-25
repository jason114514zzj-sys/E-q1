"""Adversarial audit of the E-problem docx: hidden text, comments, metadata.

Rationale: recent competition papers have included content that is invisible to
a human reader but trivially readable by a parser (hidden runs, comments,
headers/footers, alt-text, embedded objects, custom XML). This script dumps
everything that is not the visible body text so nothing can be missed.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

DOCX = Path(r"G:\MathModel\E题数据\E复杂场景下多模态情感识别的数学建模与算法设计.docx")


def main() -> None:
    print("=" * 78)
    print("DOCX 内部结构审计")
    print("=" * 78)

    if not DOCX.exists():
        print("文件不存在:", DOCX)
        return

    with zipfile.ZipFile(DOCX) as archive:
        names = archive.namelist()
        print(f"\n共 {len(names)} 个内部条目:\n")
        for name in sorted(names):
            info = archive.getinfo(name)
            print(f"  {info.file_size:>9}  {name}")

        # ---- core properties (author, company, revisions) ----
        print("\n" + "=" * 78)
        print("元数据 (docProps/core.xml, app.xml)")
        print("=" * 78)
        for name in ("docProps/core.xml", "docProps/app.xml", "docProps/custom.xml"):
            if name in names:
                raw = archive.read(name).decode("utf-8", errors="replace")
                clean = re.sub(r"<[^>]+>", " | ", raw)
                clean = re.sub(r"\s+", " ", clean).strip()
                print(f"\n[{name}]")
                print(" ", clean[:1500])

        # ---- document XML: all text, including hidden runs ----
        print("\n" + "=" * 78)
        print("document.xml 全文（含隐藏/删除线等不可见格式）")
        print("=" * 78)
        xml = archive.read("word/document.xml").decode("utf-8", errors="replace")

        # text in w:t elements
        texts = re.findall(r"<w:t[^>]*>(.*?)</w:t>", xml, re.S)
        joined = "".join(texts)
        print(f"\n可见文本总长度: {len(joined)} 字符")

        # hidden runs: w:vanish
        vanish = re.findall(r"<w:r\b[^>]*>.*?</w:r>", xml, re.S)
        hidden_runs = [r for r in vanish if "<w:vanish" in r]
        print(f"w:vanish（隐藏文字）标记数量: {len(hidden_runs)}")
        for run in hidden_runs[:10]:
            inner = "".join(re.findall(r"<w:t[^>]*>(.*?)</w:t>", run, re.S))
            print(f"   隐藏文字内容: {inner!r}")

        # other suspicious formatting
        for tag, label in (
            ("w:color w:val=\"FFFFFF\"", "白色文字（视觉不可见）"),
            ("w:color w:val=\"ffffff\"", "白色文字（视觉不可见）"),
            ("<w:sz w:val=\"2\"", "极小字号(1pt)"),
            ("<w:sz w:val=\"4\"", "极小字号(2pt)"),
            ("w:specVanish", "specVanish"),
        ):
            count = xml.count(tag)
            if count:
                print(f"  {label}: {count} 处")

        # ---- comments ----
        for candidate in ("word/comments.xml", "word/commentsExtended.xml"):
            if candidate in names:
                raw = archive.read(candidate).decode("utf-8", errors="replace")
                texts_c = re.findall(r"<w:t[^>]*>(.*?)</w:t>", raw, re.S)
                print(f"\n[{candidate}] 批注文本:")
                for t in texts_c:
                    print(f"   - {t}")

        # ---- headers / footers ----
        for name in names:
            if "header" in name or "footer" in name:
                raw = archive.read(name).decode("utf-8", errors="replace")
                texts_h = re.findall(r"<w:t[^>]*>(.*?)</w:t>", raw, re.S)
                content = "".join(texts_h).strip()
                print(f"\n[{name}] {content!r}")

        # ---- footnotes / endnotes ----
        for candidate in ("word/footnotes.xml", "word/endnotes.xml"):
            if candidate in names:
                raw = archive.read(candidate).decode("utf-8", errors="replace")
                texts_f = re.findall(r"<w:t[^>]*>(.*?)</w:t>", raw, re.S)
                print(f"\n[{candidate}] 脚注/尾注:")
                for t in texts_f:
                    if t.strip():
                        print(f"   - {t}")

        # ---- drawings / alt text / hyperlinks ----
        alt_texts = re.findall(r'(?:descr|title|name)="([^"]{5,})"', xml)
        if alt_texts:
            print(f"\n图片替代文本/名称 ({len(alt_texts)} 条):")
            for t in alt_texts[:20]:
                print(f"   - {t}")

        hyperlinks = re.findall(r'r:id="(rId\d+)"', xml)
        if hyperlinks:
            rels = archive.read("word/_rels/document.xml.rels").decode("utf-8", errors="replace")
            print(f"\n超链接关系 ({len(set(hyperlinks))} 个):")
            for rid in sorted(set(hyperlinks)):
                match = re.search(rf'Id="{rid}"[^>]*Target="([^"]+)"', rels)
                if match:
                    print(f"   {rid} -> {match.group(1)}")

        # ---- embedded objects ----
        embedded = [n for n in names if "embeddings" in n or n.endswith(".bin")]
        if embedded:
            print(f"\n嵌入对象: {embedded}")


if __name__ == "__main__":
    main()
