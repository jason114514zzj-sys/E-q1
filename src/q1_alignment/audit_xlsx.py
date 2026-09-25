"""Audit all competition spreadsheets for hidden content.

Checks: hidden sheets/rows/columns, comments, defined names, formulas that
reference external workbooks, and any cell content that differs from what a
normal reader would see.
"""

from __future__ import annotations

from pathlib import Path

import openpyxl

TARGETS = [
    Path(r"G:\MathModel\E题数据\附件1-数据集原始多模态样本\MOSEI数据集部分原始视频-100条\label-100.xlsx"),
    Path(r"G:\MathModel\E题数据\附件2-数据集特征文件\label.xlsx"),
]


def audit(path: Path) -> None:
    print("=" * 78)
    print(path.name)
    print("=" * 78)
    if not path.exists():
        print("  不存在\n")
        return

    wb = openpyxl.load_workbook(path, data_only=False, read_only=False)

    print(f"工作表: {wb.sheetnames}")
    for name in wb.sheetnames:
        ws = wb[name]
        state = ws.sheet_state
        print(f"\n  [{name}]  状态={state}  {ws.max_row} 行 x {ws.max_column} 列")
        if state != "visible":
            print(f"    !! 该表被隐藏 ({state})")

        # hidden rows / columns
        hidden_rows = [r for r, d in ws.row_dimensions.items() if d.hidden]
        hidden_cols = [c for c, d in ws.column_dimensions.items() if d.hidden]
        if hidden_rows:
            print(f"    !! 隐藏行: {hidden_rows[:30]}{' ...' if len(hidden_rows) > 30 else ''}")
        if hidden_cols:
            print(f"    !! 隐藏列: {hidden_cols}")

        # comments
        comments = []
        for row in ws.iter_rows():
            for cell in row:
                if cell.comment is not None:
                    comments.append((cell.coordinate, cell.comment.text))
        if comments:
            print(f"    批注 ({len(comments)} 条):")
            for coord, text in comments[:20]:
                print(f"      {coord}: {text[:150]}")

        # data validation / conditional formatting hints
        if ws.data_validations and ws.data_validations.dataValidation:
            print(f"    数据验证规则: {len(ws.data_validations.dataValidation)}")

        # formulas referencing external sources
        external = []
        for row in ws.iter_rows():
            for cell in row:
                value = cell.value
                if isinstance(value, str) and value.startswith("="):
                    if any(k in value for k in ("http", "[", ".xls", ".csv", "WEBSERVICE")):
                        external.append((cell.coordinate, value))
        if external:
            print(f"    !! 可疑公式 ({len(external)} 条):")
            for coord, value in external[:10]:
                print(f"      {coord}: {value[:160]}")

    # defined names
    if wb.defined_names:
        try:
            names = list(wb.defined_names.items())
            print(f"\n  定义名称: {[n for n, _ in names]}")
        except Exception:
            print(f"\n  定义名称: {list(wb.defined_names)}")

    print()


def main() -> None:
    for path in TARGETS:
        audit(path)


if __name__ == "__main__":
    main()
