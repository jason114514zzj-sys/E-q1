"""Verify that the trap-mitigation modules are actually wired into the pipeline.

Writing a correct module is not the same as using it.  Three modules exist to
neutralise specific traps:

* ``text_pooling.py``   -- neutralises C7 (naive 50-step text averaging is ~43%
  wrong because padding is non-zero);
* ``feature_scaling.py``-- neutralises C2/C8 (audio dim 0 has ~394x the std of
  the other dims);
* ``submission_schema.py`` -- neutralises F4/G5 (submission column and value
  legality).

C7 and C8 currently pass only because those modules exist AND are exercised by
their own tests.  Nothing yet proves the *pipeline* calls them.  If the pipeline
still averages text naively, the guard would report green while the real
features are the wrong ones -- exactly the failure the guard is meant to
prevent.

This script scans the pipeline and the modules that produce aligned output for
imports/usages, and reports any mitigation module that is never referenced.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

SRC = Path("/home/user/MathModel/src/q1_alignment")
SCRIPTS = Path("/home/user/MathModel/scripts")

# module -> (trap it neutralises, symbols that count as "used")
MITIGATIONS = {
    "text_pooling": (
        "C7 文本填充非零，直接平均错 43%",
        ("pool_text_sequence", "masked_mean_pool", "extract_attention_mask"),
    ),
    "feature_scaling": (
        "C2/C8 audio dim0 量纲比 394x",
        ("fit_scaler", "FeatureScaler", "audio_valid_mask", "split_scale_report"),
    ),
    "submission_schema": (
        "F4/G5 提交文件列与值合法性",
        ("validate_csv", "SCHEMAS"),
    ),
    "claim_check": (
        "H4 文档数值与数据一致",
        ("verify_claims", "recompute"),
    ),
}

# files that constitute "the pipeline" (production code, not tests/checks)
PRODUCTION = set()
for p in list(SRC.glob("*.py")) + list(SCRIPTS.glob("*.py")) + list(SCRIPTS.glob("*.sh")):
    if p.name.startswith("test_"):
        continue
    PRODUCTION.add(p)

print("=" * 76)
print("缓解模块接线检查：模块是否真的被流水线调用")
print("=" * 76)
print(f"\n生产代码文件 {len(PRODUCTION)} 个：")
for p in sorted(PRODUCTION):
    print(f"   {p.relative_to(SRC.parent.parent)}")

problems: list[str] = []
for mod, (trap, symbols) in MITIGATIONS.items():
    print("\n" + "-" * 76)
    print(f"模块 {mod}.py  —— 针对 {trap}")
    print("-" * 76)
    users: list[str] = []
    for p in PRODUCTION:
        if p.stem == mod:
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        if re.search(rf"\b{mod}\b", text):
            hit = [s for s in symbols if s in text]
            users.append(f"{p.name} (用到 {hit})" if hit
                         else f"{p.name} (仅提及模块名)")
    if users:
        for u in users:
            print(f"   [被引用] {u}")
    else:
        print("   [未被引用] 没有任何生产代码使用它")
        problems.append(mod)

# deeper check: does the alignment step produce text via the pooling module?
print("\n" + "=" * 76)
print("深入检查：对齐步骤如何处理文本")
print("=" * 76)
align_candidates = [p for p in PRODUCTION
                    if p.suffix == ".py" and ("align" in p.stem or "adapter" in p.stem
                                              or "pipeline" in p.stem)]
for p in sorted(align_candidates):
    text = p.read_text(encoding="utf-8", errors="ignore")
    naive = bool(re.search(r"\.mean\(\s*axis\s*=\s*0\s*\)", text)) or \
            bool(re.search(r"\.mean\(\s*axis\s*=\s*1\s*\)", text))
    uses_pool = "text_pooling" in text or "pool_text_sequence" in text
    uses_mask = "text_mask" in text or "attention_mask" in text
    print(f"\n   {p.name}")
    print(f"      使用 text_pooling      : {uses_pool}")
    print(f"      使用掩码               : {uses_mask}")
    print(f"      出现 .mean(axis=...)   : {naive}")

print("\n" + "=" * 76)
if problems:
    print(f"⚠️  {len(problems)} 个缓解模块未被生产代码调用: {problems}")
    print("    这意味着对应的陷阱只是「有解」而未被「采用」。")
else:
    print("所有缓解模块均已被生产代码引用。")
print("=" * 76)
sys.exit(0)
