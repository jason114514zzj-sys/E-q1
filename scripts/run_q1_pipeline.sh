#!/usr/bin/env bash
# ============================================================================
# 问题一 一键全流程
#
# 队友特征一到位，跑这一条命令即可产出问题一的全部结果。
#
#   bash scripts/run_q1_pipeline.sh                    # 完整流程（需已交付特征）
#   bash scripts/run_q1_pipeline.sh --dry-run          # 只跑到校验，不依赖特征
#   bash scripts/run_q1_pipeline.sh --handoff <路径>   # 指定交付目录
#
# 产出：
#   work/word_timelines.jsonl      逐词真实时间轴（100条）
#   work/aligned_real/*.npz        三模态对齐序列（含四类掩码）
#   work/alignment_trace.csv       逐位置追溯表
#   work/qc_report.csv             质量检查报告
#   work/summary/*.csv             全量结果汇总表
#   work/examples/*                典型样本验证材料
#   output/evidence/*              论文证据表
#   output/figures/*.png           论文图表
# ============================================================================
set -euo pipefail

HANDOFF="${HANDOFF:-$HOME/MathModel/handoff/q1_feature_handoff}"
DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --handoff) HANDOFF="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "unknown option: $1"; exit 2 ;;
  esac
done

cd "$HOME/MathModel"
export PYTHONPATH="$HOME/MathModel/src"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate mosei

FFMPEG="$HOME/miniconda3/envs/mosei/bin/ffmpeg"
MANIFEST="work/manifest.jsonl"
TIMELINES="work/word_timelines.jsonl"

DATA_ROOT=$(python - <<'PYEOF'
from pathlib import Path
for c in (Path.home()/"MathModel"/"data").rglob("label-100.xlsx"):
    print(c.parent); break
PYEOF
)

banner() { echo; echo "########## $* ##########"; }

banner "0/8 单元测试"
python -m unittest discover -s src/q1_alignment/tests -t src 2>&1 | tail -3

banner "1/8 构建 manifest（ffprobe 真实时长）"
python -m q1_alignment.cli build-manifest \
  --data-root "$DATA_ROOT" --output "$MANIFEST"

banner "2/8 词级时间轴（约束式单调对齐）"
python -m q1_alignment.cli align-transcripts \
  --manifest "$MANIFEST" --data-root "$DATA_ROOT" \
  --output "$TIMELINES" --method monotonic \
  --ffmpeg "$FFMPEG" --report work/word_align_report.csv

banner "3/8 校验队友交付（validate-handoff）"
if [[ -d "$HANDOFF" ]]; then
  python -m q1_alignment.cli validate-handoff \
    --handoff "$HANDOFF" --manifest "$MANIFEST" \
    --report work/handoff_validation.json || {
      echo "!! 交付校验未通过，详见 work/handoff_validation.json"; exit 1; }
else
  echo "!! 未找到交付目录 $HANDOFF"
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "   (--dry-run 模式，跳过后续依赖特征的步骤)"
  else
    echo "   请用 --handoff 指定，或把特征放到该路径"
    exit 1
  fi
fi

if [[ $DRY_RUN -eq 1 ]]; then
  banner "dry-run 结束"
  echo "以下步骤需要队友特征："
  echo "  4/8 导入  5/8 对齐  6/8 报告  7/8 汇总  8/8 图表"
  exit 0
fi

banner "4/8 导入交付（文档格式 -> 内部 npz）"
python -m q1_alignment.cli import-handoff \
  --handoff "$HANDOFF" --manifest "$MANIFEST" \
  --output-dir work/raw_features --report work/handoff_import.json

banner "5/8 三模态对齐（50 位置 + 四类掩码）"
# --allow-partial lets a 3-sample pilot run before all 100 features exist,
# as the collaboration document recommends for the joint commissioning step.
python -m q1_alignment.cli align \
  --manifest "$MANIFEST" --timelines "$TIMELINES" \
  --feature-dir work/raw_features --output-dir work/aligned_real \
  --allow-partial

banner "6/8 追溯表与 QC 报告"
# Detect whether every sample has features yet, so a pilot run reports as
# "partial" instead of falsely claiming 97 failures.
PARTIAL_FLAG=""
FEATURE_COUNT=$(ls work/raw_features/*.npz 2>/dev/null | wc -l)
if [[ "$FEATURE_COUNT" -lt 100 ]]; then
  PARTIAL_FLAG="--allow-partial"
  echo "注意：当前仅有 $FEATURE_COUNT / 100 条特征，按部分运行处理"
fi
python - "$PARTIAL_FLAG" <<'PYEOF'
import sys
from pathlib import Path
sys.path.insert(0, str(Path.home()/"MathModel"/"src"))
from q1_alignment.reports import build_alignment_trace, build_qc_report
allow_partial = "--allow-partial" in sys.argv
w = Path.home()/"MathModel"/"work"
print("trace rows:", build_alignment_trace(w/"manifest.jsonl", w/"aligned_real", w/"alignment_trace.csv"))
print("qc       :", build_qc_report(w/"manifest.jsonl", w/"aligned_real", w/"qc_report.csv",
                                    allow_partial=allow_partial))
PYEOF

banner "7/8 全量汇总表 + 典型样本"
python -m q1_alignment.make_summary \
  --manifest "$MANIFEST" --timelines "$TIMELINES" \
  --aligned-dir work/aligned_real --output-dir work/summary

python -m q1_alignment.make_examples \
  --manifest "$MANIFEST" --timelines "$TIMELINES" \
  --data-root "$DATA_ROOT" --output-dir work/examples \
  --ffmpeg "$FFMPEG" --limit 6

banner "8/8 论文证据表与图表"
python -m q1_alignment.make_evidence
python -m q1_alignment.make_figures

banner "完成"
python - <<'PYEOF'
import csv, statistics, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path.home()/"MathModel"/"src"))
from q1_alignment.common import read_jsonl
w = Path.home()/"MathModel"/"work"
man = {r["id"]: r for r in read_jsonl(w/"manifest.jsonl")}
cov = []
for r in read_jsonl(w/"word_timelines.jsonl"):
    ws = r.get("words", [])
    if not ws: continue
    cov.append((float(ws[-1]["end"]) - float(ws[0]["start"])) / man[r["id"]]["duration_sec"])
print(f"对齐覆盖率: min={min(cov):.4f} median={statistics.median(cov):.4f} n={len(cov)}")

with (w/"qc_report.csv").open(encoding="utf-8") as fh:
    qc = list(csv.DictReader(fh))
print("QC 状态分布:", dict(Counter(r["status"] for r in qc)))
print()
print("产出目录:")
for p in sorted((Path.home()/"MathModel").glob("work/*")):
    if p.is_file() or p.name in ("summary", "examples", "aligned_real"):
        print("  ", p)
PYEOF
