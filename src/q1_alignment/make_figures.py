"""Paper figures for the problem-1 alignment evidence.

Figures:
  fig1_coverage_comparison.png  coverage distribution, old vs new
  fig2_coverage_scatter.png     per-sample coverage, old vs new
  fig3_pause_ablation.png       boundary energy, snapping on vs off
  fig4_timing_defect.png        declared vs real frame count

Chinese labels use a CJK font when one is available; otherwise the figures fall
back to English labels so they never render as boxes.
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .common import read_jsonl  # noqa: E402


def _configure_font() -> bool:
    """Enable a CJK font if present. Returns True when Chinese can be drawn."""

    candidates = [
        "Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei",
        "WenQuanYi Micro Hei", "Source Han Sans CN", "SimHei",
        "Microsoft YaHei", "AR PL UMing CN",
    ]
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            plt.rcParams["axes.unicode_minus"] = False
            return True
    return False


CJK = _configure_font()
L = (lambda zh, en: zh) if CJK else (lambda zh, en: en)


def _coverage_map(path: Path, manifest: dict) -> dict[str, float]:
    """Per-sample coverage, keyed by id.

    Returned as a mapping rather than a list because the two series are no
    longer the same length: the v1.1 run withholds word times on the two
    digitally silent clips, so a positional pairing would compare each clip
    against an unrelated one (and, before this fix, index past the end of the
    shorter list).
    """

    out: dict[str, float] = {}
    for record in read_jsonl(path):
        words = record.get("words", [])
        if not words:
            continue
        # A clip whose audio track is digitally silent has no measurable word
        # boundary at all; its words carry the -1.0 sentinel, which would show
        # up here as a bogus coverage of 0.  Excluded rather than plotted.
        if not (record.get("diagnostics") or {}).get("audio_present", True):
            continue
        duration = float(manifest[record["id"]]["duration_sec"])
        if duration <= 0:
            continue
        out[record["id"]] = (
            float(words[-1]["end"]) - float(words[0]["start"])
        ) / duration
    return out


def _paired(
    old: dict[str, float], new: dict[str, float]
) -> tuple[list[str], list[float], list[float]]:
    """Ids present in both series, with their coverages, in a stable order."""

    ids = sorted(set(old) & set(new))
    return ids, [old[i] for i in ids], [new[i] for i in ids]


def _coverage_series(path: Path, manifest: dict) -> list[float]:
    """Backward-compatible wrapper: coverage values only."""

    return list(_coverage_map(path, manifest).values())


def fig_coverage_comparison(old: list[float], new: list[float], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    bins = np.linspace(0, 1, 21)
    ax.hist(old, bins=bins, alpha=0.7, label=L("分段法 (v1.0)", "Span-wise (v1.0)"),
            color="#d9534f", edgecolor="white")
    ax.hist(new, bins=bins, alpha=0.7, label=L("约束式 (v1.1)", "Constrained (v1.1)"),
            color="#2e7d32", edgecolor="white")
    ax.set_xlabel(L("对齐覆盖率", "Alignment coverage"))
    ax.set_ylabel(L("样本数", "Sample count"))
    ax.set_title(L("覆盖率分布对比", "Coverage distribution"))
    ax.legend()
    ax.grid(alpha=0.25)

    ax = axes[1]
    metrics = [
        (L("最小值", "Min"), min(old), min(new)),
        (L("中位数", "Median"), statistics.median(old), statistics.median(new)),
        (L("均值", "Mean"), statistics.mean(old), statistics.mean(new)),
    ]
    x = np.arange(len(metrics))
    width = 0.36
    ax.bar(x - width / 2, [m[1] for m in metrics], width,
           label=L("分段法 (v1.0)", "Span-wise (v1.0)"), color="#d9534f")
    ax.bar(x + width / 2, [m[2] for m in metrics], width,
           label=L("约束式 (v1.1)", "Constrained (v1.1)"), color="#2e7d32")
    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in metrics])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel(L("覆盖率", "Coverage"))
    ax.set_title(L("覆盖率关键指标", "Key coverage metrics"))
    for index, metric in enumerate(metrics):
        ax.text(index - width / 2, metric[1] + 0.03, f"{metric[1]:.3f}",
                ha="center", fontsize=8)
        ax.text(index + width / 2, metric[2] + 0.03, f"{metric[2]:.3f}",
                ha="center", fontsize=8)
    ax.legend()
    ax.grid(alpha=0.25, axis="y")

    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_coverage_scatter(old: dict, new: dict, out: Path) -> None:
    ids, old_v, new_v = _paired(old, new)
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    order = np.argsort(old_v)
    y = np.arange(len(order))
    ax.scatter(np.array(old_v)[order], y, s=12, color="#d9534f",
               label=L("分段法 (v1.0)", "Span-wise (v1.0)"), alpha=0.8)
    ax.scatter(np.array(new_v)[order], y, s=12, color="#2e7d32",
               label=L("约束式 (v1.1)", "Constrained (v1.1)"), alpha=0.8)
    ax.axvline(0.9, color="gray", linestyle="--", linewidth=1)
    ax.text(0.9, len(y) * 0.02, L(" 0.90 阈值", " 0.90 threshold"),
            color="gray", fontsize=8, rotation=90)
    ax.set_xlabel(L("覆盖率", "Coverage"))
    ax.set_ylabel(L(f"样本序号（{len(y)} 条共有样本，按旧方法覆盖率排序）",
                   f"Sample rank ({len(y)} shared samples, sorted by v1.0)"))
    ax.set_title(L("逐样本覆盖率对比", "Per-sample coverage"))
    ax.set_xlim(0, 1.05)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_pause_ablation(data: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    values = [data["energy_off"], data["energy_on"]]
    bars = ax.bar([L("关闭吸附", "Snapping off"), L("开启吸附", "Snapping on")],
                  values, color=["#9e9e9e", "#2e7d32"], width=0.55)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.03,
                f"{value:.4f}", ha="center", fontsize=9)
    ax.set_ylabel(L("边界处归一化能量（越低越好）",
                    "Normalised energy at boundaries (lower is better)"))
    ax.set_title(L(f"停顿吸附消融实验\n边界能量降低 {data['relative_gain_pct']:.2f}%，"
                   f"{data['improved']}/{data['samples']} 条改善",
                   f"Pause snapping ablation\n{data['relative_gain_pct']:.2f}% lower, "
                   f"{data['improved']}/{data['samples']} improved"))
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_timing_defect(manifest: dict, out: Path) -> None:
    declared = np.array([int(r["declared_frame_count"]) for r in manifest.values()])
    real = np.array([int(r["frame_count"]) for r in manifest.values()])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    ax.scatter(real, declared, s=14, alpha=0.7, color="#1565c0")
    limit = max(declared.max(), real.max()) * 1.05
    ax.plot([0, limit], [0, limit], "k--", linewidth=1,
            label=L("理想情况 y=x", "Ideal y=x"))
    ax.set_xlabel(L("实际可解码帧数", "Decodable frames"))
    ax.set_ylabel(L("元数据声明帧数", "Declared frames"))
    ax.set_title(L("帧数元数据缺陷", "Frame-count metadata defect"))
    ax.legend()
    ax.grid(alpha=0.25)

    ax = axes[1]
    diff = declared - real
    ax.hist(diff, bins=30, color="#ef6c00", edgecolor="white")
    ax.axvline(0, color="k", linestyle="--", linewidth=1)
    ax.set_xlabel(L("虚报帧数（声明 − 实际）", "Over-reported frames"))
    ax.set_ylabel(L("样本数", "Sample count"))
    ax.set_title(L(f"虚报分布（{int((diff>0).sum())}/{len(diff)} 条虚报）",
                   f"Over-report distribution ({int((diff>0).sum())}/{len(diff)})"))
    ax.grid(alpha=0.25, axis="y")

    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    import argparse

    home = Path.home()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", default=str(home / "MathModel/work"))
    parser.add_argument("--output-dir", default=str(home / "MathModel/output/figures"))
    args = parser.parse_args()

    work = Path(args.work_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest = {r["id"]: r for r in read_jsonl(work / "manifest.jsonl")}
    old = _coverage_map(work / "word_timelines_energy.jsonl", manifest)
    new = _coverage_map(work / "word_timelines.jsonl", manifest)
    ids, old_v, new_v = _paired(old, new)

    fig_coverage_comparison(old_v, new_v, out / "fig1_coverage_comparison.png")
    fig_coverage_scatter(old, new, out / "fig2_coverage_scatter.png")

    ablation_path = work / "ablation_pause.json"
    if ablation_path.exists():
        fig_pause_ablation(
            json.loads(ablation_path.read_text(encoding="utf-8")),
            out / "fig3_pause_ablation.png",
        )

    fig_timing_defect(manifest, out / "fig4_timing_defect.png")

    print(json.dumps({
        "cjk_font": CJK,
        "figures": [p.name for p in sorted(out.glob("fig*.png"))],
        "output_dir": str(out),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
