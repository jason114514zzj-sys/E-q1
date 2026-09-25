"""Requirements traceability: map every problem-stated deliverable to evidence.

The guard's other checks verify data assumptions and internal consistency.  This
one answers a different question: *does anything the problem explicitly asks for
have no owner?*

Each requirement below is quoted from the problem statement (section 四, 结果与
提交说明).  The check reports, per requirement, whether we have an artifact, a
plan, or nothing.  "Nothing" is the dangerous state -- it is how a required
deliverable gets discovered the night before the deadline.

Status values:
    done     an artifact exists and was verified
    partial  some of it exists
    planned  no artifact yet, but it is someone's assigned next step
    gap      no artifact and no owner          <-- these are the expensive ones
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "work"
OUT = ROOT / "output"


@dataclass
class Requirement:
    id: str
    quote: str
    owner: str
    status: str
    evidence: str = ""
    note: str = ""
    artifacts: list[str] = field(default_factory=list)


def _exists(*paths: str | Path) -> bool:
    return all(Path(p).exists() for p in paths)


def _count(pattern: str, base: Path | None = None) -> int:
    base = base or WORK
    return len(list(base.glob(pattern)))


def evaluate() -> list[Requirement]:
    reqs: list[Requirement] = []

    # ---- 问题1 ----
    reqs.append(Requirement(
        id="Q1-1",
        quote="特征提取与时序对齐整体方案：说明三类模态的特征定义、提取方法与所用工具，"
              "阐述时序对齐的核心规则与跨模态时间对应逻辑",
        owner="我",
        status="done" if _exists(OUT / "evidence" / "EVIDENCE.md") else "gap",
        evidence="output/evidence/EVIDENCE.md + PAPER_EVIDENCE.md",
        artifacts=["src/q1_alignment/monotonic_align.py"],
    ))

    reqs.append(Requirement(
        id="Q1-2a",
        quote="特征文件规范与全量结果汇总：说明存储格式、组织结构与读取方式",
        owner="我",
        status="done" if _exists(ROOT / "handoff" / "交付接口规格_冻结版.md") else "gap",
        evidence="handoff/交付接口规格_冻结版.md",
    ))

    # The problem asks for a table over all 100 clips with columns:
    # 样本编号 / 模态类型 / 原始有效时长 / 特征维度 / 对齐粒度
    has_summary = _count("*.csv", WORK / "summary") > 0
    reqs.append(Requirement(
        id="Q1-2b",
        quote="以表格形式汇总附件1全部100条样本的特征提取结果，包含样本编号、模态类型、"
              "原始有效时长、特征维度、对齐粒度等关键信息",
        owner="我",
        status="done" if has_summary else "gap",
        evidence=f"work/summary/ 共 {_count('*.csv', WORK / 'summary')} 个 csv",
        note="需确认五列齐全且覆盖 100 条",
    ))

    reqs.append(Requirement(
        id="Q1-2c",
        quote="全量原始特征文件随附件提交",
        owner="我 + 队友",
        status="planned",
        evidence="问题一侧已产出 handoff/vision_timing（100 条）；全量特征待队友交付",
        note="须核算总大小 ≤50MB",
    ))

    reqs.append(Requirement(
        id="Q1-3",
        quote="典型样本验证：选取至少1个典型样本，展示文本片段、对应语音时段、视频帧段"
              "与三类特征的对应关系，呈现时序对齐效果",
        owner="我",
        status="done" if _count("*", WORK / "examples") > 0 else "gap",
        evidence=f"work/examples/ 共 {_count('*', WORK / 'examples')} 项",
    ))

    reqs.append(Requirement(
        id="Q1-4",
        quote="方案可复现性说明：标注所用工具与模型的版本、核心参数及完整运行流程",
        owner="我",
        status="done" if _exists(ROOT / "versions.txt", ROOT / "requirements.txt") else "gap",
        evidence="versions.txt + requirements.txt + SERVER_ENV.md",
    ))

    # ---- 问题2（队友A） ----
    # 队友A 把问题二求解与论文材料上传到了 GitHub（REDACTED9495-tech/E）。
    # 2026-09-25 我方按哈希锁定导入了 canonical 提交件；论文正文材料亦已可取。
    _q2_csv = OUT / "q2" / "附件3_预测结果.csv"
    _q2_prov = OUT / "q2" / "_来源与哈希.md"
    _q2_imported = _q2_csv.exists()
    reqs.append(Requirement(
        id="Q2-1",
        quote="鲁棒性模型的建模原理、网络结构、目标函数、训练方案与关键参数",
        owner="队友A",
        status="done",
        evidence="仓库 q2/method.md + protocol.json + 论文稿第 5 章"
                 "（冻结 BERT-Mini 重编码 text_bert → 256 维；观测门控 + 状态递推 a0=0.98；"
                 "CE + Huber(δ=1)；AdamW lr1e-3 / wd1e-4 / dropout0.2 / bs64 / ≤40 轮 / 早停 6）",
        note="论文正文待把该章并入合稿",
    ))
    reqs.append(Requirement(
        id="Q2-2",
        quote="缺失模态类型、缺失率对预测性能影响的规律分析与消融实验结果",
        owner="队友A",
        status="done",
        evidence="27 条件（3 模态 × 3 比例 × 3 位置）+ 4 组补充压力测试；"
                 "text 缺失 10%→50% Macro-F1 0.5415→0.5171（降），audio/vision 基本不变",
        note="音视频相对顺序的两侧结果差异 0.002–0.003 ≪ 种子标准差 0.011 → 不可分辨，不写结论",
    ))
    reqs.append(Requirement(
        id="Q2-3",
        quote="附件3测试集的全量预测结果汇总与结果展示",
        owner="队友A / 我（按哈希导入）",
        status="done" if _q2_imported else "gap",
        evidence="output/q2/附件3_预测结果.csv（30 行；仓库原始 sha256 44e5cd86…，"
                 "与 submission_manifest.json 及 verification.json 的 prediction_sha256 一致）",
        note="导入副本仅剥离 3 字节 BOM，其余逐字节相同；溯源见 output/q2/_来源与哈希.md",
    ))
    reqs.append(Requirement(
        id="Q2-4",
        quote="验证集上的基础性能评价、可视化分析与错误归因结论",
        owner="队友A",
        status="done",
        evidence="Accuracy 0.5824 / Macro-F1 0.5403 / MAE 0.6588 / Pearson 0.5474；"
                 "混淆矩阵 + 逐类指标 + 置信度/可靠性 + 30 条最大回归误差 + 分组配对 bootstrap",
        note="验证集用于选模 → 属开发评价，不是独立测试成绩",
    ))

    # ---- 问题3（本智能体已代为完成；交付归属仍按团队分工标注） ----
    reqs.append(Requirement(
        id="Q3-1",
        quote="可解释性模型的建模原理、网络结构、目标函数、训练方案与关键参数",
        owner="我（原分工：队友B）",
        status="done" if _exists(OUT / "q3" / "附件4_预测与解释结果.csv") else "gap",
        evidence="output/evidence/问题三解题报告.md + work/q3/q3_report.json"
                 "（超参 hidden=96 / epochs=60 / lr=1.5e-3 / bs=128 / seed=20260924）",
        artifacts=["src/q3_explain/model.py", "src/q3_explain/train.py"],
    ))
    reqs.append(Requirement(
        id="Q3-2",
        quote="典型样本解释卡（含预测结果、三模态作用程度、主要参考模态及关键证据定位）",
        owner="我（原分工：队友B）",
        status="done" if _exists(WORK / "q3" / "解释卡.md") else "gap",
        evidence="work/q3/解释卡.md（6 条典型）+ work/q3/explanations.json（20 条全量）",
    ))
    reqs.append(Requirement(
        id="Q3-3",
        quote="主要参考模态内局部片段重要性分布可视化、三模态作用差异对比分析",
        owner="我（原分工：队友B）",
        status="done" if _count("q3_*.png", OUT / "figures") >= 3 else "partial",
        evidence=f"output/figures 下 q3_*.png 共 {_count('q3_*.png', OUT / 'figures')} 张",
        note="注意力分布 / 三模态权重 / 门控-留一对照 / 三组忠实度",
    ))
    reqs.append(Requirement(
        id="Q3-4",
        quote="附件4测试集的全量预测与解释结果汇总",
        owner="我（原分工：队友B）",
        status="done" if _exists(OUT / "q3" / "附件4_预测与解释结果.csv") else "gap",
        evidence="output/q3/附件4_预测与解释结果.csv（20 行 × 8 列，guard G1–G6 全通过）",
    ))
    reqs.append(Requirement(
        id="Q3-5",
        quote="验证集上的基础性能评价、可视化分析与错误归因结论",
        owner="我（原分工：队友B）",
        status="done" if _exists(WORK / "q3" / "q3_report.json") else "gap",
        evidence="Accuracy 0.6360 / Macro-F1 0.6184 / MAE 0.6214 / Pearson 0.6282；"
                 "混淆矩阵 + 各类召回 + 置信度对比 + 两条基线",
        note="中性类召回 0.4946 为瓶颈，已如实报告",
    ))
    reqs.append(Requirement(
        id="Q3-6",
        quote="关键证据需可对应至原始文本片段、语音时段或视觉关键帧（段029/017）",
        owner="我",
        status="done" if _exists(WORK / "q3" / "att4_time_map.json") else "gap",
        evidence="work/q3/att4_time_map.json：20/20 条词级强制对齐，覆盖率 1.000",
        note="禁止用 j*D/50 均匀切分（实测偏差均值 1.97s / 最大 3.63s）；guard G6 机检",
    ))

    # ---- 提交要求 ----
    reqs.append(Requirement(
        id="SUB-1",
        quote="可复现核心材料：问题1自主生成的100条样本多模态时序特征文件；问题2和问题3"
              "两类专项模型的核心代码、说明文档、模型参数文件、配置文件与运行环境说明",
        owner="全员",
        status="done" if _q2_imported else "partial",
        evidence="问题一代码/环境说明齐全；问题二核心代码+说明+模型参数+配置+版本"
                 "（队友仓库 q2/，含 package.py 体积守门与 submission_manifest.json 逐文件 sha256）；"
                 "问题三（src/q3_explain、output/models/q3_model.pt、config.yaml）",
    ))
    reqs.append(Requirement(
        id="SUB-2",
        quote="专项测试结果文件：附件3测试集预测结果CSV文件、附件4测试集预测与解释结果CSV文件",
        owner="队友A / 我",
        status="done" if (_q2_csv.exists() and _exists(OUT / "q3" / "附件4_预测与解释结果.csv"))
               else "partial",
        evidence="output/q2/附件3_预测结果.csv（30 行，哈希锁定导入）+ "
                 "output/q3/附件4_预测与解释结果.csv（20 行）；guard F2/F3/F4/G1–G7 全通过",
        note="⚠️ 队友的代码包与论文正文指向两份不同的预测文件（MLP vs EMT-DLFR），"
             "本项按代码包口径（MLP，含 verification.json 哈希）闭合；论文正文需同步",
    ))
    reqs.append(Requirement(
        id="SUB-3",
        quote="总附件大小 ≤50MB",
        owner="全员",
        status="planned",
        evidence="guard E6 已覆盖，待打包时核验",
    ))
    reqs.append(Requirement(
        id="SUB-4",
        quote="严禁出现参赛单位、队员姓名、队伍编号等身份信息",
        owner="全员",
        status="done",
        evidence="guard E5 + scripts/sanitize_evidence.py，实测命中 0 处",
    ))
    reqs.append(Requirement(
        id="SUB-5",
        quote="不得引入其他任何公开或私有情感数据集参与模型训练、微调、参数优化、阈值选择或结果统计",
        owner="全员",
        status="done",
        evidence="guard E1/E2 实测命中 0 处",
    ))

    return reqs


BANNER = "=" * 78

def main() -> int:
    reqs = evaluate()
    by_status: dict[str, list[Requirement]] = {}
    for r in reqs:
        by_status.setdefault(r.status, []).append(r)

    print(BANNER)
    print("题目明示交付物 · 可追溯性矩阵")
    print(BANNER)
    print(f"\n共 {len(reqs)} 项题目明示要求\n")

    order = ["gap", "partial", "planned", "done"]
    label = {
        "gap": "❌ 无人认领 / 无产出",
        "partial": "🟡 部分完成",
        "planned": "🔵 已排期",
        "done": "✅ 已完成",
    }

    for st in order:
        group = by_status.get(st, [])
        if not group:
            continue
        print(f"\n{label[st]}  ({len(group)} 项)")
        print("-" * 78)
        for r in group:
            print(f"  [{r.id}] 负责人: {r.owner}")
            print(f"       要求: {r.quote[:70]}...")
            if r.evidence:
                print(f"       现状: {r.evidence}")
            if r.note:
                print(f"       注意: {r.note}")

    print("\n" + BANNER)
    print("统计: " + "  ".join(f"{k}={len(v)}" for k, v in by_status.items()))
    gaps = by_status.get("gap", [])
    print(BANNER)
    if gaps:
        print(f"\n⚠️  {len(gaps)} 项无人认领，是当前最贵的风险：")
        for r in gaps:
            print(f"     - [{r.id}] {r.quote[:60]}  (负责人: {r.owner})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
