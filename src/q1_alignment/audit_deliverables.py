# -*- coding: utf-8 -*-
"""Audit the actual Q1 deliverables on the server.

The guard checks *samples and assumptions*.  This checks the *artifacts*:
does the manifest cover all 100 clips, does the QC report actually flag the
right things, are the word timelines structurally sound, and is the handoff
directory self-consistent?
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/user/MathModel")
WORK = ROOT / "work"
HANDOFF = ROOT / "handoff" / "vision_timing"

fail = 0
warn = 0


def check(ok, label, detail="", level="FAIL"):
    global fail, warn
    if ok:
        print(f"  [ok]   {label}")
    else:
        if level == "FAIL":
            fail += 1
            print(f"  [FAIL] {label}: {detail}")
        else:
            warn += 1
            print(f"  [warn] {label}: {detail}")


print("=" * 72)
print("Q1 交付物审计")
print("=" * 72)

# ---------- manifest ----------
print("\n[1] manifest.jsonl")
man = [json.loads(l) for l in (WORK / "manifest.jsonl").read_text().splitlines() if l.strip()]
check(len(man) == 100, "覆盖 100 条样本", f"实际 {len(man)}")
ids = [m["id"] for m in man]
check(len(set(ids)) == len(ids), "样本 id 无重复", f"{len(ids)-len(set(ids))} 个重复")

have_dur = [m for m in man if m.get("duration_sec", 0) > 0]
check(len(have_dur) == len(man), "每条都有正时长", f"{len(man)-len(have_dur)} 条缺失")
durs = [m["duration_sec"] for m in have_dur]
print(f"         时长 min={min(durs):.3f} max={max(durs):.3f} mean={np.mean(durs):.3f}")

# timestamps must be strictly increasing
bad_mono = 0
for m in man:
    ts = m.get("vision_times")
    if ts and len(ts) > 1 and not np.all(np.diff(ts) > 0):
        bad_mono += 1
check(bad_mono == 0, "视觉时间戳严格递增", f"{bad_mono} 条违反")

# frame count vs duration sanity: fps implied must be plausible
bad_fps = []
for m in man:
    ts = m.get("vision_times")
    if ts and len(ts) > 1 and m["duration_sec"] > 0:
        fps = len(ts) / m["duration_sec"]
        if not (5 <= fps <= 60):
            bad_fps.append((m["id"], round(fps, 1)))
check(len(bad_fps) == 0, "隐含帧率在 5-60 之间", f"{bad_fps[:5]}", level="warn")

# ---------- word timelines ----------
print("\n[2] word_timelines.jsonl")
tl = [json.loads(l) for l in (WORK / "word_timelines.jsonl").read_text().splitlines() if l.strip()]
check(len(tl) == 100, "覆盖 100 条样本", f"实际 {len(tl)}")

by_id = {m["id"]: m for m in man}
bad_cov, bad_order, bad_range, no_words, silent = [], [], [], [], []
for t in tl:
    w = t.get("words", [])
    if not w:
        no_words.append(t["id"])
        continue
    d = by_id.get(t["id"], {}).get("duration_sec", 0)
    # A digitally silent clip has no measurable boundary at all: its words carry
    # the -1.0 sentinel, which would otherwise show up here as a negative
    # coverage AND as a failed monotonicity AND as an out-of-range boundary --
    # three false alarms from one real property of the source material.
    if not (t.get("diagnostics") or {}).get("audio_present", True):
        silent.append(t["id"])
        if any(x["start"] != -1.0 or x["end"] != -1.0 for x in w):
            bad_range.append((t["id"], "sentinel violation", ""))
        continue
    if d > 0:
        cov = (w[-1]["end"] - w[0]["start"]) / d
        if cov < 0.90:
            bad_cov.append((t["id"], round(cov, 3)))
    starts = [x["start"] for x in w]
    ends = [x["end"] for x in w]
    if not (np.all(np.diff(starts) > 0) and np.all(np.diff(ends) > 0)):
        bad_order.append(t["id"])
    if d > 0 and (w[0]["start"] < -1e-6 or w[-1]["end"] > d + 1e-6):
        bad_range.append((t["id"], round(w[-1]["end"], 3), round(d, 3)))

check(len(no_words) == 0, "每条都有词", f"{no_words[:5]}")
check(len(bad_cov) == 0, "覆盖率全部 >= 0.90（仅统计可发声样本）", f"{bad_cov[:5]}")
check(len(bad_order) == 0, "词边界严格单调（仅统计可发声样本）", f"{bad_order[:5]}")
check(len(bad_range) == 0, "词边界不越出时长 / 静音样本恰为哨兵", f"{bad_range[:5]}")
check(len(silent) == 2, "音轨数字静音样本恰为 2 条且已被识别", f"{silent}")
print(f"     音轨数字静音（无词级时间，已从覆盖率/单调性统计中剔除）: {silent}")

# words must not overlap
overlap = []
for t in tl:
    w = t.get("words", [])
    for a, b in zip(w, w[1:]):
        if a["end"] > b["start"] + 1e-6:
            overlap.append(t["id"])
            break
check(len(overlap) == 0, "相邻词不重叠", f"{overlap[:5]}")

# ---------- qc report ----------
print("\n[3] qc_report.csv")
rows = list(csv.DictReader((WORK / "qc_report.csv").open()))
check(len(rows) == 100, "覆盖 100 条", f"实际 {len(rows)}")
statuses = {}
for r in rows:
    statuses[r.get("status", "?")] = statuses.get(r.get("status", "?"), 0) + 1
print(f"         status 分布: {statuses}")
check("fail" not in statuses, "无 fail 状态", f"{statuses}")

# ---------- handoff ----------
print("\n[4] handoff/vision_timing")
samples = sorted((HANDOFF / "samples").iterdir())
check(len(samples) == 100, "覆盖 100 条样本目录", f"实际 {len(samples)}")
lookup = list(csv.DictReader((HANDOFF / "frame_count_lookup.csv").open()))
check(len(lookup) == 100, "frame_count_lookup 覆盖 100 条", f"实际 {len(lookup)}")

bad_files = []
for s in samples:
    ts = s / "vision_timestamps.npy"
    meta = s / "vision_frame_meta.csv"
    if not ts.exists() or not meta.exists():
        bad_files.append(s.name)
        continue
    arr = np.load(ts)
    mrows = list(csv.DictReader(meta.open()))
    if len(arr) != len(mrows):
        bad_files.append(f"{s.name}(npy={len(arr)} csv={len(mrows)})")
    if len(arr) > 1 and not np.all(np.diff(arr) > 0):
        bad_files.append(f"{s.name}(not monotonic)")
check(len(bad_files) == 0, "每样本 npy/csv 行数一致且单调", f"{bad_files[:5]}")

# ---------- cross-check handoff vs manifest ----------
print("\n[5] handoff 与 manifest 一致性")
mismatch = []
for s in samples:
    sid = s.name
    arr = np.load(s / "vision_timestamps.npy")
    # safe_id replaces problematic chars; find manifest entry by suffix match
    cand = [m for m in man if m["id"].replace("$", "__").replace("/", "_") == sid
            or m["id"].replace("$_$", "__") == sid]
    if not cand:
        mismatch.append(f"{sid}: 无对应 manifest 条目")
        continue
    mt = cand[0].get("vision_times")
    if mt is not None and len(mt) != len(arr):
        mismatch.append(f"{sid}: len {len(mt)} vs {len(arr)}")
check(len(mismatch) == 0, "handoff 与 manifest 帧数一致", f"{mismatch[:5]}")

print("\n" + "=" * 72)
print(f"FAIL={fail}  WARN={warn}")
print("=" * 72)
sys.exit(1 if fail else 0)
