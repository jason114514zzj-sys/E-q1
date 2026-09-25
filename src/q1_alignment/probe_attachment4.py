"""Probe attachment 4 (interpretability test set) and its paired videos.

Checks what the three-modality feature files contain, whether they match
attachment 2's layout, and whether the videos can be aligned back to the
features (the interpretability task requires locating evidence in the video).
"""

from __future__ import annotations

import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

HOME = Path.home()
DATA = HOME / "MathModel" / "data"
ATT4 = DATA / "附件4-可解释专项视频样本与特征文件"
FFPROBE = str(HOME / "miniconda3/envs/mosei/bin/ffprobe")


def probe(path: Path) -> dict:
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-show_entries",
         "format=duration:stream=codec_type,duration,nb_frames,r_frame_rate,width,height",
         "-of", "json", str(path)],
        capture_output=True, text=True)
    import json
    try:
        return json.loads(out.stdout)
    except Exception:
        return {}


def frame_times(path: Path) -> list[float]:
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "frame=pts_time", "-of", "json", str(path)],
        capture_output=True, text=True)
    import json
    try:
        frames = json.loads(out.stdout).get("frames", [])
    except Exception:
        return []
    return [float(f["pts_time"]) for f in frames if "pts_time" in f]


def main() -> None:
    print("=" * 78)
    print("附件4 结构探查")
    print("=" * 78)

    # locate the nested directory (the attachment has a duplicated level)
    roots = [p for p in ATT4.rglob("*") if p.is_dir() and (p / "对齐版本").exists()]
    if not roots:
        print("未找到对齐版本目录")
        return
    base = roots[0]
    print(f"根目录: {base.relative_to(ATT4.parent)}")

    for version in ("对齐版本", "未对齐版本"):
        directory = base / version
        pkls = sorted(directory.glob("*.pkl"))
        videos = sorted((directory / "videos").glob("*.mp4")) if (directory / "videos").exists() else []
        print(f"\n{'='*78}")
        print(f"{version}: {len(pkls)} 个 pkl, {len(videos)} 个视频")
        print("=" * 78)

        if pkls:
            with pkls[0].open("rb") as handle:
                obj = pickle.load(handle)
            print(f"\n--- {pkls[0].name} 结构 ---")
            print(f"  顶层类型: {type(obj).__name__}")
            if isinstance(obj, dict):
                print(f"  顶层键: {list(obj.keys())}")
                for key, value in obj.items():
                    if isinstance(value, dict):
                        print(f"    [{key}] dict keys={list(value.keys())}")
                        for k2, v2 in value.items():
                            if isinstance(v2, np.ndarray):
                                print(f"        {k2:20s} {str(v2.shape):18s} {v2.dtype}")
                            elif isinstance(v2, list):
                                print(f"        {k2:20s} list[{len(v2)}]")
                                if v2 and isinstance(v2[0], str):
                                    print(f"            [0]={v2[0][:70]}")
                    elif isinstance(value, np.ndarray):
                        print(f"    [{key}] ndarray {value.shape} {value.dtype}")
                    else:
                        print(f"    [{key}] {type(value).__name__}: {str(value)[:70]}")

        # scan all for missing / lengths
        print(f"\n  扫描 {len(pkls)} 个文件的完整性:")
        for path in pkls[:3]:
            with path.open("rb") as handle:
                obj = pickle.load(handle)
            block = obj.get("test", obj)
            for key in ("text", "audio", "vision", "text_bert"):
                if key in block:
                    arr = np.asarray(block[key])
                    if arr.ndim >= 2 and arr.dtype.kind in "fc":
                        zero = np.all(arr == 0, axis=-1)
                        if zero.ndim > 1:
                            zero = zero.reshape(zero.shape[0], -1).all(axis=1)
                        n_zero = int(zero.sum())
                        print(f"    {path.name} {key:10s} {str(arr.shape):16s} "
                              f"全零步={n_zero}/{zero.shape[0]}")

    # video analysis
    print(f"\n{'='*78}")
    print("视频分析（可解释性需要用视频回看证据位置）")
    print("=" * 78)
    vdir = base / "对齐版本" / "videos"
    if vdir.exists():
        vids = sorted(vdir.glob("*.mp4"))
        print(f"\n共 {len(vids)} 个视频")
        print(f"\n{'文件':10s} {'真实时长':>10s} {'帧数':>7s} {'fps':>6s} {'分辨率':>12s}")
        for vid in vids[:8]:
            info = probe(vid)
            dur = float(info.get("format", {}).get("duration", 0) or 0)
            vstream = next((s for s in info.get("streams", [])
                            if s.get("codec_type") == "video"), {})
            frames = len(frame_times(vid))
            fps = vstream.get("r_frame_rate", "?")
            size = f"{vstream.get('width','?')}x{vstream.get('height','?')}"
            print(f"{vid.name:10s} {dur:10.3f} {frames:7d} {fps:>6s} {size:>12s}")

        # consistency between pkl 50 positions and video
        print(f"\n  pkl 位置数(50) 与视频帧数是否可用于定位？")
        v0 = vids[0]
        frames0 = len(frame_times(v0))
        dur0 = float(probe(v0).get("format", {}).get("duration", 0) or 0)
        print(f"    视频 {v0.name}: {dur0:.3f}s, {frames0} 帧")
        print(f"    对应 pkl: 50 个位置 -> 每位置约 {dur0/50:.3f}s")
        print(f"    -> 可建立「位置索引 -> 视频时间」的映射，用于关键证据定位")

    # are the two versions' videos identical?
    a = base / "对齐版本" / "videos"
    b = base / "未对齐版本" / "videos"
    if a.exists() and b.exists():
        import hashlib
        def digests(d):
            return {p.name: hashlib.md5(p.read_bytes()).hexdigest()
                    for p in sorted(d.glob("*.mp4"))}
        da, db = digests(a), digests(b)
        same = sum(1 for k in da if k in db and da[k] == db[k])
        print(f"\n  两版本视频是否相同: {same}/{len(da)} 个文件哈希一致")
        only_a = set(da) - set(db)
        only_b = set(db) - set(da)
        if only_a or only_b:
            print(f"    仅对齐版有: {sorted(only_a)}")
            print(f"    仅未对齐版有: {sorted(only_b)}")


if __name__ == "__main__":
    sys.exit(main())
