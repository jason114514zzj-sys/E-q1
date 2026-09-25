"""Compare the collaboration-document handoff format with the current loader.

This is a documentation/verification script: it prints the field-by-field
mismatch between what the collaboration technical document requires the
feature-extraction teammate to deliver, and what ``load_bundle`` accepts today.
Running it makes the integration work explicit instead of assumed.
"""

from __future__ import annotations

# What the collaboration document (section 三) requires the teammate to deliver,
# per sample, under samples/<sample_id>/
DOC_HANDOFF = {
    "metadata.json": "该样本的实际文件信息与异常说明",
    "text_words.csv": "word_idx, word —— 逐词文本记录",
    "text_features.npy": "(词数, D_text) float32 —— 逐词向量，子词取平均",
    "audio_features.npy": "(Ta, Da) —— 逐帧低层声学特征",
    "audio_intervals.npy": "(Ta, 2) —— 每帧 [start_sec, end_sec]",
    "vision_features.npy": "(Tv, Dv) —— 逐帧视觉特征",
    "vision_timestamps.npy": "(Tv,) —— 每帧时间戳（秒）",
    "vision_valid.npy": "(Tv,) —— 源级有效标记，未检出人脸为 0",
    "vision_confidence.npy": "(Tv,) —— 检测置信度",
}

# What the current loader in q1_alignment.pipeline requires, in ONE npz per sample
CURRENT_LOADER = {
    "text_features": "(Tt, Dt) —— 必需",
    "audio_features": "(Ta, Da) —— 必需",
    "audio_times": "(Ta,) 或 (Ta,2) —— 必需，键名与文档不同",
    "vision_features": "(Tv, Dv) —— 必需",
    "vision_times": "(Tv,) 或 (Tv,2) —— 必需，键名与文档不同",
    "text_word_index": "(Tt,) —— 可选",
    "text_valid": "(Tt,) —— 可选",
    "audio_valid": "(Ta,) —— 可选",
    "vision_valid": "(Tv,) —— 可选",
}

GAPS = [
    ("文件组织", "每样本一个目录 + 9 个文件", "每样本一个 .npz 文件"),
    ("文件名", "samples/<sample_id>/ 下固定文件名", "<safe_id>.npz"),
    ("音频时间键名", "audio_intervals.npy", "audio_times"),
    ("视觉时间键名", "vision_timestamps.npy", "vision_times"),
    ("数组格式", ".npy 独立文件", "打包进 .npz 的数组"),
    ("样本 ID", "sample_id = video_id$_$clip_id（目录名）", "safe_id = video_id__clip_id（文件名）"),
    ("文字记录", "text_words.csv 存在", "无对应字段（但有 text_word_index 可选）"),
    ("置信度", "vision_confidence.npy 存在", "无对应字段"),
    ("元数据", "metadata.json 存在", "无对应字段"),
]


def main() -> None:
    print("=" * 78)
    print("协作技术文档交付格式  vs  当前 load_bundle 接口")
    print("=" * 78)
    print()
    print("【文档要求：每样本一个目录，9 个文件】")
    for name, desc in DOC_HANDOFF.items():
        print(f"  {name:24s} {desc}")
    print()
    print("【当前 load_bundle 要求：每样本一个 npz】")
    for name, desc in CURRENT_LOADER.items():
        print(f"  {name:24s} {desc}")
    print()
    print("=" * 78)
    print("不匹配项")
    print("=" * 78)
    for index, (aspect, doc, cur) in enumerate(GAPS, start=1):
        print(f"{index}. {aspect}")
        print(f"     文档交付 : {doc}")
        print(f"     当前期望 : {cur}")
    print()
    print("结论:")
    print("  队友严格按文档交付 -> 文件组织、键名、ID 命名均与当前 load_bundle 不同，")
    print("  直接调用会在第一步（找不到 <safe_id>.npz）就失败，")
    print("  因此需要一个适配层，而不是直接对接。")
    print()
    print("  另外，文档交付的字段比当前接口更丰富（text_words.csv /")
    print("  vision_confidence.npy / metadata.json），适配时应保留而不是丢弃。")


if __name__ == "__main__":
    main()
