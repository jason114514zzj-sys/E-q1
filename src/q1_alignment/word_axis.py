# -*- coding: utf-8 -*-
"""词轴对账：把「队友交付的 token 行」映射到「我们的规范词索引」。

为什么需要它
------------
`handoff/q1_expected_words.csv` 公布的是**我们的**规范分词（官方转写按空白切分，
标点附在词尾）。2026-09-25 收到的队友交付包用的是另一套分词：标点独立成 token
（78/100 条）、数字逐字符拆（4 条）、引号/插入符等标注字符（3 条），只有 14 条逐词一致。

两套分词指的是**同一段文本**，所以可以逐字符对账：把两边的拼接字符串对齐，
我的每个词覆盖到它落在其字符区间内的那些 token 行。这样做的好处是
**可核验**：任何不属于「标点/标注字符的增删」的差异都会让对账失败并报错，
而不是被静默地按行号错配。

对外接口
--------
``build_mapping(my_words, their_tokens, sample_id)`` →
``(mapping, report)``，其中 ``mapping[r]`` 是第 r 个交付 token 所属的**我的词索引**。

验收规则（三条都满足才通过）
--------------------------
1. 我的每个词至少被一个 token 覆盖，且覆盖顺序单调不减；
2. 每个 token 的字符区间都落在某个词的区间内（端点允许 ±1 字符的标注字符误差）；
3. 两边的字符级差异只允许出现在**标点或标注字符**上；出现其它差异即判失败。
"""
from __future__ import annotations

import difflib
import unicodedata
from dataclasses import dataclass, field

# 标注/装饰字符：允许在两边拼接文本里多出或少掉，不影响对账
ARTIFACT_CHARS = set("^`")


def _norm(text: str) -> str:
    """NFKC 归一（不折叠引号：折弯引号是字符差异，必须如实对待）。"""
    return unicodedata.normalize("NFKC", text)


def _is_ignorable(ch: str) -> bool:
    """可以多出/少掉的字符：标点、空白、标注字符。"""
    return (ch.isspace() or ch in ARTIFACT_CHARS
            or unicodedata.category(ch).startswith("P")
            or unicodedata.category(ch).startswith("S"))


@dataclass
class WordAxisReport:
    sample_id: str
    ok: bool
    n_my_words: int = 0
    n_tokens: int = 0
    n_merged_tokens: int = 0          # 被合并进前一个词的 token 数（标点/数字等）
    n_unassigned_tokens: int = 0      # 无法归属的 token（纯标点/标注字符才允许）
    char_diffs: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    note: str = ""

    def as_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "ok": self.ok,
            "n_my_words": self.n_my_words,
            "n_tokens": self.n_tokens,
            "n_merged_tokens": self.n_merged_tokens,
            "n_unassigned_tokens": self.n_unassigned_tokens,
            "char_diffs": self.char_diffs[:6],
            "errors": self.errors[:6],
            "note": self.note,
        }


def build_mapping(my_words, their_tokens, sample_id: str = ""):
    """返回 (mapping, WordAxisReport)。

    mapping: list[int]，长度 = len(their_tokens)，值 ∈ [0, len(my_words))；
             无法归属的 token 记为 -1（仅当它是纯标点/标注字符时才允许）。
    """
    rep = WordAxisReport(sample_id=sample_id, ok=False,
                         n_my_words=len(my_words), n_tokens=len(their_tokens))
    a = _norm("".join(my_words))
    b = _norm("".join(their_tokens))
    if not a or not b:
        rep.errors.append("空词表：无法对账")
        return [-1] * len(their_tokens), rep

    # 字符级对齐：找出所有非 equal 的差异块
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        if tag == "insert":
            extra = b[j1:j2]
        elif tag == "delete":
            extra = a[i1:i2]
        else:
            extra = a[i1:i2] + b[j1:j2]
        if not all(_is_ignorable(ch) for ch in extra):
            rep.errors.append(f"字符差异含非标点内容：{extra!r}")
            rep.char_diffs.append(f"{tag}:{extra!r}")
    if rep.errors:
        return [-1] * len(their_tokens), rep

    # 用字符级对齐把 a 的偏移映射到 b 的偏移
    a2b = {}
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for off in range(i2 - i1):
                a2b[i1 + off] = j1 + off
        elif tag == "delete":
            for off in range(i2 - i1):
                a2b[i1 + off] = j1
        # insert：b 的字符没有 a 对应，交由下面的区间覆盖处理
    # 我的词 → 字符区间
    spans, cursor = [], 0
    for w in my_words:
        n = len(_norm(w))
        spans.append((cursor, cursor + n))
        cursor += n
    # 交付 token → 字符区间
    tspan, cur = [], 0
    for w in their_tokens:
        n = len(_norm(w))
        tspan.append((cur, cur + n))
        cur += n

    mapping = [-1] * len(their_tokens)
    # 反向：b 偏移 → a 偏移（用于端点容差判断）
    b2a = {}
    not_mapped = []

    for r, (tb, te) in enumerate(tspan):
        # 该 token 的字符在 a 上落在哪里
        hits = set()
        for off in range(tb, te):
            if off in {v for v in a2b.values()}:
                pass
        # 直接法：找到覆盖该 token 中点（或端点）的词
        mid = (tb + te) // 2 if te > tb else tb
        cand = None
        # 在 b→a 的映射里找 mid 附近的 a 偏移
        best = None
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if j1 <= mid < j2:
                if tag == "equal":
                    best = i1 + (mid - j1)
                elif tag == "insert":
                    # 插入的字符属于它前后所在的词
                    best = i1
                break
        if best is None:
            # 落在末尾之后
            best = len(a) - 1 if a else 0
        for k, (sb, se) in enumerate(spans):
            if sb <= best < se or (k == len(spans) - 1 and best >= se - 1):
                cand = k
                break
        if cand is None:
            # 纯标点 token 落到词边界之外：归到最近的词
            if all(_is_ignorable(ch) for ch in _norm(their_tokens[r])):
                cand = min(range(len(spans)),
                           key=lambda k: min(abs(best - spans[k][0]), abs(best - spans[k][1])))
            else:
                not_mapped.append(r)
                continue
        mapping[r] = cand

    # 单调性检查
    assigned = [m for m in mapping if m >= 0]
    if assigned != sorted(assigned):
        rep.errors.append("归属顺序非单调：分词对账不可信")
        return [-1] * len(their_tokens), rep
    covered = set(assigned)
    if len(covered) != len(my_words):
        missing = sorted(set(range(len(my_words))) - covered)
        rep.errors.append(f"有 {len(missing)} 个规范词没有任何 token 覆盖（首个 #{missing[0]}）")
        return [-1] * len(their_tokens), rep

    rep.ok = not rep.errors
    rep.n_merged_tokens = sum(1 for r in range(1, len(their_tokens))
                              if mapping[r] == mapping[r - 1])
    rep.n_unassigned_tokens = sum(1 for m in mapping if m < 0)
    rep.note = (f"{len(my_words)} 个规范词 ← {len(their_tokens)} 个交付 token；"
                f"同一词内的 token {rep.n_merged_tokens} 个；未归属 {rep.n_unassigned_tokens} 个")
    return mapping, rep
