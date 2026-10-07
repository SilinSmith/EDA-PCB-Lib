# -*- coding: utf-8 -*-
"""字符表：CTC 的标签空间。

约定：**索引 0 永远是 CTC blank**，真实字符从 1 开始编号。
字符表从语料统计里生成，并按最小频次过滤——语料里出现 1~2 次的字符
（比如 `m3螺丝` 里的汉字）学不出来，留着只会让 CTC 分心，所以默认丢掉，
但要记在 meta 里以便追溯。
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

BLANK = 0


class Charset:
    def __init__(self, chars: Sequence[str], meta: "dict | None" = None):
        self.chars: List[str] = list(chars)
        self.stoi: Dict[str, int] = {c: i + 1 for i, c in enumerate(self.chars)}
        self.itos: Dict[int, str] = {i + 1: c for i, c in enumerate(self.chars)}
        self.meta: dict = meta or {}

    # ---------------- 基本信息 ----------------
    @property
    def num_classes(self) -> int:
        return len(self.chars) + 1

    def __len__(self) -> int:
        return self.num_classes

    def __contains__(self, ch: str) -> bool:
        return ch in self.stoi

    # ---------------- 编解码 ----------------
    def encode(self, text: str) -> List[int]:
        """未知字符会抛 KeyError——调用方要用 try_encode。"""
        return [self.stoi[c] for c in text]

    def try_encode(self, text: str) -> "List[int] | None":
        """任何一个字符不在表里就返回 None（整个样本丢弃）。"""
        out = []
        for c in text:
            i = self.stoi.get(c)
            if i is None:
                return None
            out.append(i)
        return out

    def decode(self, ids: Iterable[int]) -> str:
        return "".join(self.itos.get(int(i), "") for i in ids if int(i) != BLANK)

    @staticmethod
    def ctc_collapse(ids: Iterable[int]) -> List[int]:
        """CTC 贪心解码：先并掉连续重复，再删 blank。"""
        out: List[int] = []
        prev = None
        for i in ids:
            i = int(i)
            if i != prev and i != BLANK:
                out.append(i)
            prev = i
        return out

    def greedy(self, ids: Iterable[int]) -> str:
        return self.decode(self.ctc_collapse(ids))

    # ---------------- 持久化 ----------------
    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"chars": self.chars, "meta": self.meta},
                       ensure_ascii=False, indent=1),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path) -> "Charset":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(d["chars"], d.get("meta"))

    # ---------------- 构造 ----------------
    @classmethod
    def from_corpus(cls, counter: Counter, min_freq: int = 20,
                    extra: str = "") -> "Charset":
        kept: List[str] = []
        dropped: Dict[str, int] = {}
        for ch, n in counter.items():
            if n >= min_freq:
                kept.append(ch)
            else:
                dropped[ch] = n
        for ch in extra:
            if ch not in kept:
                kept.append(ch)
        kept.sort()
        meta = {
            "min_freq": min_freq,
            "n_distinct_seen": len(counter),
            "dropped": dict(sorted(dropped.items(), key=lambda kv: -kv[1])),
        }
        return cls(kept, meta)

    def describe(self) -> str:
        return ("%d chars + blank = %d classes\n  chars: %s"
                % (len(self.chars), self.num_classes, "".join(self.chars)))
