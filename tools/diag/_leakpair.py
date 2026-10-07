"""对照 train 侧的 `synth_<id>` 与 val 侧的 `200_train_cases-<id>__full` 的标注直方图。

用途：owner 怀疑「训练集本身一开始就有几个问题，然后你又一直从里面切图出来组训练集」。
如果同一张源原理图在 train 与 val 里都带着**同一批错标**（例如 0050 的 28 个 `r`
落在电感式大锯齿上、而 `l` 计数为 0），那这条污染链就坐实了。
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import yaml

R = Path(r"Y:\workspace\EDA\trainset_v3")


def load_names() -> list:
    d = yaml.safe_load((R / "data.yaml").read_text(encoding="utf-8"))
    return list(d["names"])


def hist(split: str, stem: str, names: list):
    p = R / "labels" / split / (stem + ".txt")
    if not p.exists():
        return None, None
    c: Counter = Counter()
    rs: Counter = Counter()
    for ln in p.read_text(encoding="utf-8").splitlines():
        t = ln.split()
        if len(t) < 5:
            continue
        i = int(float(t[0]))
        k = names[i] if 0 <= i < len(names) else f"#{i}"
        c[k] += 1
        if k == "r":
            rs[(round(float(t[3])), round(float(t[4])))] += 1
    return c, rs


def show(split: str, stem: str, names: list) -> None:
    c, rs = hist(split, stem, names)
    if c is None:
        print(f"{split:6s} {stem:36s} MISSING")
        return
    n = sum(c.values())
    top = " ".join(f"{k}={v}" for k, v in c.most_common())
    print(f"{split:6s} {stem:36s} n={n:4d}  {top}")
    if rs:
        print(f"        r 尺寸 → {dict(rs)}")


def main() -> None:
    names = load_names()
    print(f"trainset_v3 类别数 = {len(names)}")
    pairs = [
        ("200_train_cases-0050__full", "synth_0050"),
        ("200_train_cases-0011__full", "synth_0011"),
        ("200_train_cases-0027__full", "synth_0027"),
        ("200_train_cases-0002__full", "synth_0002"),
        ("200_train_cases-0029__full", "synth_0029"),
        ("200_train_cases-0026__full", "synth_0026"),
        ("200_train_cases-0151__full", "synth_0151"),
    ]
    for stem_v, stem_t in pairs:
        print("-" * 100)
        show("val", stem_v, names)
        show("train", stem_t, names)
    # 非泄漏的 val 源，作对照
    print("-" * 100)
    for s in ("200_train_cases-0101__full", "200_train_cases-0127__full",
              "10_GTcase-0037-Datasheet__full"):
        show("val", s, names)
        show("train", "synth_" + s.split("-")[-1], names)


if __name__ == "__main__":
    main()
