# -*- coding: utf-8 -*-
"""split 审计：val 的 77 张到底是不是 77 张**独立**的图？train/val 有没有同源泄漏？

背景：`trainset_v3` 里同一张原图既有 `__full` 又有 `__t00_00`... 这样的切片，
且实测 `0027__full` 与 `0027__t00_00` 的类别直方图**逐项相同**（像是同一份内容）。
而之前的结论「77 张 val 里约 5 张贡献了 44%~94% 的漏检」如果成立，很可能是因为
val 里真正独立的原图本来就没几张 —— 那就不是「模型处理不了少数风格」，
而是「val 本身是少数几张图的重复」。

用法: python work\\tools\\diag\\_splitaudit.py [--root ...]
"""
import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path

import yaml

EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def source_id(stem: str) -> str:
    """把 `200_train_cases-0050__t02_01` / `synth_0050` / `10_GTcase-0070-Datasheet__full`
    都归到同一个源 id（取 `__` 前那段的**最后一串数字**）。"""
    s = stem.split("__")[0]
    m = re.findall(r"\d+", s)
    return m[-1] if m else s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"Y:\workspace\EDA\trainset_v3")
    a = ap.parse_args()
    root = Path(a.root)
    names = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))["names"]

    srcs = {}
    for split in ("train", "val", "test"):
        idir = root / "images" / split
        if not idir.exists():
            continue
        stems = sorted(p.stem for p in idir.iterdir()
                       if p.suffix.lower() in EXTS and p.is_file())
        by = defaultdict(list)
        for s in stems:
            by[source_id(s)].append(s)
        srcs[split] = by
        n_multi = sum(1 for v in by.values() if len(v) > 1)
        print("[%s] 文件 %d 个 -> 不同源 id %d 个 ｜ 有多个文件的源 %d 个"
              % (split, len(stems), len(by), n_multi))
        top = sorted(by.items(), key=lambda kv: -len(kv[1]))[:6]
        for sid, v in top:
            print("     src %s : %d 个文件  %s" % (sid, len(v), v[:4]))
        # 精确重复：文件内容相同的（用大小+名字长度粗筛后按字节比）
        sig = Counter()
        for s in stems:
            for ext in EXTS:
                p = idir / (s + ext)
                if p.exists():
                    sig[(p.stat().st_size, s.split("__")[0])] += 1
                    break
        dup = [(k, v) for k, v in sig.items() if v > 1]
        print("     同名源下大小完全相同的文件组 %d 个（可疑重复）" % len(dup))

    if "train" in srcs and "val" in srcs:
        tr, va = set(srcs["train"]), set(srcs["val"])
        both = sorted(tr & va)
        print("\n*** train/val 同源 id 交集：%d 个 -> %s" % (len(both), both[:40]))
        # 精确同名（去掉 __ 后完全一样的 stem）
        ntr = {s.split("__")[0] for s in srcs["train"].values() for s in s}
        nva = {s.split("__")[0] for s in srcs["val"].values() for s in s}
        print("*** 去 __ 后完全同名的 stem 交集：%d 个 -> %s"
              % (len(ntr & nva), sorted(ntr & nva)[:20]))

    # val 内部的重复内容（同一源被切成多块）
    if "val" in srcs:
        v = srcs["val"]
        print("\nval 里被切成 >1 块的源：")
        for sid, lst in sorted(v.items(), key=lambda kv: -len(kv[1])):
            if len(lst) > 1:
                print("     %s -> %d 块" % (sid, len(lst)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
