# -*- coding: utf-8 -*-
"""把官方两份元件库合成本地开发用的第三份（**只做测量，不改交付**）。

背景（用户 m42932「以新的这个为参考」）：`赛题资料\\BoxICLibrary_200cases.json` 与
`work\\data\\BoxICLibrary_200cases.json` 字节相同（MD5 A163E4B1…，360 键），它的
IC 名命中率在 200 训练案例上是 61.6%（cpntLibrary 337 键只有 30.9%），引脚名覆盖
69.7% vs 64.9%。但 cpntLibrary 里有 BoxIC 没有的键（元件库更全）。
⇒ 合成规则：**BoxIC 优先（同名键覆盖），cpntLibrary 补缺键**。

输出 `work\\data\\_lib_merge.json`（不进仓库；若最终采用，交付侧由 `libpins.py`
在加载时合并，仍然只随包走两份**官方原件**）。
"""
from __future__ import annotations

import io
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[3]      # Y:\workspace\EDA
CPNT = ROOT / "eda-pcb" / "data" / "cpntLibrary.json"
BOXIC = ROOT / "work" / "data" / "BoxICLibrary_200cases.json"
OUT = ROOT / "work" / "data" / "_lib_merge.json"


def load(p: pathlib.Path) -> dict:
    return json.load(io.open(str(p), encoding="utf-8"))


def main() -> None:
    a, b = load(CPNT), load(BOXIC)
    merged: dict = {}
    for k, v in a.items():
        merged[k] = v
    n_over = 0
    for k, v in b.items():
        if k in merged:
            n_over += 1
        merged[k] = v
    OUT.write_text(json.dumps(merged, ensure_ascii=False), encoding="utf-8")
    only_a = [k for k in a if k not in b]
    only_b = [k for k in b if k not in a]
    print("cpnt %d 键 / boxic %d 键 / 同名 %d / 合成 %d 键"
          % (len(a), len(b), n_over, len(merged)))
    print("只 cpnt 有 %d 键，例：%s" % (len(only_a), ", ".join(only_a[:12])))
    print("只 boxic 有 %d 键，例：%s" % (len(only_b), ", ".join(only_b[:12])))
    print("写出 %s（%.1f KB）" % (OUT, OUT.stat().st_size / 1024.0))


if __name__ == "__main__":
    main()
