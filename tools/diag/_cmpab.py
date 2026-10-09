# -*- coding: utf-8 -*-
"""多臂基线对比：`work\\_fast_base\\<arm>.json`（`fast.py --save <arm>` 的产物）。

用法（在仓库根 `Y:\\workspace\\EDA` 下跑）：
    python -X utf8 eda-pcb\\tools\\diag\\_cmpab.py base=work\\_fast_base\\dlv10.json a=work\\_fast_base\\c4.json

打印：每张图的总分矩阵 + 各通道均值 + TOTAL + 「逐案单调」计数
（相对第一臂有多少张图涨/平/跌）。判据用「逐案单调性优先于均值」：
均值涨但逐案有反复的臂要先看机制，别急着翻默认（见 `docs\\RESULTS.md` §2.16）。

配套装置：`work\\_pinab*.ps1`（交付口径 × 官方 10 张的串行 A/B 脚本模板）。
"""
from __future__ import annotations

import io
import json
import sys

ROOT = None  # 只吃命令行给的路径，不依赖仓库位置


def load(p):
    with io.open(p, encoding="utf-8") as fp:
        return {str(e["case"]): e for e in json.load(fp)}


def main(argv):
    arms = []
    for a in argv:
        name, _, path = a.partition("=")
        arms.append((name, load(path)))
    if not arms:
        print("用法：_cmpab.py base=<a.json> arm=<b.json> …")
        return 1
    cases = sorted(set().union(*[set(d) for _n, d in arms]), key=lambda s: (len(s), s))
    base_name, base = arms[0]
    print("case   " + "".join("%12s" % n for n, _ in arms))
    for c in cases:
        row = []
        for _n, d in arms:
            s = (d.get(c) or {}).get("score")
            row.append("%12.2f" % s if s is not None else "%12s" % "-")
        print("%-7s" % c + "".join(row))
    print()
    for ch in ("component", "pin", "net_hypergraph", "net_line"):
        line = []
        for _n, d in arms:
            vs = [(e.get(ch) or {}).get("f1") for e in d.values()]
            vs = [v for v in vs if v is not None]
            line.append("%12.4f" % (sum(vs) / len(vs)) if vs else "%12s" % "-")
        print("%-7s" % ch + "".join(line))
    line = []
    for _n, d in arms:
        vs = [e.get("score") for e in d.values() if e.get("score") is not None]
        line.append("%12.2f" % (sum(vs) / len(vs)))
    print("%-7s" % "TOTAL" + "".join(line))
    # 逐案单调性（相对第一臂）
    print("\n相对 %s 的逐案 Δ（涨/平/跌）：" % base_name)
    for n, d in arms[1:]:
        up = dn = eq = 0
        for c in cases:
            s0 = (base.get(c) or {}).get("score")
            s1 = (d.get(c) or {}).get("score")
            if s0 is None or s1 is None:
                continue
            if s1 > s0 + 1e-9:
                up += 1
            elif s1 < s0 - 1e-9:
                dn += 1
            else:
                eq += 1
        tot0 = sum((base.get(c) or {}).get("score") or 0.0 for c in cases)
        tot1 = sum((d.get(c) or {}).get("score") or 0.0 for c in cases)
        print("  %-12s Δ总分 %+7.2f  涨 %d / 平 %d / 跌 %d" % (n, tot1 - tot0, up, eq, dn))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
