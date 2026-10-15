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
    # 只跑子集的臂：通道均值与 Δ 必须在**所有臂都跑过的公共 case** 上算，否则
    # 把缺的 case 当 0 会读出 −200 这种假读数（2026-10-10 踩过）。
    common = [c for c in cases
              if all((d.get(c) or {}).get("score") is not None for _n, d in arms)]
    for ch in ("component", "pin", "net_hypergraph", "net_line"):
        line = []
        for _n, d in arms:
            vs = [(d[c].get(ch) or {}).get("f1") for c in common]
            vs = [v for v in vs if v is not None]
            line.append("%12.4f" % (sum(vs) / len(vs)) if vs else "%12s" % "-")
        print("%-7s" % ch + "".join(line))
    line, nline = [], []
    for _n, d in arms:
        vs = [e.get("score") for e in d.values() if e.get("score") is not None]
        line.append("%12.2f" % (sum(vs) / len(vs)) if vs else "%12s" % "-")
        nline.append("%12d" % len(vs))
    print("%-7s" % "TOTAL" + "".join(line))
    print("%-7s" % "n" + "".join(nline))
    cline = []
    for _n, d in arms:
        vs = [d[c].get("score") for c in common]
        cline.append("%12.2f" % (sum(vs) / len(vs)) if vs else "%12s" % "-")
    print("%-7s" % ("COMMON%d" % len(common)) + "".join(cline))
    # 逐案单调性（相对第一臂；只在两者都跑过的 case 上比）
    print("\n相对 %s 的逐案 Δ（只算两臂都跑过的 case）：" % base_name)
    for n, d in arms[1:]:
        up = dn = eq = nc = 0
        tot = 0.0
        for c in cases:
            s0 = (base.get(c) or {}).get("score")
            s1 = (d.get(c) or {}).get("score")
            if s0 is None or s1 is None:
                continue
            nc += 1
            tot += s1 - s0
            if s1 > s0 + 1e-9:
                up += 1
            elif s1 < s0 - 1e-9:
                dn += 1
            else:
                eq += 1
        print("  %-12s sum %+8.2f (n=%d)  mean %+6.2f  涨 %d / 平 %d / 跌 %d"
              % (n, tot, nc, (tot / nc if nc else 0.0), up, eq, dn))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
