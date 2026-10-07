# -*- coding: utf-8 -*-
"""汇总 work/_fast_base/*.json 的均分与四个分项（fast.py --save 存的 per-case 列表）。

只比对**两边都有的 case**，避免 n 不同造成假 delta。
用法: python work/tools/diag/_basemean.py [名字A 名字B]   # 不给名字就跑内置清单
"""
import json
import sys
from pathlib import Path

B = Path(r"Y:\workspace\EDA\work\_fast_base")
MET = ["component", "pin", "net_hypergraph", "net_line"]

PAIRS = [
    ("dropmixed_10gt", "dropmixed_vlm_10gt"),
    ("pinmerge_off", "pinmerge_on"),
    ("edgemerge19", "edgemerge_off19"),
    ("along0_19", "along1_19"),
    ("dropunread1_19", "dropmixed_19"),
    ("vdw0035", "vdw010"),
    ("steal0", "steal80"),
    ("vu0", "vu1"),
    ("paddef_19", "pad23y20_19"),
]


def get(n):
    p = B / (n + ".json")
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def by_case(rows):
    return {r["case"]: r for r in rows if isinstance(r, dict) and "score" in r}


def mean(rs, m):
    return sum((r.get(m) or {}).get("f1", 0.0) for r in rs) / len(rs)


def main():
    pairs = [(sys.argv[1], sys.argv[2])] if len(sys.argv) > 2 else PAIRS
    for a, b in pairs:
        ra, rb = get(a), get(b)
        if ra is None or rb is None:
            print("%s / %s : 缺" % (a, b))
            continue
        da, db = by_case(ra), by_case(rb)
        keys = sorted(set(da) & set(db), key=lambda x: (len(x), x))
        if not keys:
            print("%s / %s : 无公共 case" % (a, b))
            continue
        la = [da[k] for k in keys]
        lb = [db[k] for k in keys]
        sa = sum(r["score"] for r in la) / len(la)
        sb = sum(r["score"] for r in lb) / len(lb)
        print("%-38s n=%d(n=%d/%d)  %.4f -> %.4f   Δ %+.4f" % (a, len(keys), len(da), len(db), sa, sb, sb - sa))
        for m in MET:
            x, y = mean(la, m), mean(lb, m)
            print("      %-15s %.4f -> %.4f  (%+.4f)" % (m, x, y, y - x))
        nz = [(k, db[k]["score"] - da[k]["score"]) for k in keys if abs(db[k]["score"] - da[k]["score"]) > 1e-9]
        print("      逐 case：%s" % (", ".join("%s %+.2f" % t for t in nz[:14]) or "** 完全相同 **"))
        print()


if __name__ == "__main__":
    main()
