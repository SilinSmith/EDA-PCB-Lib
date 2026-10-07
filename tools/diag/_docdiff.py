# -*- coding: utf-8 -*-
"""对比两个融合产物目录里同一张图的交付文档，说清「到底改了什么」。

用法（workdir = Y:\\workspace\\EDA）：
    python work\\tools\\diag\\_docdiff.py <目录A> <目录B> <case> [--max 12]

只报**实质差异**：元件键的增删、元件 value/Name 变化、pins 键集合变化、
nets 的 hyperGraph 变化。用来判读 LLM 轮（judge/rescan/audit）在某张图上
到底动了什么 —— 尤其那些「一改就掉几十分」的 case。
"""
import argparse
import json
import os


def _load(d, case):
    with open(os.path.join(d, case + ".json"), encoding="utf-8") as f:
        return json.load(f)


def _hg(doc):
    out = {}
    for nk, nv in (doc.get("nets") or {}).items():
        if isinstance(nv, dict):
            out[nk] = nv.get("hyperGraph")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("case")
    ap.add_argument("--max", type=int, default=12)
    a = ap.parse_args()

    da, db = _load(a.a, a.case), _load(a.b, a.case)
    ca, cb = da.get("components") or {}, db.get("components") or {}
    print("== case %s" % a.case)
    print("   元件 %d -> %d   引脚组 %d -> %d   网络 %d -> %d" % (
        len(ca), len(cb), len(da.get("pins") or {}), len(db.get("pins") or {}),
        len(da.get("nets") or {}), len(db.get("nets") or {})))

    only_a = sorted(set(ca) - set(cb))
    only_b = sorted(set(cb) - set(ca))
    if only_a:
        print("   A 独有元件键 %d 个: %s" % (len(only_a), only_a[:a.max]))
    if only_b:
        print("   B 独有元件键 %d 个: %s" % (len(only_b), only_b[:a.max]))

    n = 0
    for k in sorted(set(ca) & set(cb)):
        va, vb = ca[k], cb[k]
        d = {}
        for f in ("value", "Name", "type"):
            if str(va.get(f)) != str(vb.get(f)):
                d[f] = (va.get(f), vb.get(f))
        pa = sorted((da.get("pins") or {}).get(k, {}).keys())
        pb = sorted((db.get("pins") or {}).get(k, {}).keys())
        if pa != pb:
            d["pins"] = ("%d 个" % len(pa), "%d 个" % len(pb))
        if d and n < a.max:
            print("   %-10s %s" % (k, json.dumps(d, ensure_ascii=False)))
            n += 1

    ha, hb = _hg(da), _hg(db)
    ndiff = 0
    for k in sorted(set(ha) | set(hb)):
        if ha.get(k) != hb.get(k):
            if ndiff < a.max:
                print("   net %-10s\n        A %s\n        B %s" % (k, ha.get(k), hb.get(k)))
            ndiff += 1
    print("   元件字段差异 %d 个（已列前 %d）｜网络差异 %d 条（已列前 %d）" % (
        sum(1 for k in set(ca) & set(cb)
            if any(str(ca[k].get(f)) != str(cb[k].get(f)) for f in ("value", "Name", "type"))),
        a.max, ndiff, a.max))


if __name__ == "__main__":
    main()
