# -*- coding: utf-8 -*-
"""全量 220 张的**阶段漏斗**分析 —— 不跑管线，只读已经打好的分。

一份 `work/_fast_base/_220_*.json`（形状：{config, summary, cases:[{case_id,
score, component_f1, pin_f1, net_hypergraph_f1, net_line_f1, counts, diagnostic}]}）
就是一个阶段。本脚本把 N 份这样的文件叠起来看**每一段加了多少**、
**分丢在哪一类**、**哪几张最差**。

用法：
    python work\\tools\\diag\\_stages.py _220_stage2 _220_stage3 _220_stage3_vlm
    python work\\tools\\diag\\_stages.py _220_stage3_vlm --worst 25
    python work\\tools\\diag\\_stages.py _220_stage3_vlm --diag 30
    python work\\tools\\diag\\_stages.py _220_stage3_vlm --split 10GT
    python work\\tools\\diag\\_stages.py --list
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

BASE = Path(r"Y:\workspace\EDA\work\_fast_base")
METRICS = ["component", "pin", "net_hypergraph", "net_line"]
ZH = {"component": "元件", "pin": "引脚", "net_hypergraph": "网络", "net_line": "导线"}

# 10GT 官方评测集（fast.py 里那张表用的 id 形态）
GT10 = {"1", "2", "3", "4", "5", "7", "8", "9", "10"}


def load(name):
    p = BASE / (name + ".json")
    d = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(d, list):                       # base_full 那种「行列表」
        return d
    return d.get("cases", [])


def split_of(cid):
    if cid in GT10:
        return "10GT"
    return "200"


def get_f1(c, m):
    """两种形状都吃：{component_f1} 或 {component:{f1}}。"""
    if ("%s_f1" % m) in c:
        return float(c["%s_f1" % m] or 0.0)
    v = c.get(m)
    if isinstance(v, dict):
        return float(v.get("f1") or 0.0)
    return 0.0


def get_tp(c, m, key):
    v = (c.get("counts") or {}).get(m)
    if v is None:
        v = c.get(m)
    if isinstance(v, dict):
        return int(v.get(key) or 0)
    return 0


def get_score(c):
    if "score" in c:
        return float(c["score"] or 0.0)
    return 0.0


def summarize(rows, title):
    n = len(rows)
    if not n:
        print("== %s : 空" % title)
        return
    print("== %-22s %3d 张  总均分 %6.3f" % (title, n, sum(get_score(r) for r in rows) / n))
    for m in METRICS:
        f1 = sum(get_f1(r, m) for r in rows) / n
        tp = sum(get_tp(r, m, "tp") for r in rows)
        fp = sum(get_tp(r, m, "fp") for r in rows)
        fn = sum(get_tp(r, m, "fn") for r in rows)
        print("     %-4s f1=%.4f  tp=%-6d fp=%-6d fn=%-6d  (tp+fp=%d)" % (
            ZH[m], f1, tp, fp, fn, tp + fp))
    z = {m: [r.get("case_id") or r.get("case") for r in rows if get_f1(r, m) <= 0.0] for m in METRICS}
    for m in METRICS:
        if z[m]:
            print("     爆0 %-4s %2d 张：%s" % (ZH[m], len(z[m]), " ".join(map(str, z[m][:30]))))
    return z


def diff(a, b, na, nb):
    """b 相对 a 的增量，按分项拆开。"""
    ka = {str(r.get("case_id") or r.get("case")): r for r in a}
    kb = {str(r.get("case_id") or r.get("case")): r for r in b}
    common = [k for k in kb if k in ka]
    print("== 增量 %s → %s （%d 张共同 case）" % (na, nb, len(common)))
    dtot = sum(get_score(kb[k]) - get_score(ka[k]) for k in common) / max(1, len(common))
    print("     总均分 %+7.3f" % dtot)
    for m in METRICS:
        d = sum(get_f1(kb[k], m) - get_f1(ka[k], m) for k in common) / max(1, len(common))
        dtp = sum(get_tp(kb[k], m, "tp") - get_tp(ka[k], m, "tp") for k in common)
        dfp = sum(get_tp(kb[k], m, "fp") - get_tp(ka[k], m, "fp") for k in common)
        dfn = sum(get_tp(kb[k], m, "fn") - get_tp(ka[k], m, "fn") for k in common)
        print("     %-4s Δf1=%+.4f  Δtp=%+6d Δfp=%+6d Δfn=%+6d" % (ZH[m], d, dtp, dfp, dfn))


def agg_diag(rows, top):
    """把 diagnostic 字典按「总量」和「每张平均」排出来 —— 这是丢分归因。"""
    tot = Counter()
    cnt = Counter()
    for r in rows:
        d = r.get("diagnostic") or {}
        for k, v in d.items():
            if isinstance(v, (int, float)):
                tot[k] += v
                cnt[k] += 1
    if not tot:
        print("== 没有 diagnostic 字段")
        return
    print("== diagnostic 总量 Top %d（%d 张）" % (top, len(rows)))
    for k, v in tot.most_common(top):
        print("   %-36s 合计 %-9.1f  出现 %3d 张  均 %.2f" % (k, v, cnt[k], v / max(1, cnt[k])))


def group_diag(rows, top):
    """把 `a.b` 前缀聚成组，看哪一大类是主要漏失。"""
    g = Counter()
    for r in rows:
        for k, v in (r.get("diagnostic") or {}).items():
            if isinstance(v, (int, float)) and "." in k:
                g[k.split(".")[0]] += v
    print("== diagnostic 按前缀分组")
    for k, v in g.most_common(top):
        print("   %-24s %9.1f" % (k, v))


def worst(rows, top, key=""):
    s = sorted(rows, key=lambda r: get_f1(r, key) if key else get_score(r))
    print("== 最差 %d 张（按 %s）" % (top, ZH.get(key, "总分")))
    print("   %-8s %-5s %7s  %8s %8s %8s %8s" % ("case", "split", "总分",
                                                 ZH["component"], ZH["pin"],
                                                 ZH["net_hypergraph"], ZH["net_line"]))
    for r in s[:top]:
        cid = str(r.get("case_id") or r.get("case"))
        print("   %-8s %-5s %7.2f  %8.4f %8.4f %8.4f %8.4f" % (
            cid, split_of(cid), get_score(r), get_f1(r, "component"), get_f1(r, "pin"),
            get_f1(r, "net_hypergraph"), get_f1(r, "net_line")))


def main(argv=None):
    ap = argparse.ArgumentParser(description="全量 220 张阶段漏斗")
    ap.add_argument("files", nargs="*", help="_fast_base 下的名字（不带 .json）")
    ap.add_argument("--worst", type=int, default=0, help="打印最差 N 张")
    ap.add_argument("--worst-key", default="", choices=[""] + METRICS)
    ap.add_argument("--diag", type=int, default=0, help="打印 diagnostic Top N")
    ap.add_argument("--split", default="", help="只看某个 split（10GT / 200）")
    ap.add_argument("--list", action="store_true", help="列出 _fast_base 下所有 json")
    ap.add_argument("--diff", nargs=2, default=None, help="两两比较（名字对）")
    a = ap.parse_args(argv)

    if a.list:
        for p in sorted(BASE.glob("*.json")):
            print(p.stem)
        return 0

    if a.diff:
        x, y = a.diff
        diff(load(x), load(y), x, y)
        return 0

    if not a.files:
        a.files = ["_220_stage2", "_220_stage3", "_220_stage3_vlm"]

    allrows = {}
    for name in a.files:
        rows = load(name)
        if a.split:
            rows = [r for r in rows if split_of(str(r.get("case_id") or r.get("case"))) == a.split]
        allrows[name] = rows
        summarize(rows, name)

    if len(a.files) >= 2:
        print()
        b = allrows[a.files[-1]]
        for name in a.files[:-1]:
            diff(allrows[name], b, name, a.files[-1])
            print()

    last = allrows[a.files[-1]]
    if a.diag:
        print()
        agg_diag(last, a.diag)
        print()
        group_diag(last, 12)
    if a.worst:
        print()
        worst(last, a.worst, a.worst_key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
