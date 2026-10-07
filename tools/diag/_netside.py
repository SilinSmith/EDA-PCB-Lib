# -*- coding: utf-8 -*-
"""把一张图上「GT 的网络」与「我们的网络」并排打出来，用来定位拓扑失分。

用官方指标自己的 `assign_components` + `rewrite_keys` 做键改写，再逐 GT 网列出
「我们把它分到了哪几个网上」，并标出 `∅`（号随机、不可赢）成员。

用法：
    python work\\tools\\diag\\_netside.py --dir work\\_fast_out_wtol_def --case 10
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DSROOT = os.path.join(ROOT, "eda-pcb", "yolo_part", "Dataset(waitforpreprocess)")
sys.path.insert(0, os.path.join(ROOT, "eda-pcb"))

from schematic_parser import data as D          # noqa: E402
from schematic_parser import metrics as M       # noqa: E402


def pred_path(d, case):
    for cand in (os.path.join(d, "final_results", case + "_Pred.json"),
                 os.path.join(d, case + ".json"),
                 os.path.join(d, "runs", "fusion", case + ".json")):
        if os.path.exists(cand):
            return cand
    return None


def ours(d, case):
    p = pred_path(d, case)
    return json.load(open(p, encoding="utf-8")) if p else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(ROOT, "work", "_fast_out_wtol_def"))
    ap.add_argument("--case", default="10")
    a = ap.parse_args()

    gt = None
    for p in Path(DSROOT).rglob(a.case + ".json"):
        if p.parent.name == a.case:
            gt = json.load(open(p, encoding="utf-8"))
            break
    if gt is None:
        print("找不到 GT", a.case)
        return 1
    pd = ours(a.dir, a.case)
    if pd is None:
        print("找不到产物", a.case)
        return 1
    cfg = M.EvalConfig()
    _comp, keymap = M.assign_components(gt, pd, cfg, None)
    p2 = M.rewrite_keys(pd, keymap)

    p_net_of = {}
    for nid, n in (p2.get("nets") or {}).items():
        for key, pn in D.net_pin_set(n):
            p_net_of[(key, str(pn))] = nid

    print("GT 网 → 我们把它拆成了哪些网（本工具用度量自己的 keymap 改写键）")
    for gid, g in (gt.get("nets") or {}).items():
        mem = sorted(D.net_pin_set(g))
        if not mem:
            continue
        mine = {}
        miss = []
        for key, pn in mem:
            nid = p_net_of.get((key, str(pn)))
            if nid is None:
                miss.append((key, pn))
            else:
                mine.setdefault(nid, []).append("%s.%s" % (key, pn))
        tag = " ".join("%s(%d)" % (k, len(v)) for k, v in sorted(mine.items(), key=lambda kv: -len(kv[1])))
        print("\n%-7s GT %2d 脚 | 我们 %d 个网: %s" % (gid, len(mem), len(mine), tag))
        for k, v in sorted(mine.items(), key=lambda kv: -len(kv[1])):
            print("            %-8s %s" % (k, " ".join(v[:26])))
        if miss:
            print("            没发出: %s" % " ".join("%s.%s%s" % (k, n, "(∅)" if M.is_unkeyed(k) else "")
                                                    for k, n in miss[:26]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
