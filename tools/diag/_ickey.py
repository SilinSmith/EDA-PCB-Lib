# -*- coding: utf-8 -*-
"""IC 的**钥匙**为什么配不上 —— 逐条拆开。

`_pinjunk.py`/`_pinprune.py` 已经把账算明白：快速回归集上 GT 的 IC 引脚有 587 个，
我们只产出 225 个，`fn = 538` 压倒一切；而它们的根因是**所属 IC 的键没配上**
（官方 `pin_f1` 的 `box_comps` 按 GT 的键走）。所以这一轮只看一件事：

    对一个「GT 是 IC、有真名、但键没配上」的元件，我们的预测里到底有没有对应的框？
    如果有，它被赋了什么键、是什么 type、
    `eda-pcb\\vlm\\enrich.py` 的位号通道**会不会去读它**？

分类（互斥，从上往下第一个命中）：
  `N`  **整颗没输出** —— 预测里没有任何 bbox 与它 IoU ≥ `--iou` 的元件。
  `T`  **类型不对** —— 有框，但 `type` 不在 `IC = ("box","ic","boxic")` 里，且键不是 `pNNN`
       ⇒ `enrich` 的 `want = ct in IC or (desig_fallback and pNNN)` 为假，**位号通道根本不会碰它**。
  `P`  **键是兜底键** —— `pNNN`，`enrich` 会读；读出来就能改名。
  `S`  **键是真名但读错** —— 有 IC 类型的框、键形如 `U5`/`J2`，`enrich` 会读。

用法：
    python work\\tools\\diag\\_ickey.py [--pred 目录] [--cases 1,2,3] [--iou 0.35] [--show]
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))
sys.path.insert(0, str(ROOT / "work" / "tools" / "eval"))

from stage2_eval import DEFAULT_DATA_ROOT, build_index, load_json, resolve  # noqa: E402

FAST = ["1", "2", "3", "4", "5", "7", "8", "9", "10",
        "0092", "0048", "0019", "0024", "0075", "0086", "0002", "0001", "0042", "0103"]
IC = ("box", "ic", "boxic")
UKEY = re.compile(r"^\u2205\d+$")
FKEY = re.compile(r"^p\d{3,4}$")
DESIG = re.compile(r"^[A-Za-z]{1,3}\d{1,4}$")


def pred_file(d: Path, cid: str) -> Path:
    for name in ("%s_Pred.json" % cid, "%s.json" % cid):
        p = d / name
        if p.exists():
            return p
    raise FileNotFoundError(cid)


def iou(a, b) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=str(ROOT / "work" / "_fast_out" / "final_results"))
    ap.add_argument("--cases", default="")
    ap.add_argument("--iou", type=float, default=0.35)
    ap.add_argument("--show", type=int, default=0)
    a = ap.parse_args()

    pred_dir = Path(a.pred)
    cases = [x.strip() for x in a.cases.split(",") if x.strip()] or FAST
    exact, digits = build_index(DEFAULT_DATA_ROOT)

    tot = collections.Counter()
    detail = []
    for cid in cases:
        c, _how = resolve(cid, exact, digits)
        gt = load_json(c.json_path)
        pr = json.loads(pred_file(pred_dir, cid).read_text(encoding="utf-8"))
        gcomps = gt.get("components") or {}
        pcomps = pr.get("components") or {}
        pkeys = set(pcomps)
        gkeys = set(gcomps)
        for gk, gv in gcomps.items():
            if str(gv.get("type") or "").lower() not in IC:
                continue
            if UKEY.match(gk) or not DESIG.match(gk):
                continue                      # GT 自己就是无名键 / 不是位号
            tot["gt_ic"] += 1
            if gk in pkeys:
                tot["matched"] += 1
                continue
            tot["unmatched"] += 1
            # 找重叠的预测框
            best, bk = 0.0, None
            for pk, pv in pcomps.items():
                bb = pv.get("bbox")
                if not bb:
                    continue
                v = iou(list(gv["bbox"]), list(bb))
                if v > best:
                    best, bk = v, pk
            if bk is None or best < a.iou:
                tot["N_整颗没输出"] += 1
                detail.append((cid, gk, "N", "-", "-", round(best, 2)))
                continue
            pt = str(pcomps[bk].get("type") or "").lower()
            if pt in IC:
                kind = "P_兜底键" if FKEY.match(bk) else "S_读到别的位号"
            elif FKEY.match(bk):
                kind = "P_兜底键"
            else:
                kind = "T_类型不是IC"
            tot[kind] += 1
            detail.append((cid, gk, kind[0], bk, pt, round(best, 2)))

    n = max(1, tot["gt_ic"])
    print("GT 的 IC 元件（有真名）：%d" % tot["gt_ic"])
    print("  键配上  %4d  (%.3f)" % (tot["matched"], tot["matched"] / n))
    print("  键没配上 %4d  (%.3f)  <- 下面按原因拆" % (tot["unmatched"], tot["unmatched"] / n))
    print()
    un = max(1, tot["unmatched"])
    for k in ("N_整颗没输出", "T_类型不是IC", "P_兜底键", "S_读到别的位号"):
        print("  %-14s %4d  (%.3f)" % (k, tot[k], tot[k] / un))
    print()
    if a.show:
        print("%-8s %-8s %-3s %-10s %-8s %s" % ("case", "GT键", "类", "我们的键", "type", "IoU"))
        shown = 0
        for row in detail:
            if row[2] == "S":
                continue
            print("%-8s %-8s %-3s %-10s %-8s %s" % row)
            shown += 1
            if shown >= a.show:
                break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
