# -*- coding: utf-8 -*-
"""「孤儿引脚」体检：`pins` 里有、却不在任何网里的引脚，离最近的网有多远？

背景（`work/_netwhy.py` 的归因）：netH 的 ③ 里有 6 条是「这颗脚**任何网都没包含**」，
典型是 case 7/10 的 `LED1.1`/`LED1.2` —— LED 的脚点在 `pins[LED1]` 里好好地写着，
但它既不在自己的网里、也不在任何别的网里；而它 GT 的同伴（`R5.1`/`R4.1`）却**各自单独**
成了一条单成员网。这看起来是「网生成时这颗脚没被接上」，而不是「接错了网」。

本探针不修任何东西，只量机会与风险：
  * 对每颗孤儿脚，算它到**每条网每一段边**的最短距离；
  * 标记这条脚**是否是 GT 的网成员**（用 `assign_components` + `rewrite_keys` 后的成员名）；
  * 若是，检查「最近的网」是不是 GT 里与它同网的那些同伴所在的网 ⇒ 区分「能救」与「会误接」。

用法：
    python -X utf8 work/_orphan.py
    python -X utf8 work/_orphan.py --dir work/_fast_out_cd10on --radius 12
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"Y:\workspace\EDA")
EDA = ROOT / "eda-pcb"
sys.path.insert(0, str(EDA))
DSROOT = EDA / "yolo_part" / "Dataset(waitforpreprocess)"
from schematic_parser import metrics as M                       # noqa: E402

CASES = ("1", "2", "3", "4", "5", "6", "7", "8", "9", "10")


def gt_of(case):
    for p in DSROOT.rglob(case + ".json"):
        if p.parent.name == case:
            return json.load(open(p, encoding="utf-8"))
    return None


def hg_members(hg: str):
    """把 `"(U1.5,C1.2)"` 解析成 `["U1.5", "C1.2"]`（与 `assemble._hg_refs` 同口径）。"""
    s = str(hg or "").strip().strip("()")
    return [x.strip() for x in s.split(",") if x.strip()]


def _pt(p):
    return float(p["x"]), float(p["y"])


def _seg_dist(px, py, a, b):
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    dd = dx * dx + dy * dy
    t = 0.0 if dd <= 1e-9 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / dd))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def net_edges(net):
    out = []
    for seg in (net.get("edges") or {}).values():
        try:
            a, b = _pt(seg[0]), _pt(seg[1])
        except Exception:                                        # noqa: BLE001
            continue
        out.append((a, b))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "work" / "_fast_out_cd10on"))
    ap.add_argument("--radius", type=float, default=12.0)
    ap.add_argument("--cases", default="")
    a = ap.parse_args()
    cases = [c.strip() for c in a.cases.split(",") if c.strip()] or list(CASES)
    cfg = M.EvalConfig()
    tot = defaultdict(int)
    for case in cases:
        g = gt_of(case)
        pf = Path(a.dir) / "final_results" / (case + "_Pred.json")
        if g is None or not pf.exists():
            continue
        pred = json.load(open(pf, encoding="utf-8"))
        _comp, keymap = M.assign_components(g, pred, cfg, None)
        p2 = M.rewrite_keys(pred, keymap)
        # GT 成员 → 它 GT 那条网的同伴
        gt_partner = {}
        for _n, s in M._net_sets(g.get("nets") or {}):
            mems = [m for m in s if isinstance(m, tuple) and len(m) == 2]
            for m in mems:
                gt_partner[(str(m[0]), str(m[1]))] = set("%s.%s" % (x, y) for x, y in mems)
        # 我们每条网的成员集
        ours = []
        for nid, net in (p2.get("nets") or {}).items():
            mems = set("%s.%s" % (r[0], r[1]) for r in (net.get("_refs") or [])
                       if isinstance(r, (list, tuple)) and len(r) == 2)
            if not mems:
                mems = set(hg_members(net.get("hyperGraph", "")))
            ours.append((nid, mems, net_edges(net)))
        in_any = set()
        for _nid, mems, _e in ours:
            in_any |= mems
        orphan = []
        for ck, pv in (p2.get("pins") or {}).items():
            for pk, pv2 in (pv or {}).items():
                num = str(pk).replace("pin_", "")
                name = "%s.%s" % (ck, num)
                if name in in_any:
                    continue
                pt = (pv2 or {}).get("point")
                if not pt:
                    continue
                px, py = _pt(pt)
                best = (1e9, None, None)
                for nid, mems, edges in ours:
                    for a2, b2 in edges:
                        d = _seg_dist(px, py, a2, b2)
                        if d < best[0]:
                            best = (d, nid, mems)
                orphan.append((name, best[0], best[1], best[2] or set(), gt_partner.get((ck, num))))
        if not orphan:
            print("%-4s ✓ 没有孤儿脚" % case)
            continue
        print("=" * 100)
        print("case %s：孤儿脚 %d 颗（半径 %.0fpx 内）" % (case, len(orphan), a.radius))
        for name, d, nid, mems, partner in sorted(orphan, key=lambda x: x[1]):
            tag = "GT成员" if partner else "（非GT成员）"
            hit = ""
            if partner and d <= a.radius:
                # 最近的网是否包含 GT 同伴
                shared = sorted(partner & mems)
                hit = "★能救" if shared else "⚠误接(最近网无GT同伴)"
            elif partner:
                hit = "太远"
            elif d <= a.radius:
                hit = "⚠非GT脚会接上"
            tot[("GT孤儿" if partner else "非GT孤儿")] += 1
            if partner and d <= a.radius and "★" in hit:
                tot["★能救"] += 1
            print("  %-12s 最近网 %-7s d=%6.1fpx  %-10s %s  最近网成员=%s"
                  % (name, nid, d, tag, hit, ",".join(sorted(mems))[:60]))
    print()
    print("合计：", dict(tot))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
