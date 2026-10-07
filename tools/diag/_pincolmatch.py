# -*- coding: utf-8 -*-
"""逐元件对账：stage1 产物的 `pincol` 枚举点 vs GT box 元件的引脚点。

    python work\\tools\\diag\\_pincolmatch.py --cases 0001,0103,0019 --top 4

对每个枚举元件，选**命中最多**的那个 GT IC 元件（5px 内）作为对手，打印：
枚举数 / GT 数 / 命中 / 有桩 / 有号 / 纯几何外推（synth 且无号）。
用来看「多出来的脚」是从哪来的。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

EDA = Path(__file__).resolve().parents[3] / "eda-pcb"
sys.path[:0] = [str(EDA)]

from schematic_parser.data import DEFAULT_DATA_ROOT, IC_TYPES, discover, load_json  # noqa: E402

TOL = 5.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="0001,0103,0019,3,7,8")
    ap.add_argument("--stage1", default=r"Y:\workspace\EDA\work\_fast_out\runs\stage1")
    ap.add_argument("--top", type=int, default=4)
    a = ap.parse_args()
    cases = {c.case_id: c for c in discover(DEFAULT_DATA_ROOT)}
    for cid in [s.strip() for s in a.cases.split(",") if s.strip()]:
        p = os.path.join(a.stage1, "%s.stage1.json" % cid)
        if cid not in cases or not os.path.exists(p):
            print("%-6s 跳过" % cid)
            continue
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        gt = load_json(cases[cid].json_path)
        H = ((d.get("image") or {}).get("h")) or 0
        gp = gt.get("pins") or {}
        gpts = {}
        for k, v in (gt.get("components") or {}).items():
            if str((v or {}).get("type", "")).lower() not in IC_TYPES:
                continue
            pts = []
            for pk, info in (gp.get(k) or {}).items():
                pt = (info or {}).get("point") or {}
                if "x" in pt and "y" in pt:
                    pts.append((float(pt["x"]), float(H) - float(pt["y"])))
            if pts:
                gpts[k] = pts
        comps = {c["id"]: c for c in (d.get("components") or [])}
        cols = d.get("pincol") or {}
        if not cols:
            print("%-6s 无 pincol" % cid)
            continue
        rows = []
        for k, recs in cols.items():
            pts = [(float(r["x"]), float(r["y"])) for r in recs]
            best, bh = None, 0
            for gk, g in gpts.items():
                h = sum(1 for (x, y) in pts
                        if any(abs(x - gx) <= TOL and abs(y - gy) <= TOL for gx, gy in g))
                if h > bh:
                    best, bh = gk, h
            b = comps.get(k, {}).get("bbox_img") or [0, 0, 0, 0]
            n_ev = sum(1 for r in recs if r.get("synth") is False or r.get("out_texts"))
            n_synth = sum(1 for r in recs if r.get("synth") and not r.get("out_texts"))
            rows.append((len(recs), k, best, len(gpts.get(best) or []), bh,
                         n_ev, n_synth, round(b[2] - b[0]), round(b[3] - b[1])))
        rows.sort(key=lambda r: -r[0])
        print("== %s  GT IC: %s" % (cid, {k: len(v) for k, v in gpts.items()}))
        print("   %-7s %-10s %5s %5s %5s %5s %5s   %s"
              % ("cid", "GT对手", "枚举", "GT", "命中", "有据", "纯外推", "框"))
        for n, k, best, ng, bh, n_ev, n_synth, w, h in rows[:a.top]:
            print("   %-7s %-10s %5d %5d %5d %5d %5d   %dx%d"
                  % (k, best or "-", n, ng, bh, n_ev, n_synth, w, h))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
