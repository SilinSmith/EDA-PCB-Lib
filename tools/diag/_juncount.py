# -*- coding: utf-8 -*-
"""统计某个产物目录里 stage1 工件的结点/段计数（看 `STAGE1_WIRE_CLOSE_K` 是否把交叉芯补回来了）。

用法: python -X utf8 work\\_juncount.py --run ck0 ck5 ck7 ck9
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(r"Y:\workspace\EDA")
sys.path.insert(0, str(ROOT / "eda-pcb"))

FIELDS = ["n_nodes", "junctions", "n_junction_nodes", "endpoints", "n_segments", "n_rays"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs="+", default=["ck0"])
    ap.add_argument("--cases", default="1,2,3,4,5,6,7,8,9,10")
    a = ap.parse_args()
    want = [x for x in a.cases.split(",") if x]
    for r in a.run:
        d = ROOT / "work" / ("_fast_out_" + r) / "runs" / "stage1"
        if not d.exists():
            print("%-6s 缺目录 %s" % (r, d))
            continue
        print("== %s ==" % r)
        print("   %-4s %s" % ("case", "  ".join("%-17s" % f for f in FIELDS)))
        tot = {f: 0 for f in FIELDS}
        for cid in want:
            p = d / ("%s.stage1.json" % cid)
            if not p.exists():
                continue
            st = json.loads(p.read_text(encoding="utf-8"))
            w = st.get("wires") or {}
            vals = []
            for f in FIELDS:
                v = w.get(f)
                v = len(v) if isinstance(v, list) else (v if isinstance(v, (int, float)) else 0)
                vals.append(v)
                tot[f] += v
            print("   %-4s %s" % (cid, "  ".join("%-17s" % v for v in vals)))
        print("   合计 %s" % "  ".join("%-17s" % tot[f] for f in FIELDS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
