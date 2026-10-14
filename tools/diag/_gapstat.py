"""看一眼某一跑里每张图的接线/结点计数（`work/_fast_out_<run>/runs/stage1/*.stage1.json`）。

用法：`python -X utf8 work\_gapstat.py gapb1`
"""
import glob
import json
import os
import sys

run = sys.argv[1] if len(sys.argv) > 1 else "gapb1"
pat = os.path.join(r"Y:\workspace\EDA\work\_fast_out_" + run, "runs", "stage1", "*.stage1.json")
files = sorted(glob.glob(pat))
if not files:
    print("没有 stage1 产物：", pat)
    sys.exit(1)
print("%-8s %8s %10s %8s %8s %9s" % ("case", "bridges", "junc_nodes", "segs", "juncs", "endpoints"))
tb = tj = 0
for p in files:
    with open(p, encoding="utf-8") as fh:
        d = json.load(fh)
    w = d.get("wires") or {}
    cid = os.path.basename(p).split(".")[0]
    nb = int(w.get("n_gap_bridges", 0) or 0)
    jn = int(w.get("n_junction_nodes", 0) or 0)
    tj += jn
    tb += nb
    print("%-8s %8d %10d %8s %8s %9s" % (
        cid, nb, jn, w.get("n_segments"),
        len(w.get("junctions") or []), w.get("n_endpoints")))
print("合计：接线 %d 条，结点 %d 个" % (tb, tj))
