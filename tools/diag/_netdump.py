"""打印某个 case 我们每条网的成员数 / 段数 / 段总长 / bbox。
用法: python work\\tools\\diag\\_netdump.py --pred work\\_gt10_ns0 --case 1
"""
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb")); sys.path.insert(0, str(ROOT / "work" / "tools" / "eval"))
from stage2_eval import build_index, resolve, load_json, DEFAULT_DATA_ROOT

ap = argparse.ArgumentParser(); ap.add_argument("--pred", required=True); ap.add_argument("--case", default="1")
a = ap.parse_args()
exact, digits = build_index(DEFAULT_DATA_ROOT)
case, how = resolve(a.case, exact, digits)
pr = load_json(Path(a.pred) / (case.case_id + ".json"))
nets = pr.get("nets") or {}
rows = []
for k, v in nets.items():
    hg = (v.get("hyperGraph") or "").strip("()")
    mem = [p for p in hg.split(",") if p.strip()]
    ed = list((v.get("edges") or {}).values())
    L = 0.0; xs = []; ys = []
    for e in ed:
        if len(e) >= 2:
            L += ((e[1]["x"]-e[0]["x"])**2 + (e[1]["y"]-e[0]["y"])**2) ** .5
            for p in e: xs.append(p["x"]); ys.append(p["y"])
    bb = "[%d,%d,%d,%d]" % (min(xs), min(ys), max(xs), max(ys)) if xs else "-"
    rows.append((len(mem), k, len(ed), round(L), bb, mem))
rows.sort(reverse=True)
for n, k, ne, L, bb, mem in rows:
    print("%-8s 成员 %-3d 段 %-3d 总长 %-5d bbox %-22s %s" % (k, n, ne, L, bb, mem[:8]))
print("合计 %d 条网 / %d 个引用" % (len(rows), sum(r[0] for r in rows)))
