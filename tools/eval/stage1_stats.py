"""汇总 stage1 在全部 220 个官方 case 上的产物：元件数/类型对照 + 耗时分布。"""
import json, sys, time
from pathlib import Path
from collections import Counter
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")
from schematic_parser import data as D

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else r"Y:\workspace\EDA\work\stage1_all")

files = sorted(OUT.glob("*.stage1.json"))
print(f"产物 {len(files)} 个")

pred = Counter(); gt = Counter()
n_comp = n_pending = n_att = 0
tow, toy, tot_s = [], [], []
slow = []
cases = {c.case_id: c for c in D.discover(D.DEFAULT_DATA_ROOT)}
for f in files:
    d = json.load(open(f, encoding="utf-8"))
    for x in d["components"]:
        pred[x["type"]] += 1
        n_comp += 1
        if x["pending"]:
            n_pending += 1
    n_att += d["stats"]["n_attached"]
    t = d["timing"]
    tow.append(t["wires_s"]); toy.append(t["yolo_s"]); tot_s.append(t["total_s"])
    slow.append((t["total_s"], d["case"]))
    cid = d["case"].split("-")[-1] if "-" in d["case"] else d["case"]
    c = cases.get(cid)
    if c:
        doc = D.load_json(c.json_path)
        for k, comp in (D.normalise_doc(doc)["components"] if isinstance(D.normalise_doc(doc), dict) else {}).items():
            gt[comp.get("type")] += 1

def pct(a, q):
    return float(np.percentile(a, q)) if a else 0.0

print(f"\n元件  pred {n_comp}   gt {sum(gt.values())}   差 {n_comp - sum(gt.values()):+d}")
print(f"待裁决 {n_pending}   已接线 {n_att}  ({100.0*n_att/max(1,n_comp):.1f}%)")
print(f"\n耗时   total_s  p50 {pct(tot_s,50):.1f}  p90 {pct(tot_s,90):.1f}  max {max(tot_s):.1f}"
      f"   串行合计 {sum(tot_s):.0f} s")
print(f"       wires_s  p50 {pct(tow,50):.1f}  p90 {pct(tow,90):.1f}  max {max(tow):.1f}"
      f"   占比 {100.0*sum(tow)/sum(tot_s):.1f}%")
print(f"       yolo_s   p50 {pct(toy,50):.2f}  p90 {pct(toy,90):.2f}  max {max(toy):.2f}")
n_over = sum(1 for v in tot_s if v > 120)
print(f"       超过 120 s 的 case: {n_over} / {len(tot_s)}")
print("\n最慢 10 个（total_s）：")
for v, c in sorted(slow, reverse=True)[:10]:
    print(f"   {c:<28} {v:7.1f} s")

print(f"\n{'type':<18}{'pred':>7}{'gt':>7}{'diff':>8}{'召回':>8}")
keys = sorted(set(pred) | set(gt), key=lambda k: -gt.get(k, 0))
for k in keys:
    p_, g_ = pred.get(k, 0), gt.get(k, 0)
    r = (min(p_, g_) / g_) if g_ else float("nan")
    rs = f"{r:.3f}" if g_ else "  -"
    print(f"{k:<18}{p_:>7}{g_:>7}{p_-g_:>+8}{rs:>8}")
