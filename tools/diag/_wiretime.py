"""把 stage1 产物的 wires 耗时与图片规模/骨架复杂度关联起来。"""
import json, glob, sys
from pathlib import Path
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
rows = []
for f in glob.glob(r"Y:\workspace\EDA\work\stage1_all\*.stage1.json"):
    d = json.load(open(f, encoding="utf-8"))
    im, t, w = d["image"], d["timing"], d["wires"]
    rows.append(dict(t=t["wires_s"], case=d["case"], W=im["w"], H=im["h"],
                     mpx=im["w"] * im["h"] / 1e6, scale=w.get("scale"),
                     branch=w.get("n_branch_paths"), poly=w.get("n_polylines"),
                     junc=w.get("n_junctions"), nodes=w.get("n_nodes"),
                     thin=w.get("n_thin_px"),
                     segs=len((d.get("wire_detail") or {}).get("segments", []))))
rows.sort(key=lambda r: -r["t"])
hdr = f"{'case':<30}{'wires_s':>8}{'W':>6}{'H':>6}{'Mpx':>7}{'scale':>7}{'branch':>8}{'poly':>6}{'segs':>6}"
print(hdr)
print("-" * len(hdr))
for r in rows[:12]:
    print(f"{r['case']:<30}{r['t']:>8.1f}{r['W']:>6}{r['H']:>6}{r['mpx']:>7.2f}"
          f"{r['scale']:>7.2f}{r['branch']:>8}{r['poly']:>6}{r['segs']:>6}")

fast = [r for r in rows if r["t"] < 5]
slow = [r for r in rows if r["t"] > 60]
print(f"\nwires<5s: {len(fast)} 张  平均 {np.mean([r['mpx'] for r in fast]):.2f} Mpx"
      f"  平均 branch {np.mean([r['branch'] for r in fast]):.0f}")
print(f"wires>60s: {len(slow)} 张  平均 {np.mean([r['mpx'] for r in slow]):.2f} Mpx"
      f"  平均 branch {np.mean([r['branch'] for r in slow]):.0f}")
t = np.array([r["t"] for r in rows])
mpx = np.array([r["mpx"] for r in rows])
br = np.array([r["branch"] for r in rows], float)
print(f"\n相关度  wires_s ~ Mpx      : {np.corrcoef(t, mpx)[0,1]:+.3f}")
print(f"相关度  wires_s ~ branch   : {np.corrcoef(t, br)[0,1]:+.3f}")
print(f"相关度  wires_s ~ Mpx*branch: {np.corrcoef(t, mpx * br)[0,1]:+.3f}")
print(f"线性回归 wires_s ≈ {np.polyfit(mpx, t, 1)[0]:.1f}*Mpx + {np.polyfit(mpx, t, 1)[1]:.1f}")
tot_w = sum(r["t"] for r in rows)
ev = glob.glob("Y:/workspace/EDA/work/stage1_all/*.stage1.json")
tot_a = sum(json.load(open(f, encoding="utf-8"))["timing"]["total_s"] for f in ev)
print(f"\n总 wires 时间 {tot_w:.0f} s   占 total {tot_w/tot_a*100:.1f}%")
