"""Part 3b: per-arm comparison of the mode-validator's threshold sweep.

Reads the per-box rows dumped by _gnd_modes_p3.py and re-derives the sweep with
the decision-relevant numbers (what the dropped boxes actually were, and the
upper bound a perfect validator could reach).
"""
from __future__ import annotations

import json
import sys
from collections import Counter

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

rows = json.load(open(r"Y:\workspace\EDA\work\_gnd_p3_rows.json", encoding="utf-8"))["rows"]
box = [r for r in rows if "maxsim" in r]
miss = sum(1 for r in rows if r.get("miss"))
tp = sum(1 for r in box if r["tp"])
fp = len(box) - tp
n_sheets = len({r["case"] for r in box})
n_gt = tp + miss
print(f"sheets {n_sheets}  boxes {len(box)}  TP {tp}  FP {fp}  GT {n_gt}")
print(f"baseline  P {tp/len(box):.3f}  R {tp/n_gt:.3f}  FP/sheet {fp/n_sheets:.2f}")
print(f"ORACLE (drop every FP, keep every TP): P 1.000  R {tp/n_gt:.3f}  FP/sheet 0.00"
      f"   -> the entire headroom is {fp} boxes = {100*fp/len(box):.1f}% of predictions\n")

s_tp = np.array([r["maxsim"] for r in box if r["tp"]])
s_fp = np.array([r["maxsim"] for r in box if not r["tp"]])
q = (10, 25, 50, 75, 90)
print("maxsim   " + "".join(f"{p:>8}" for p in q) + f"{'mean':>8}")
for nm, s in (("TP", s_tp), ("FP", s_fp)):
    print(f"{nm:<9}" + "".join(f"{v:>8.3f}" for v in np.percentile(s, q)) + f"{s.mean():>8.3f}")

print("\n  thr | kept     P      R   FP/sheet | dropped dropTP dropFP  P(dropped) | AUC-style")
for thr in (0.3, 0.4, 0.5, 0.6, 0.7):
    keep = s_tp >= thr, s_fp >= thr
    ktp = int(keep[0].sum())
    kfp = int(keep[1].sum())
    dtp = tp - ktp
    dfp = fp - kfp
    print(f" {thr:.2f} | {ktp+kfp:5d} {ktp/max(ktp+kfp,1):7.3f} {ktp/n_gt:7.3f} "
          f"{kfp/n_sheets:8.2f} | {dtp+dfp:7d} {dtp:6d} {dfp:6d}  {dfp/max(dtp+dfp,1):10.3f}")

# how separable are TP and FP at all?  rank-based AUC with TP as the positive class
allv = np.concatenate([s_tp, s_fp])
order = np.argsort(allv)
ranks = np.empty(len(allv))
ranks[order] = np.arange(1, len(allv) + 1)
r_tp = ranks[:len(s_tp)].sum()
auc = (r_tp - len(s_tp) * (len(s_tp) + 1) / 2) / (len(s_tp) * len(s_fp))
print(f"\nrank AUC of maxsim separating TP from FP: {auc:.3f}  (0.5 = no information)")


def auc_of(vals_tp, vals_fp):
    allv = np.concatenate([vals_tp, vals_fp])
    order = np.argsort(allv)
    rk = np.empty(len(allv))
    rk[order] = np.arange(1, len(allv) + 1)
    return (rk[:len(vals_tp)].sum() - len(vals_tp) * (len(vals_tp) + 1) / 2) / (len(vals_tp) * len(vals_fp))


# ---- control: is mode similarity better than YOLO's own confidence score? ----
sc_tp = np.array([r["score"] for r in box if r["tp"]])
sc_fp = np.array([r["score"] for r in box if not r["tp"]])
print(f"yolo score: TP mean {sc_tp.mean():.3f} median {np.median(sc_tp):.3f} | "
      f"FP mean {sc_fp.mean():.3f} median {np.median(sc_fp):.3f}")
print(f"rank AUC  maxsim {auc:.3f}   yolo-score {auc_of(sc_tp, sc_fp):.3f}")
print(f"corr(maxsim, score) over all boxes: {np.corrcoef([r['maxsim'] for r in box], [r['score'] for r in box])[0,1]:.3f}")

print("\ncontrol sweep -- drop by YOLO score instead of by mode similarity:")
for thr in (0.30, 0.50, 0.70, 0.85, 0.92):
    ktp = int((sc_tp >= thr).sum())
    kfp = int((sc_fp >= thr).sum())
    print(f" score>={thr:.2f} | kept {ktp+kfp:5d}  P {ktp/max(ktp+kfp,1):5.3f}  R {ktp/n_gt:5.3f}  "
          f"FP/sheet {kfp/n_sheets:5.2f} | dropped TP {tp-ktp:4d} FP {fp-kfp:3d}")

print("\ncost per false positive removed (mode similarity):",
      {f"{t:.2f}": round((tp - int((s_tp >= t).sum())) / max(fp - int((s_fp >= t).sum()), 1), 1)
       for t in (0.3, 0.4, 0.5, 0.6, 0.7)})

by_src: dict = {}
for r in box:
    by_src.setdefault(r["case"], []).append(r)
jlc = [r for r in box if r["case"].startswith(("00", "01"))]
print("\nmode -> FP rate (same table as the run log):",
      {m: f"{sum(1 for r in box if r['mode']==m and not r['tp'])}/{sum(1 for r in box if r['mode']==m)}"
       for m in range(10)})
print("pred source mix:", dict(Counter(r["src"] for r in box)))
