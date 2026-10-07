# -*- coding: utf-8 -*-
"""分析一次 ultralytics 训练的 results.csv 曲线。"""
import csv
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

p = Path(sys.argv[1] if len(sys.argv) > 1 else
         r"Y:\workspace\EDA\eda-pcb\yolo_part\runs\detect\pcb43_v2_from_best\results.csv")
rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(open(p, encoding="utf-8"))]
print("file:", p)
print("epochs recorded:", len(rows))

M50, M95 = "metrics/mAP50(B)", "metrics/mAP50-95(B)"
P, R = "metrics/precision(B)", "metrics/recall(B)"
BL, CL, DL = "train/box_loss", "train/cls_loss", "train/dfl_loss"


def show(tag, r):
    print("%-14s ep=%-4s mAP50=%.4f  mAP50-95=%.4f  P=%.4f  R=%.4f  box=%.4f cls=%.4f dfl=%.4f lr=%.2e"
          % (tag, r["epoch"], float(r[M50]), float(r[M95]), float(r[P]), float(r[R]),
             float(r[BL]), float(r[CL]), float(r[DL]), float(r["lr/pg0"])))


show("best-mAP50", max(rows, key=lambda r: float(r[M50])))
show("best-mAP50-95", max(rows, key=lambda r: float(r[M95])))
show("last", rows[-1])

print("\n  ep   mAP50   mAP95    P       R      box    cls    dfl")
step = max(1, len(rows) // 20)
idx = list(range(0, len(rows), step))
if idx[-1] != len(rows) - 1:
    idx.append(len(rows) - 1)
for i in idx:
    r = rows[i]
    print("%4s  %.4f  %.4f  %.4f  %.4f  %.4f %.4f %.4f%s"
          % (r["epoch"], float(r[M50]), float(r[M95]), float(r[P]), float(r[R]),
             float(r[BL]), float(r[CL]), float(r[DL]),
             "   <- last" if i == len(rows) - 1 else ""))
