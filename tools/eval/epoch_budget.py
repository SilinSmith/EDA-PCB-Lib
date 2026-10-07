# -*- coding: utf-8 -*-
"""从 results.csv 读 time 列，给出每 epoch 真实耗时与 2 小时能跑多少 epoch。"""
import csv
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
p = Path(sys.argv[1] if len(sys.argv) > 1 else
         r"Y:\workspace\EDA\eda-pcb\yolo_part\runs\detect\pcb43_v2_from_best\results.csv")
budget = float(sys.argv[2]) if len(sys.argv) > 2 else 2 * 3600.0

rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(open(p, encoding="utf-8"))]
t = [float(r["time"]) for r in rows]
ep = [int(float(r["epoch"])) for r in rows]
print("epochs        :", len(rows), " last epoch:", ep[-1])
print("total time    : %.1f s = %.2f h" % (t[-1], t[-1] / 3600.0))
print("mean s/epoch  : %.1f s" % (t[-1] / len(rows)))

# 去掉最慢/最快各 10% 后的稳健均值
mid = sorted(b - a for a, b in zip(t, t[1:]))[len(t) // 10: -max(1, len(t) // 10)]
if mid:
    print("robust s/epoch: %.1f s (median %.1f)" % (sum(mid) / len(mid), mid[len(mid) // 2]))
print()
print("=== 累计时间达到预算(%.0f s)的位置 ===" % budget)
for r, sec in zip(rows, t):
    if sec <= budget:
        last = (r, sec)
print("  最多跑到 epoch %s（累计 %.0f s = %.2f h）" % (last[0]["epoch"], last[1], last[1] / 3600.0))
print("  => 预算内约可跑 %s 个 epoch" % last[0]["epoch"])
print()
print("=== 按同样的 mean s/epoch 外推不同 imgsz/图片数的 epoch 预算 ===")
base = t[-1] / len(rows)
n = len(rows)
for imgsz, scale, nimg, tag in ((1024, 1.0, 3624, "当前设置"), (1024, 1.0, 2000, "砍到 2000 张"),
                                (1280, 1.56, 3624, "imgsz 1280"), (1536, 2.25, 3624, "imgsz 1536"),
                                (1536, 2.25, 1500, "imgsz 1536 + 1500 张"),
                                (1024, 1.0, 1500, "1024 + 1500 张")):
    sp = base * scale * (nimg / 3624.0)
    print("  %-22s %.1f s/epoch -> %5.1f epochs / 2h" % (tag, sp, budget / sp))
