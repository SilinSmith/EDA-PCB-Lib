"""看 Dataset300 的 CVJsonStd 里 texts 是否可信（精确框 + 内容）。"""
import glob
import json
import os
import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")

ROOT = r"Y:\workspace\EDA\eda-pcb\yolo_part\Dataset300我不知道干嘛用的先放置"
files = sorted(glob.glob(os.path.join(ROOT, "*", "*_CVJsonStd.json")))
print(f"CVJsonStd 文件 {len(files)} 个")

cat = Counter()
det = Counter()
n = 0
for f in files:
    j = json.load(open(f, encoding="utf-8"))
    for tid, t in (j.get("texts") or {}).items():
        n += 1
        cat[t.get("category")] += 1
        det[t.get("details")] += 1
print(f"texts 总数 {n}")
print("category:", dict(cat))
print("details :", dict(det))

# 看 0059 与 0003 的前若干条
for want in ("0059", "0003", "0016"):
    fs = [f for f in files if os.path.basename(os.path.dirname(f)) == want]
    if not fs:
        continue
    j = json.load(open(fs[0], encoding="utf-8"))
    print(f"\n--- {want}  canvas={j.get('canvas')}  texts={len(j.get('texts') or {})} ---")
    for tid, t in list((j.get("texts") or {}).items())[:14]:
        loc = t.get("location")
        print(f"  {tid:<6} cat={t.get('category'):<10} det={str(t.get('details')):<12} "
              f"content={str(t.get('content'))[:26]!r:<28} loc={loc}")
