"""列出 stage1 产物里某个 case 的长斜线段，用来确认大尺度斜线是否消失。"""
import json
import math
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
case = sys.argv[1] if len(sys.argv) > 1 else "0101"
root = sys.argv[2] if len(sys.argv) > 2 else r"Y:\workspace\EDA\work\stage1_all7"
minlen = float(sys.argv[3]) if len(sys.argv) > 3 else 50.0

p = os.path.join(root, f"{case}.stage1.json")
d = json.load(open(p, encoding="utf-8"))
segs = d["wire_detail"]["segments"]
axes = [0.0, 45.0, 90.0, 135.0]
bad, long_all, diag = [], [], []
for s in segs:
    (x1, y1), (x2, y2) = s[0], s[1]
    L = math.hypot(x2 - x1, y2 - y1)
    if L < minlen:
        continue
    ang = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0
    dev = min(min(abs(ang - a), 180.0 - abs(ang - a)) for a in axes)
    long_all.append((L, ang, dev, (x1, y1), (x2, y2)))
    if dev > 12.0:
        bad.append((L, ang, dev, (x1, y1), (x2, y2)))
    d45 = abs(ang - 45.0)
    if d45 <= 12.0:
        diag.append((L, ang, d45, (x1, y1), (x2, y2)))

print(f"{root}\\{case}.stage1.json   线段总数 {len(segs)}   L>={minlen:.0f} 的 {len(long_all)} 条")
print(f"    其中偏离轴向 12 度以上（应被 max_axis_dev 拦掉）：{len(bad)} 条")
print(f"    其中落在 45 度家族（±12 度，斜线伪影的嫌疑对象）：{len(diag)} 条")
for L, ang, dd, a, b in sorted(diag, key=lambda r: -r[0])[:10]:
    print(f"      L {L:7.1f}  ang {ang:6.1f}   ({a[0]:7.1f},{a[1]:7.1f}) -> ({b[0]:7.1f},{b[1]:7.1f})")
