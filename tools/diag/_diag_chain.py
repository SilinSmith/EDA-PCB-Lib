"""找到包含那条斜线的 polyline（RDP 之前的 chain），看跳变发生在哪一步。"""
import math
import sys

import cv2

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")
from pipeline.pskel import WireExtractor  # noqa: E402
from schematic_parser.data import discover, DEFAULT_DATA_ROOT  # noqa: E402

cases = {c.case_id: c for c in discover(DEFAULT_DATA_ROOT)}
case = cases["0101"]
bgr = cv2.imread(case.png_path, cv2.IMREAD_COLOR)
ex = WireExtractor()
res = ex.extract(bgr)

TARGETS = [(1179.0, 224.0), (980.0, 25.0), (19.0, 639.0), (208.0, 828.0)]
print(f"polylines {len(res.polylines)}  segments {len(res.segments)}")

found = set()
for i, chain in enumerate(res.polylines):
    for t in TARGETS:
        if any(math.hypot(p[0] - t[0], p[1] - t[1]) <= 2.0 for p in chain):
            if i in found:
                break
            found.add(i)
            # 找出这个 chain 里最长的两个连续顶点跳变
            best = []
            for a, b in zip(chain[:-1], chain[1:]):
                best.append((math.hypot(b[0] - a[0], b[1] - a[1]), a, b))
            best.sort(key=lambda r: -r[0])
            print(f"\n--- chain #{i}  顶点 {len(chain)}  命中 {t} ---")
            print("  最长 5 段跳变:")
            for L, a, b in best[:5]:
                print(f"    {L:8.1f}  ({a[0]:8.2f},{a[1]:8.2f}) -> ({b[0]:8.2f},{b[1]:8.2f})")
            print("  首尾:", [f"({p[0]:.1f},{p[1]:.1f})" for p in chain[:3]],
                  "...", [f"({p[0]:.1f},{p[1]:.1f})" for p in chain[-3:]])
            break

# 全局：chain 里有多少条 >50px 的跳变，它们是否压墨
thin, ink, chroma = ex.foreground(bgr)
big = []
for i, chain in enumerate(res.polylines):
    for a, b in zip(chain[:-1], chain[1:]):
        L = math.hypot(b[0] - a[0], b[1] - a[1])
        if L > 50.0:
            n = int(L)
            h = 0
            for k in range(n + 1):
                tt = k / n
                x = int(round(a[0] + (b[0] - a[0]) * tt))
                y = int(round(a[1] + (b[1] - a[1]) * tt))
                if 0 <= y < thin.shape[0] and 0 <= x < thin.shape[1] and thin[y, x]:
                    h += 1
            big.append((L, h / (n + 1), i, a, b))
big.sort(key=lambda r: -r[0])
print(f"\n所有 chain 里 >50px 的连续跳变：{len(big)} 条")
for L, cov, i, a, b in big[:15]:
    print(f"  L {L:8.1f} 压 thin {cov:6.1%}  chain#{i}  "
          f"({a[0]:7.1f},{a[1]:7.1f})->({b[0]:7.1f},{b[1]:7.1f})")
