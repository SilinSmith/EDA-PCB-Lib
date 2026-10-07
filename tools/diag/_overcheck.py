# -*- coding: utf-8 -*-
"""规则 (5)「穿越式微越过」的自检。

用手工造的两组几何验证：
  A 期望并：电容引线 `(194,172)->(194,238)`（竖）**跨过** GND 母线 `y=233`
    （`(202,233)->(71,233)`，横），只是端点多画了 5px  —— 这是 case 1 的真实现场。
  B 期望不并：电容符号的**两块平行极板** `(70,751)->(71,758)` 与
    `(77,751)->(78,759)`，相隔 7px 但**平行、不穿越** —— 这是 case 10 被
    `WIRE_TOL=10` 误并、把整个电路焊成一坨的元凶。

用法：python work\\tools\\diag\\_overcheck.py
"""
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "eda-pcb", "stage1"))

import labels as L  # noqa: E402

SEGS = np.array([
    [194.0, 172.0, 194.0, 238.0],   # 0 引线（竖，越界 5px）
    [202.0, 233.0, 71.0, 233.0],    # 1 母线（横）
    [70.0, 751.0, 71.0, 758.0],     # 2 极板甲
    [77.0, 751.0, 78.0, 759.0],     # 3 极板乙（与 2 平行、隔 7px）
], np.float64)


def main():
    ends = np.stack([SEGS[:, :2], SEGS[:, 2:]], axis=1).reshape(-1, 2)
    own = np.repeat(np.arange(len(SEGS)), 2)
    ok = True
    for over in (0.0, 12.0):
        u = L._overshoot_unions(SEGS, ends, own, over)
        g = L.wire_groups(SEGS, tol=2.0, weld=0.0, ext=0.0, over=over)
        lead_rail = g[0] == g[1]
        plates = g[2] == g[3]
        print("over=%-5g unions=%-16s groups=%s  引线-母线并=%s  两极板并=%s"
              % (over, u, g, lead_rail, plates))
        if over > 0.0:
            if not lead_rail:
                print("  ✗ 期望：引线应跨过母线并上")
                ok = False
            if plates:
                print("  ✗ 期望：两块平行极板**不应**被并")
                ok = False
    print("WIRE_TOL=%s WIRE_OVER=%s" % (L.WIRE_TOL, L.WIRE_OVER))
    print("规则 (5) 自检：%s" % ("全过" if ok else "**不过**"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
