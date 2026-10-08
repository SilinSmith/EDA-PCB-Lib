# -*- coding: utf-8 -*-
"""`ink_recenter` 的三条硬测试（合成图，不碰真实数据）。

1) 框本来就套在墨迹上 -> 必须原样返回 'ok'（不许把好框挪走）。
2) 框空、符号在左边 5 px  -> 必须挪过去，且挪完框里有墨迹。
3) **老版本的坑**：一个「口」字形大连通域（质心在框中央的空白里）+ 框整个落在它内部空白中。
   老版本会把中心挪到质心（还是空白）；新版本必须落到笔画上（挪完框里有墨迹）。
"""
import importlib.util
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(r"Y:\workspace\EDA\eda-pcb\tools\dev\build_trainset_v4.py")
spec = importlib.util.spec_from_file_location("btv4", HERE)
m = importlib.util.module_from_spec(spec)
sys.modules["btv4"] = m
spec.loader.exec_module(m)


def blank():
    g = np.full((120, 160), 255, np.uint8)
    return g


def ink(g, x0, y0, x1, y1):
    g[y0:y1, x0:x1] = 0


ok = True
# 1) 好框不许动
g = blank()
ink(g, 40, 40, 80, 80)
r = m.ink_recenter(g, 60.0, 60.0, 30, 30)
print("1) 好框:", r, "->", "PASS" if r[2] == "ok" and (r[0], r[1]) == (60.0, 60.0) else "FAIL")
ok &= r[2] == "ok"

# 2) 空框、符号（细横条）在框上方 8 px
g = blank()
ink(g, 30, 40, 70, 45)
before = m.ink_frac(g, 60.0, 52.0, 20, 8)
r = m.ink_recenter(g, 60.0, 52.0, 20, 8)
f = m.ink_frac(g, r[0], r[1], 20, 8)
print(f"2) 偏 8px 的空框（挪前 {before:.3f}）: ({r[0]:.1f},{r[1]:.1f}) {r[2]}  挪完墨迹占比 {f:.3f}",
      "->", "PASS" if r[2] == "moved" and f > 0 else "FAIL")
ok &= r[2] == "moved" and f > 0

# 3) 大连通域的质心陷阱：「口」字形环，框落在内圈空白里、笔画离中心 ~12 px
g = blank()
ink(g, 60, 48, 100, 52)     # 上边
ink(g, 60, 70, 100, 74)     # 下边
ink(g, 56, 48, 60, 74)      # 左边
ink(g, 100, 48, 104, 74)    # 右边
# 该「口」的质心 ≈ (80, 61) —— 内圈空白正中；框中心放那里就是空框
before = m.ink_frac(g, 80.0, 61.0, 20, 8)
r = m.ink_recenter(g, 80.0, 61.0, 20, 8)
after = m.ink_frac(g, r[0], r[1], 20, 8)
print(f"3) 环形大blob内的空框: 挪前 {before:.3f} -> ({r[0]:.1f},{r[1]:.1f}) {r[2]} 挪后 {after:.3f}",
      "->", "PASS" if r[2] == "moved" and after > 0 else "FAIL")
ok &= r[2] == "moved" and after > 0

# 4) 真的一个墨迹都没有 -> 'noink_noblob'（调用方据此决定 drop）
g = blank()
r = m.ink_recenter(g, 60.0, 60.0, 20, 8)
print("4) 纯白:", r, "->", "PASS" if r[2] == "noink_noblob" else "FAIL")
ok &= r[2] == "noink_noblob"

print("\n全部通过" if ok else "\n有失败项")
sys.exit(0 if ok else 1)
