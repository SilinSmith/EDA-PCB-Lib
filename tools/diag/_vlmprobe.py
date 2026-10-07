# -*- coding: utf-8 -*-
"""探一次：这个端点上的模型到底能不能读图（用 vlm.client 的同一条通道）。"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.abspath(os.path.join(HERE, "..", "..", "..", "eda-pcb"))
sys.path.insert(0, PKG)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from vlm import client as C  # noqa: E402

print("BASE     =", C.BASE)
print("MODEL    =", C.MODEL)
print("KEY      =", (C.KEY[:12] + "...") if C.KEY else "**未设**")
print("REASONING=", C.REASONING, "| SCALE =", C.SCALE, "| MAXTOK =", C.MAXTOK)

# 造一张能靠眼睛分辨的图：黑底白字 "C151"，故意不用 32x32 那种糊图
img = np.full((120, 420, 3), 255, np.uint8)
cv2.putText(img, "C151", (12, 92), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 0), 6)
cv2.imwrite(os.path.join(HERE, "_vlmprobe_src.png"), img)

print("\n-- probe() --")
try:
    print(C.probe(timeout=120.0))
except Exception as e:  # noqa: BLE001
    print("probe 抛异常:", repr(e))

print("\n-- ask: 图里写了什么 --")
txt, dt = C.ask(img, "图里的字符串是什么？只回字符串本身，不要解释。",
                timeout=180.0, max_tokens=256)
print("[%.1fs] %r" % (dt, txt))

print("\n-- ask: 图上是不是有一行字符，读出来（考它认不认字）--")
txt2, dt2 = C.ask(img, "读出图中唯一的文字。只输出文字。", timeout=180.0, max_tokens=256)
print("[%.1fs] %r" % (dt2, txt2))
