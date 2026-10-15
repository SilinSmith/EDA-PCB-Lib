# -*- coding: utf-8 -*-
"""直接调 stage2.pinrot.fix()（跳过整条管线，省掉每轮几分钟），把每个候选读串打出来。

用法（在仓库根 `Y:\workspace\EDA` 下）：
    python -X utf8 eda-pcb\tools\diag\_protorun.py [run_dir] [case]
默认 run_dir = work\_fast_out_c6、case = 6。会先设 STAGE2_PINROT=1 / STAGE2_PINROT_RAW=1。
"""
import json
import os
import sys

import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PKG = ROOT / "eda-pcb"
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(PKG / "stage2"))
os.environ["STAGE2_PINROT"] = "1"
os.environ["STAGE2_PINROT_RAW"] = "1"

import cv2                                            # noqa: E402

RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else (ROOT / "work" / "_fast_out_c6")
CASE = sys.argv[2] if len(sys.argv) > 2 else "6"


def imread_u(p):
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)


s1 = json.loads((RUN / "runs" / "stage1" / ("%s.stage1.json" % CASE)).read_text(encoding="utf-8"))
a = json.loads((RUN / "runs" / "stage2" / "_assigned" / ("%s.json" % CASE)).read_text(encoding="utf-8"))
doc = {"components": s1["components"], "pins": s1["pins"]}
bgr = imread_u(s1["image"]["path"])
print("图", bgr.shape, s1["image"]["path"])

from text import TextEngine                           # noqa: E402
import pinrot                                          # noqa: E402

eng = TextEngine(device="cpu")
st = pinrot.fix(bgr, doc, a, eng)
show = {k: v for k, v in st.items() if k != "raw"}
print(json.dumps(show, ensure_ascii=False, indent=1)[:3000])
print("---- 候选读串 ----")
for r in st["raw"]:
    print("raw", r)
(ROOT / "work" / "_protorun_st.json").write_text(
    json.dumps(st, ensure_ascii=False), encoding="utf-8")
