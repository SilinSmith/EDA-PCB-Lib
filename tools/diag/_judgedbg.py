# -*- coding: utf-8 -*-
"""LLM 裁决的取景/提示词 A/B：同一个槽位，看模型到底吃到了什么、答出了什么。

用法：python _judgedbg.py [slot] [case]
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(r"Y:\workspace\EDA\eda-pcb")))
sys.stdout.reconfigure(encoding="utf-8")

import cv2                                                    # noqa: E402
import numpy as np                                            # noqa: E402

from fusion import claims as CL                               # noqa: E402
from fusion import priors as PR                               # noqa: E402
from fusion import solve as SV                                # noqa: E402
from fusion import llm_judge as LJ                            # noqa: E402
from vlm import client as VC                                  # noqa: E402

RUN = Path(r"Y:\workspace\EDA\work\_submit_out220r36\runs")
OUT = Path(r"Y:\workspace\EDA\work\preview")
slot = sys.argv[1] if len(sys.argv) > 1 else "comp.c0018.key"
case = sys.argv[2] if len(sys.argv) > 2 else "0001"

pool = CL.from_file(case, RUN / "stage1" / ("%s.stage1.json" % case),
                    RUN / "stage2" / "_assigned" / ("%s.json" % case))
PR.Priors().annotate(pool)
sol = SV.solve(pool)
q = [x for x in sol["open"] if x["slot"] == slot]
q = q[0] if q else {"slot": slot, "top": {"value": "?"}, "alt": {"value": "?"}}
reg = pool.region(slot)
bgr = LJ.load_image(pool)
print("case=%s slot=%s" % (case, slot))
print("  候选：" + ", ".join("%s=%.3f(%s)" % (a.value, a.score, a.source)
                             for a in pool.alts(slot)[:6]))
print("  本体框=%s" % (pool.meta.get(slot) or {}).get("cid"))
print("  region.box=%s focus=%s note=%s" % (
    [round(v) for v in reg.box] if reg else None,
    [round(v) for v in (reg.focus or [])] if reg else None,
    reg.note if reg else None))
if reg is not None and reg.focus:
    f = reg.focus
    print("  焦点是被谁给的：%s" % reg.src)
    print("  焦点框大小 %.0fx%.0f" % (f[2] - f[0], f[3] - f[1]))

KIND = LJ.KIND_HINT[LJ.kind_of(slot)]
variants = [
    ("as-is", LJ.PROMPT % KIND, 4, True),
    ("explicit", "看图。图中有一个**红色矩形框**，请只回答红框**内部**的字符，"
                 "原样照抄。忽略框外的任何文字。只输出那几个字符，不要解释。",
     4, True),
    ("bright", "看图。红色矩形框标出了一个%s。框里写的是什么？"
               "只输出框内的字符串。" % KIND, 4, True),
    ("no-red", LJ.PROMPT % KIND, 4, False),
    ("s4-tight", LJ.PROMPT % KIND, 4, True),
    ("s6", LJ.PROMPT % KIND, 6, True),
]

VC.preflight(strict=True)
for name, prompt, sc, red in variants:
    if name == "s4-tight" and reg is not None and reg.focus:
        tight = CL.Region(box=CL.grow_box(reg.focus, 6, 0), focus=None)
        img = tight.crop(bgr, scale=sc, focus_rect=False)
    else:
        img = reg.crop(bgr, scale=sc, focus_rect=red) if reg else None
    if img is None:
        print("  [%s] 切不出图" % name)
        continue
    p = OUT / ("_judgedbg_%s_%s.png" % (name.replace("-", ""), slot.replace(".", "_")))
    cv2.imwrite(str(p), img)
    txt, dt = VC.ask(img, prompt, timeout=120, max_tokens=64)
    print("  [%-9s] %-10s %.1fs  %dx%d  ->  %s" % (
        name, LJ.clean_answer(txt)[:20], dt, img.shape[1], img.shape[0],
        repr(str(txt)[:60])))
print("\n对照图：%s" % OUT)
print("  期望（人眼从红框读到）：%s" % "见对照图")
