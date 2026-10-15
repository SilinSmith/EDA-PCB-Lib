# -*- coding: utf-8 -*-
"""对**指定的框**用 `vlm/pinname.py` 的口径问一次 VLM（复核 §5.91/§5.92 的名字框能不能读对）。

用法（仓库根）：
    python -X utf8 eda-pcb\\tools\\diag\\_boxvlm.py --case 5 --box "559,734,626,747" --box "560,755,635,768"

坐标 = **图像坐标系（左上原点）**，与 `runs/*/_assigned/*.json` 里的 `box` 同一套
（`_assigned` 的框已经是图像坐标，不需要 `H-y` 翻转）。

实测（10GT case 5 的 `U3`，`qwen3.5-9b`）：
    559,734,626,747（`ADJ(GND)` 真名框） -> 'ADJ(GND)'
    559,754,634,767（`VOUT(TAB)` 真名框） -> 'VOUT(TAB)'
    559,723,567,731（红点引脚标记）      -> 'UNCERTAIN'
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))

from schematic_parser.data import DEFAULT_DATA_ROOT, discover    # noqa: E402
from vlm import client as CL                                     # noqa: E402
from vlm import pinname as PN                                    # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="5")
    ap.add_argument("--box", action="append", required=True)
    ap.add_argument("--pad", type=float, default=3.0)
    ap.add_argument("--scale", type=float, default=4.0)
    a = ap.parse_args()
    img_p = None
    for c in discover(DEFAULT_DATA_ROOT):
        if c.case_id == a.case:
            img_p = Path(c.json_path).parent / ("%s.png" % a.case)
            break
    img = cv2.imdecode(np.fromfile(str(img_p), dtype=np.uint8), cv2.IMREAD_COLOR)
    print("图 %s  %dx%d" % (img_p.name, img.shape[1], img.shape[0]))
    print("prompt = %r" % (PN.PROMPT,))
    for spec in a.box:
        b = [float(v) for v in spec.split(",")]
        x0 = max(0.0, b[0] - a.pad); y0 = max(0.0, b[1] - a.pad)
        x1 = min(float(img.shape[1]), b[2] + a.pad); y1 = min(float(img.shape[0]), b[3] + a.pad)
        crop = img[int(round(y0)):int(round(y1)), int(round(x0)):int(round(x1))]
        up = CL.upscale(crop, a.scale)
        ans = CL.ask(up, PN.PROMPT)
        txt = ans[0] if isinstance(ans, (tuple, list)) else ans
        print("  box=%s -> %r" % (spec, txt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
