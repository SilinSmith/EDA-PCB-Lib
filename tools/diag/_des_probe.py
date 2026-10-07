# -*- coding: utf-8 -*-
"""位号挂载探针：直接看某颗元件的**邻居文字框**和它们的 5 种读法。

    cd Y:\workspace\EDA\eda-pcb
    python -m work.tools.diag._des_probe 0001 0004 0006

或者（更简单）：

    & 'K:\python\311\python.exe' Y:\workspace\EDA\work\tools\diag\_des_probe.py 0001 0004 0006

对每颗 GT 里「Name 可读」的元件，打印：
* 我们有没有和它 IoU 配对上的元件框；
* 落在它 40px 内的所有文字框、每个框的 5 种读法、以及 `canon_designator` 在
  strict / force / off 三种模式下分别给什么 —— 用来回答「为什么没有产生位号候选」。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))
sys.path.insert(0, str(ROOT / "work" / "tools" / "viz"))

import cv2                        # noqa: E402
import numpy as np                # noqa: E402

import comp_cases as CC           # noqa: E402

import importlib                  # noqa: E402

# `stage2\__init__.py` 用同名函数遮蔽了 `stage2.assign` 子模块，所以
# `from stage2 import assign as A` / `import stage2.assign as A` 拿到的都是函数。
A = importlib.import_module("stage2.assign")                      # noqa: E402
from stage2.text import (TextEngine, imread_unicode,              # noqa: E402
                         read_stage1_texts)

STAGE1 = ROOT / "work" / "stage1_all14"
GT_ROOT = ROOT / "work" / "data"
DIAM = re.compile(r"^\u2205\d+$")


def gt_index():
    return CC.gt_index(CC.DEFAULT_DATA_ROOT)


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1])
    ub = (b[2] - b[0]) * (b[3] - b[1])
    u = ua + ub - inter
    return inter / u if u > 0 else 0.0


def main() -> None:
    cases = sys.argv[1:] or ["0001"]
    idx = gt_index()
    engine = TextEngine()
    for case in cases:
        s1 = STAGE1 / ("%s.stage1.json" % case)
        rec = idx.get(case)
        if not s1.exists() or rec is None or not rec.json_path:
            print("跳过 %s" % case)
            continue
        doc = json.loads(s1.read_text(encoding="utf-8"))
        gdoc = json.loads(Path(rec.json_path).read_text(encoding="utf-8"))
        img = imread_unicode(rec.png_path)
        H = float((doc.get("image") or {}).get("h") or (img.shape[0] if img is not None else 0))
        boxes = read_stage1_texts(doc)
        reads = engine.read(img, boxes) if img is not None else []
        comps = [{"id": c["id"], "type": c.get("type") or "other",
                  "bbox": [float(v) for v in c["bbox_img"]]}
                 for c in (doc.get("components") or []) if c.get("bbox_img")]

        print("=" * 78)
        print("%s  元件 %d  文字框 %d (%s)  图高 %.0f"
              % (case, len(comps), len(reads), (doc.get("image") or {}).get("path", ""), H))
        for gk, gv in (gdoc.get("components") or {}).items():
            gname = str(gv.get("Name") or "")
            if not gname or DIAM.match(gname):
                continue
            gb = gv.get("bbox")
            if not gb:
                continue
            gbox = [float(gb[0]), H - float(gb[3]), float(gb[2]), H - float(gb[1])]
            best, bi = 0.0, -1
            for i, c in enumerate(comps):
                v = iou(gbox, c["bbox"])
                if v > best:
                    best, bi = v, i
            tag = "配对%.2f" % best if best >= 0.1 else "**没配上**"
            print("\n-- GT %-14s type=%-12s %s" % (gname, gv.get("type"), tag))
            if bi < 0:
                continue
            c = comps[bi]
            print("   我们: %s type=%s bbox=%s" % (c["id"], c["type"], [round(x, 1) for x in c["bbox"]]))
            near = []
            for r in reads:
                b = r["bbox"]
                cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
                d = A.dist_to_rect(cx, cy, c["bbox"])
                if d <= A.FAR:
                    near.append((d, r))
            near.sort(key=lambda x: x[0])
            if not near:
                print("   **40px 内一个文字框都没有**")
                continue
            for d, r in near[:6]:
                rs = r.get("reads") or {}
                variants = []
                for mode in ("strict", "force", "off"):
                    A.PREFIX_MODE = mode
                    got = A._best_read(rs, "designator",
                                       lambda s, ty=c["type"]: A.canon_designator(s, ty)[1])
                    variants.append(mode + "=" + (A.canon_designator(got[1], c["type"])[0]
                                                  if got else "-"))
                A.PREFIX_MODE = "strict"
                shown = " ".join("%s:%s/%.2f" % (k, (v[0] or "")[:12], v[1])
                                 for k, v in list(rs.items())[:5])
                print("   d=%5.1f box=%s  %s | %s"
                      % (d, [round(x, 1) for x in r["bbox"]], shown, " ".join(variants)))


if __name__ == "__main__":
    main()
