"""量「引脚列的号画在离框边多远」以及 `TIP_PAD` 能救回多少。

用法：
    python work/tools/diag/_tipprobe.py --dir work/_fast_out_off9b --cases 1,3,5
    python work/tools/diag/_tipprobe.py --dir ... --case 1 --tip 40 --show 8

对每个 `box`/`bigic` 元件、每条边，打印：
    stubs=N  贴近桩的号=N（reach 26 内）  再远一点能救的=N（reach 26..26+tip 内且与桩对齐）
外加把每个「号」的中心到框边的**外距直方图**打出来，用来判断号到底画在多远。
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))
sys.path.insert(0, str(ROOT / "eda-pcb" / "stage1"))

import numpy as np                                            # noqa: E402
import cv2                                                    # noqa: E402
import pincol as P                                            # noqa: E402

SIDES = ("L", "R", "T", "B")


def _img(path):
    b = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(b, cv2.IMREAD_COLOR)


def _stage1(out_dir: Path, case: str):
    for p in (out_dir / "runs" / "stage1" / ("%s.stage1.json" % case),
              out_dir / ("%s.stage1.json" % case)):
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--cases", default="")
    ap.add_argument("--case", default="")
    ap.add_argument("--tip", type=float, default=40.0)
    ap.add_argument("--show", type=int, default=6)
    a = ap.parse_args()
    out = Path(a.dir)
    if not out.is_absolute():
        out = ROOT / out
    cases = [c for c in (a.cases.split(",") if a.cases else []) if c]
    if a.case:
        cases.append(a.case)
    if not cases:
        cases = [p.name.split(".")[0]
                 for p in sorted((out / "runs" / "stage1").glob("*.stage1.json"))]

    tot = collections.Counter()
    hist = collections.Counter()
    for case in cases:
        s = _stage1(out, case)
        if not s:
            continue
        img = _img(Path(s["image"]["path"])) if isinstance(s.get("image"), dict) \
            else _img(ROOT / "work" / "_fast_in" / ("%s.png" % case))
        texts = s.get("texts") or []
        comps = [c for c in (s.get("components") or [])
                 if str(c.get("type")) in P.IC_TYPES]
        # ★ 必须用**整图**的墨迹掩膜：`_stub_positions` 用原图坐标索引 mask。
        ink_full = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) < 170).astype(np.uint8)
        ink = P.text_free_ink(ink_full, texts or (), pad=0.0)
        rows = []
        for c in comps:
            box = c.get("bbox_img")
            if not box:
                continue
            for side in SIDES:
                st = P._stub_positions(ink, box, side) if side in ("L", "R", "T", "B") else []
                near = P._num_candidates(texts, box, side, stubs=st, tip_pad=0.0)
                far = [t for t in P._num_candidates(texts, box, side, stubs=st,
                                                    tip_pad=a.tip)
                       if t not in near]
                # 外距直方图：所有在 reach+tip 带内的小字
                wide = P._num_candidates(texts, box, side, stubs=st,
                                         reach=0.0, tip_pad=a.tip)
                for (along, ti) in wide:
                    b = texts[ti].get("bbox_img")
                    if side == "L":
                        d = x0 - (b[0] + b[2]) / 2.0
                    elif side == "R":
                        d = (b[0] + b[2]) / 2.0 - x1
                    elif side == "T":
                        d = y0 - (b[1] + b[3]) / 2.0
                    else:
                        d = (b[1] + b[3]) / 2.0 - y1
                    hist[int(d // 5) * 5] += 1
                tot["wide"] += len(wide)
                tot["stubs"] += len(st)
                tot["near"] += len(near)
                tot["tiprescue"] += len(far)
                rows.append((c.get("id"), side, len(st), len(near), len(far)))
        if rows and (a.show or a.case):
            print("--- case %s  (%d 个 IC)" % (case, len(comps)))
            for r in rows[:a.show]:
                print("    %-6s %s  stubs=%-3d 贴边号=%-3d TIP_PAD=%.0f 再救=%-3d"
                      % (r[0], r[1], r[2], r[3], a.tip, r[4]))
    print("=== 合计 === 桩 %d ｜ reach 内号 %d ｜ TIP_PAD=%.0f 额外救回 %d"
          % (tot["stubs"], tot["near"], a.tip, tot["tiprescue"]))
    print("外距直方图（5px 一桶）:", " ".join(
        "%d:%d" % (k, hist[k]) for k in sorted(hist)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
