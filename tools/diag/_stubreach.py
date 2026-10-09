"""引脚桩「远场判据」(`STAGE1_PINC_STUB_REACH`) 的离线取证。

问题：`stage1/pincol.py::_stub_positions` 旧判据只看「离框边最近的 `connect+2`
列有墨」，于是**贴在边上的引脚号数字**也被当成桩 ⇒ 证据里混进一串「真桩 ±10px」
的假位置 ⇒ `_fit_pitch` 把 pitch 拟成真值的一半 ⇒ `_lattice` 枚举出成倍的幻影脚。
（实测 case 8 的 U1 左列：图上真桩 11 根，free-ink 的 `_stub_positions` 给 17 个，
`_fit_pitch` 给 11.0，最终枚举出 19 根 —— 而 GT 只有 11 根。）

本探针把「桩证据」与 GT 引脚按侧对齐，直接量 **桩证据的 P/R**（reach 关 vs 开），
以及 `enumerate_component` 的输出根数变化。

用法::

    python -X utf8 tools/diag/_stubreach.py                 # 官方 10 张
    python -X utf8 tools/diag/_stubreach.py --reach 12 --tol 6
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
PKG = ROOT / "eda-pcb"
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))

from stage1 import pincol  # noqa: E402

GT_DIR = PKG / "yolo_part" / "Dataset(waitforpreprocess)" / "10GT"
S1_DIR = ROOT / "work" / "_fast_out_dlv10" / "runs" / "stage1"
IC_TYPES = ("box", "bigic", "ic")


def imread_u(p) -> np.ndarray:
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


def side_of(bx: float, by: float, bb) -> str:
    x0, y0, x1, y1 = bb
    d = {"L": abs(bx - x0), "R": abs(bx - x1), "T": abs(by - y0), "B": abs(by - y1)}
    return min(d, key=d.get)


def gt_pins(js, name, bbox_img):
    h = float(js.get("_h") or 0.0)
    out = []
    for pk, rec in (js.get("pins") or {}).get(name, {}).items():
        pt = rec.get("point") or {}
        x, y = float(pt.get("x", 0.0)), float(pt.get("y", 0.0))
        iy = h - y
        out.append((x, iy, rec.get("pinname")))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--reach", type=float, default=12.0)
    ap.add_argument("--tol", type=float, default=6.0)
    ap.add_argument("--enumerate", action="store_true", help="同时跑 enumerate_component")
    a = ap.parse_args(argv)

    tot = {0.0: [0, 0, 0], a.reach: [0, 0, 0]}   # reach -> [tp, fp, fn]
    for cid in [c.strip() for c in a.cases.split(",") if c.strip()]:
        png = GT_DIR / cid / ("%s.png" % cid)
        gtj = GT_DIR / cid / ("%s.json" % cid)
        s1p = S1_DIR / ("%s.stage1.json" % cid)
        if not (png.exists() and gtj.exists() and s1p.exists()):
            continue
        img = imread_u(png)
        H = img.shape[0]
        js = json.loads(gtj.read_text(encoding="utf-8"))
        js["_h"] = H
        s1 = json.loads(s1p.read_text(encoding="utf-8"))
        texts = s1.get("texts") or []
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ink = (gray < 170).astype(np.uint8)
        free = pincol.text_free_ink(ink, texts, pad=0.0)

        ic = [c for c in s1.get("components") or []
              if str(c.get("type", "")).lower() in IC_TYPES]
        name_of = {}
        for c in ic:
            bb = [float(v) for v in c["bbox_img"]]
            best, bk = None, None
            for nm, rec in (js.get("components") or {}).items():
                if str(rec.get("type", "")).lower() not in IC_TYPES:
                    continue
                gb = [float(v) for v in rec["bbox"]]
                gbi = [gb[0], H - gb[3], gb[2], H - gb[1]]
                d = abs(gbi[0] - bb[0]) + abs(gbi[1] - bb[1])
                if best is None or d < best:
                    best, bk = d, nm
            if bk is not None and best is not None and best < 40:
                name_of[c["id"]] = bk

        for c in ic:
            bb = [float(v) for v in c["bbox_img"]]
            nm = name_of.get(c["id"])
            if nm is None:
                continue
            g = gt_pins(js, nm, bb)
            gby = {"L": [], "R": [], "T": [], "B": []}
            for x, y, _pn in g:
                gby[side_of(x, y, bb)].append(y if side_of(x, y, bb) in ("L", "R") else x)
            line = ["case %-6s %-4s bbox=%s GT=%d" % (cid, nm, [round(v) for v in bb], len(g))]
            for reach in (0.0, a.reach):
                pincol.STUB_REACH = reach
                got = []
                for side in ("L", "R", "T", "B"):
                    v = pincol._stub_positions(free, bb, side)
                    got += list(v) if side in ("L", "R") else list(v)
                # 逐侧匹配
                tp = fp = fn = 0
                for side in ("L", "R"):
                    gg = sorted(gby[side])
                    pp = sorted(pincol._stub_positions(free, bb, side))
                    used = [False] * len(gg)
                    for q in pp:
                        hit = -1
                        for i, gv in enumerate(gg):
                            if not used[i] and abs(gv - q) <= a.tol:
                                hit = i
                                break
                        if hit >= 0:
                            used[hit] = True
                            tp += 1
                        else:
                            fp += 1
                    fn += sum(1 for u in used if not u)
                tot[reach][0] += tp
                tot[reach][1] += fp
                tot[reach][2] += fn
                line.append("reach=%-5g stubs=%2d  tp=%2d fp=%2d fn=%2d" % (reach, len(got), tp, fp, fn))
            print("  ".join(line))

    print()
    for reach in (0.0, a.reach):
        tp, fp, fn = tot[reach]
        p = tp / float(tp + fp) if tp + fp else 0.0
        r = tp / float(tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        print("reach=%-5g 桩证据 P=%.4f R=%.4f F1=%.4f  (tp %d / fp %d / fn %d)"
              % (reach, p, r, f, tp, fp, fn))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
