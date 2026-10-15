# -*- coding: utf-8 -*-
"""GT 边的像素级覆盖归因：细笔画 / 骨架 / stage1 折线段 三把尺子对照。

用途：把「导线通道的亏空」定位到具体一层 —— 若某条 GT 边上骨架命中而
stage1 segments 不命中，则丢失发生在 wires.py 的「骨架 -> 折线段」环节。

坐标为 y 翻转帧：y_img = H - y_json。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(r"Y:\workspace\EDA")
sys.path.insert(0, str(ROOT / "eda-pcb"))

from pipeline import pskel  # noqa: E402
from schematic_parser.data import DEFAULT_DATA_ROOT, discover  # noqa: E402


def cases_by_id():
    out = {}
    for c in discover(DEFAULT_DATA_ROOT):
        cid = str(getattr(c, "case_id", "") or Path(getattr(c, "json_path", "")).parent.name)
        out[cid] = c
    return out


def _dt(mask: np.ndarray) -> np.ndarray:
    return cv2.distanceTransform((~mask).astype(np.uint8) * 255, cv2.DIST_L2, 3)


def _sample(p, q, step=2.0, trim=3.0):
    dx, dy = q[0] - p[0], q[1] - p[1]
    L = math.hypot(dx, dy)
    if L <= 2 * trim + 1.0:
        return []
    n = max(2, int(L / step))
    pts = []
    for i in range(n + 1):
        t = i / n
        if t * L < trim or (1 - t) * L < trim:
            continue
        pts.append((p[0] + t * dx, p[1] + t * dy))
    return pts


def _edges(doc):
    out = []
    for nm, net in (doc.get("nets") or {}).items():
        for ek, e in (net.get("edges") or {}).items():
            pts = [(float(a["x"]), float(a["y"])) for a in e]
            if len(pts) >= 2:
                out.append((nm, ek, pts[0], pts[-1]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="work/_fast_out_leadmove")
    ap.add_argument("--cases", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--pred", default="")
    ap.add_argument("--tol", type=float, default=2.0)
    a = ap.parse_args()

    idx = cases_by_id()
    tot = {"n_edge": 0, "thin": [], "skel": [], "seg": [], "gap": 0, "gap_len": 0.0,
           "e_len": 0.0, "gap_ex": []}
    for cid in [s.strip() for s in a.cases.split(",") if s.strip()]:
        c = idx.get(cid)
        if c is None:
            print("!! 找不到 case", cid)
            continue
        bgr = cv2.imread(str(c.png_path), cv2.IMREAD_COLOR)
        H = bgr.shape[0]
        gdoc = json.loads(Path(str(c.json_path)).read_text(encoding="utf-8"))
        st = json.loads((ROOT / a.run / "runs" / "stage1" / ("%s.stage1.json" % cid)).read_text(encoding="utf-8"))
        segs = (st.get("wire_detail") or {}).get("segments") or []
        ex = pskel.WireExtractor(ink_mode="adaptive", thin_w=2.6, use_colour=True)
        thin, _ink, _h = ex.foreground(bgr)
        thin = thin > 0
        skel = pskel.zhang_suen((thin * 255).astype(np.uint8)) > 0
        seg_mask = np.zeros(thin.shape, np.uint8)
        for s in segs:
            if len(s) >= 2:
                pts = np.array([[int(round(float(p[0]))), int(round(float(p[1])))] for p in s], np.int32)
                cv2.polylines(seg_mask, [pts], False, 255, 1)
        dt_t, dt_s, dt_g = _dt(thin), _dt(skel), _dt(seg_mask > 0)
        cls = {"skel_y_seg_n": 0, "len": 0.0, "gap_len": 0.0}
        rows = []
        for nm, ek, p0, p1 in _edges(gdoc):
            p = (p0[0], H - p0[1])
            q = (p1[0], H - p1[1])
            pts = _sample(p, q, 2.0, 3.0)
            if not pts:
                continue
            f = {"thin": 0.0, "skel": 0.0, "seg": 0.0}
            for (x, y) in pts:
                xi, yi = int(round(x)), int(round(y))
                if 0 <= yi < H and 0 <= xi < bgr.shape[1]:
                    f["thin"] += dt_t[yi, xi] <= a.tol
                    f["skel"] += dt_s[yi, xi] <= a.tol
                    f["seg"] += dt_g[yi, xi] <= a.tol
            n = float(len(pts))
            for k in f:
                f[k] /= n
            L = math.hypot(q[0] - p[0], q[1] - p[1])
            tot["n_edge"] += 1
            tot["e_len"] += L
            for k in ("thin", "skel", "seg"):
                tot[k].append(f[k])
            if f["skel"] >= 0.9 and f["seg"] < 0.9:
                tot["gap"] += 1
                tot["gap_len"] += L
                cls["gap_len"] += L
                if len(tot["gap_ex"]) < 12:
                    tot["gap_ex"].append((cid, nm, ek, p, q, f, st.get("wires", {}).get("hint")))
            cls["len"] += L
            rows.append((f["skel"] - f["seg"], nm, ek, p, q, f))
        rows.sort()
        mean = lambda k: (sum(tot[k][-len(rows):]) / max(1, len(rows)))  # noqa: E731
        print("case %-4s GT 边 %3d  细笔画 %.3f  骨架 %.3f  折线段 %.3f  骨架有/段无 %.1f px"
              % (cid, len(rows), mean("thin"), mean("skel"), mean("seg"), cls["gap_len"]))
        for d, nm, ek, p, q, f in rows[:2]:
            print("      最差 %s.%s (%.0f,%.0f)-(%.0f,%.0f) thin %.2f skel %.2f seg %.2f"
                  % (nm, ek, p[0], p[1], q[0], q[1], f["thin"], f["skel"], f["seg"]))
    n = max(1, tot["n_edge"])
    print("\n== 合计 %d 条 GT 边、%d px ==" % (tot["n_edge"], int(tot["e_len"])))
    for k in ("thin", "skel", "seg"):
        v = tot[k]
        print("   %-4s 平均覆盖 %.4f  完全覆盖(>=0.98) %d/%d  零覆盖 %d"
              % (k, sum(v) / n, sum(1 for x in v if x >= 0.98), n, sum(1 for x in v if x <= 0.02)))
    print("   骨架有、折线段无：%d 条 / %.0f px（占 GT 总长 %.1f%%）"
          % (tot["gap"], tot["gap_len"], 100.0 * tot["gap_len"] / max(1.0, tot["e_len"])))
    for cid, nm, ek, p, q, f, hint in tot["gap_ex"][:8]:
        print("     e.g. case %s %s.%s (%.0f,%.0f)-(%.0f,%.0f) thin %.2f skel %.2f seg %.2f hint=%s"
              % (cid, nm, ek, p[0], p[1], q[0], q[1], f["thin"], f["skel"], f["seg"], hint))


if __name__ == "__main__":
    main()
