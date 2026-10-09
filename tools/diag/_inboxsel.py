"""_inboxsel.py -- 被框涂白吃掉的导线边，能不能用「独立证据」挑回来？

已定的两个读数（28 张，导线级）：
- `_occpol.py`：不涂自己的框 => GT 边覆盖 1093→1240（**+147 条**），但段 4493→6108
  （**+1615 条假阳**），F1 反而 0.3195→0.2917。
- `_prunepol.py`：`min_extent` / `max_axis_dev` / `_is_frame` 三道上游闸门**一条框内边都
  没吃掉**（框内覆盖恒为 17/662）=> 框内证据确实死在我们自己的框上。

所以问题只剩：**能不能只把那 147 条捞回来、不吞下 1615 条 fp**。本探针把 K(keep 段) 与
A(all 段) 的差集拆开看，并试几种「独立证据」选择器：
  S1 长度：K\A 段里 L >= --len 的
  S2 贴框：端点离某个框边 <= --edge px 的（引脚接线桩的形态）
  S3 框内+长度：中点落在框内 且 L >= --len
每档都报「GT 边覆盖 / 段数 / 段精度」，与 all 基线对比。
用法：python -X utf8 work\\tools\\dev\\_inboxsel.py --cases-file work\\_cases_w28.txt
"""
from __future__ import annotations

import argparse
import io
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"Y:\workspace\EDA")
EDA = ROOT / "eda-pcb"
for p in (str(EDA), str(EDA / "tools" / "eval"), str(ROOT / "work" / "tools" / "dev")):
    if p not in sys.path:
        sys.path.insert(0, p)

import cv2  # noqa: E402

from stage1.wires import extract_wires  # noqa: E402
import _wireedge as WE  # noqa: E402
import schematic_parser as S  # noqa: E402
import stage1_wires_eval as SE  # noqa: E402


def load_cases(f: str):
    txt = io.open(f, encoding="utf-8-sig").read()
    return [c.strip() for c in txt.replace("\r", "").replace("\n", ",").split(",") if c.strip()]


def d_edge(pt, bx):
    """点到框边界的「带符号」距离：正 = 在框外，负 = 在框内（取最近的边）。"""
    x, y = float(pt[0]), float(pt[1])
    x1, y1, x2, y2 = [float(v) for v in bx]
    dx = max(x1 - x, x - x2)
    dy = max(y1 - y, y - y2)
    return max(dx, dy)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases-file", default=str(ROOT / "work" / "_cases_w28.txt"))
    ap.add_argument("--stage1-dir", default=str(ROOT / "work" / "_full220" / "runs" / "stage1"))
    ap.add_argument("--png-dir", default=str(ROOT / "work" / "_submit_in220"))
    ap.add_argument("--tol", type=float, default=5.0)
    ap.add_argument("--len", type=float, default=30.0)
    ap.add_argument("--edge", type=float, default=8.0)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    cases = load_cases(a.cases_file)
    if a.limit:
        cases = cases[:a.limit]
    tags = ("all", "keep", "S1_len", "S2_edge", "S3_inlen")
    tot = {k: dict(cg=0, ng=0, cp=0, np_=0, nseg=0) for k in tags}
    tot_diag = {"n": 0, "ss": 0, "L": [], "dep": []}
    for cid in cases:
        png = Path(a.png_dir) / f"{cid}.png"
        sj = Path(a.stage1_dir) / f"{cid}.stage1.json"
        case = S.find_case(cid, S.DEFAULT_DATA_ROOT)
        if not (png.exists() and sj.exists() and case is not None):
            print(f"{cid}: skip")
            continue
        bgr = cv2.imread(str(png), cv2.IMREAD_COLOR)
        H = bgr.shape[0]
        s1 = json.load(io.open(sj, encoding="utf-8"))
        boxes = [[float(v) for v in c["bbox_img"]] for c in s1["components"]]
        txts = [[float(v) for v in t["bbox_img"]] for t in (s1.get("texts") or [])]
        GA = np.asarray(WE.gt_segs(case, H), np.float64).reshape(-1, 4)
        kw = dict(erase=txts, erase_pad=1.0, drop_frame=True, max_axis_dev=12.0,
                  min_extent=80.0, rescue_pad=8.0)
        wA, _ = extract_wires(bgr, occlude=boxes, **kw)
        wK, _ = extract_wires(bgr, occlude=[], **kw)
        SA = [list(s) for s in wA["segments"]]
        key = lambda s: (round(s[0][0], 1), round(s[0][1], 1), round(s[1][0], 1), round(s[1][1], 1))
        haveA = {key(s) for s in SA}
        only = [s for s in wK["segments"] if key(s) not in haveA]

        def seglen(s):
            return math.hypot(float(s[1][0]) - float(s[0][0]), float(s[1][1]) - float(s[0][1]))

        s1s = [s for s in only if seglen(s) >= a.len]
        s2s = [s for s in only
               if any(min(abs(d_edge(s[i], bx)) for bx in boxes) <= a.edge
                      for i in (0, 1)) if boxes]
        s3s = []
        for s in only:
            if seglen(s) < a.len:
                continue
            mid = ((float(s[0][0]) + float(s[1][0])) / 2.0, (float(s[0][1]) + float(s[1][1])) / 2.0)
            if any(d_edge(mid, bx) <= 0 for bx in boxes):
                s3s.append(s)
        sets = {"all": SA, "keep": wK["segments"], "S1_len": SA + s1s,
                "S2_edge": SA + s2s, "S3_inlen": SA + s3s}
        line = "%-14s n=%4d only=%4d (S1 %3d S2 %3d S3 %3d)" % (cid, len(GA), len(only),
                                                                len(s1s), len(s2s), len(s3s))
        for k in tags:
            PA = SE.seg_arrays(sets[k])
            cg, ng, cp, npd = WE.score(GA, PA, a.tol)
            t = tot[k]
            t["cg"] += cg; t["ng"] += ng; t["cp"] += cp; t["np_"] += npd; t["nseg"] += len(sets[k])
            line += " | %s %3d/%3d %4d/%4d" % (k, cg, ng, cp, npd)
        print(line)
        # ★ 形态诊断（用户 m42009 的关键一问）：`all` 漏掉、`keep` 捞回的那些 GT 边
        #   到底是什么形状？若它们是**短且贴框**的（符号自带的引脚线段），那它们本质上是
        #   元件图形而不是导线，涂白删掉是对的；若是**长且深入框内**的，才是真走线。
        def edge_cover_bool(PA):
            P = np.asarray(PA, np.float64).reshape(-1, 4)
            out = np.zeros(len(GA), bool)
            if not len(P):
                return out
            for i, (x1, y1, x2, y2) in enumerate(GA):
                ts = np.linspace(0.0, 1.0, 16)
                pts = np.stack([x1 + (x2 - x1) * ts, y1 + (y2 - y1) * ts], axis=1)
                ok = True
                for q in pts:
                    if SE.dist_pts_to_segs(q[None, :], P)[0] > a.tol:
                        ok = False
                        break
                out[i] = ok
            return out

        cA = edge_cover_bool(SA)
        cK = edge_cover_bool(wK["segments"])
        rec = np.where(cK & ~cA)[0]
        if len(rec):
            L = np.hypot(GA[rec, 2] - GA[rec, 0], GA[rec, 3] - GA[rec, 1])
            # 深度：16 个采样点到「任一框」的带符号距离的最小值（负 = 在框内多深）
            dep = []
            for i in rec:
                x1, y1, x2, y2 = GA[i]
                ts = np.linspace(0.0, 1.0, 16)
                best = 1e9
                for t in ts:
                    px, py = x1 + (x2 - x1) * t, y1 + (y2 - y1) * t
                    best = min(best, min(d_edge((px, py), bx) for bx in boxes))
                dep.append(best)
            dep = np.asarray(dep)
            short_shallow = int(((L <= 25.0) & (dep >= -8.0)).sum())
            print("      recovered n=%3d  长度 p50=%.0f p90=%.0f  深度 p50=%.0f  短且贴框 %d"
                  % (len(rec), float(np.percentile(L, 50)), float(np.percentile(L, 90)),
                     float(np.percentile(dep, 50)), short_shallow))
            tot_diag["n"] += len(rec)
            tot_diag["ss"] += short_shallow
            tot_diag["L"].extend(L.tolist())
            tot_diag["dep"].extend(dep.tolist())
    print("\n=== 合计（基线 all = 现状）===")
    for k in tags:
        t = tot[k]
        R = t["cg"] / max(1, t["ng"])
        P = t["cp"] / max(1, t["np_"])
        F1 = 2 * R * P / max(1e-9, R + P)
        print("  %-8s 边 %4d/%4d R=%.4f | 段 %5d/%5d P=%.4f | F1=%.4f | 段数 %5d"
              % (k, t["cg"], t["ng"], R, t["cp"], t["np_"], P, F1, t["nseg"]))
    if tot_diag["n"]:
        L = np.asarray(tot_diag["L"], np.float64)
        dep = np.asarray(tot_diag["dep"], np.float64)
        print("\n=== 被涂白吃掉的 GT 边形态（keep 捞回、all 漏掉）===")
        print("  n=%d  长度 p50=%.0f p90=%.0f p99=%.0f | 深度 p10=%.0f p50=%.0f p90=%.0f"
              % (len(L), float(np.percentile(L, 50)), float(np.percentile(L, 90)),
                 float(np.percentile(L, 99)), float(np.percentile(dep, 10)),
                 float(np.percentile(dep, 50)), float(np.percentile(dep, 90))))
        print("  短(<=25px)且贴框(深度>=-8px) = %d/%d (%.1f%%)  <-- 若是符号自带的引脚线段，"
              "涂白删掉是对的" % (tot_diag["ss"], len(L), 100.0 * tot_diag["ss"] / max(1, len(L))))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
