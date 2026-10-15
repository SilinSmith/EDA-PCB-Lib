# -*- coding: utf-8 -*-
"""_nlcensus.py —— 切点普查（cut-position census，§5.127 第 1 步）。

目的（父 agent m01255 第 1 步，**只看不改**）：回答「GT 的段端点和我们自己的段端点，
各自能落在哪些**不看 GT 就能算出来**的候选锚点附近」，据此判断 netL 的**切分划分重划**
（切/并）是否还有可用杠杆。

候选锚点（每案各一套，全部只用图像 + 我们自己的产物，不读 GT）：
  (a1) node_raw      pskel 骨架度>=3 结点（`PSKEL_NODE_SOLVE=0` 的 `res.junctions`）
  (a2) node_solved   同上但结点位置重解之后（`PSKEL_NODE_SOLVE=1` 的段端点，按最近 raw 结点配对）
  (b)  dot           实心圆点质心（高饱和红色连通块，EDT 厚度 >= 2.0、面积 >= 15）
  (c)  colour_tr     maroon(引线) <-> green(导线) 的色转折点（绿像素 5x5 邻域内含 maroon）
  (d1) bbox          我们检出的元件 bbox 矩形轮廓（每 1px 采样）
  (d2) lead_tip      元件外伸引线段（bbox 外的 maroon 连通块）沿主轴的两个端点
  (d3) pin_our       我们自己检出的引脚锚点
  (e1) corner        pskel 折线拐角（RDP 后内顶点，转角 >= 25 度）
  (e2) tee           我方边的轴交点型 T 点（另一条边的端点到该交点 2~40px 且交点落在其跨度内）
另有两个对照锚点（**会用到 GT**，只作为上界参考，不参与「可用杠杆」判定）：
  (x1) gt_pin        GT 引脚锚点
  (x2) gt_end        GT 段端点自身

输出：列 = 特征、行 = GT端 / 我方端 的覆盖率表（<=3px / <=5px / <=10px 三档），
以及「GT 端被某特征 <=5px 覆盖、但我们那边 3px 内没有任何端点」的**缺口清单**（按条数排序）。

用法：
  python -X utf8 tools/diag/_nlcensus.py --cases 1,2,3,4,5,6,7,8,9,10
  python -X utf8 tools/diag/_nlcensus.py --gt-dir delivery --run-dir <dir> --out <json>
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from schematic_parser.data import discover, DEFAULT_DATA_ROOT   # noqa: E402

DEFAULT_RUN = "Y:/workspace/EDA/work/_fast_out_deterministic_default"


# ------------------------------------------------------------------ doc utils

def _ends_of(doc):
    """文档里所有 net 的所有边的两个端点（doc 坐标）。"""
    out = []
    nets = doc.get("nets") or {}
    for nk in nets:
        edges = nets[nk].get("edges") or {}
        for ek in edges:
            pts = edges[ek]
            if not isinstance(pts, list) or len(pts) < 2:
                continue
            for p in (pts[0], pts[-1]):
                try:
                    out.append((float(p["x"]), float(p["y"])))
                except Exception:
                    pass
    return out


def _pins_of(doc):
    out = []
    pins = doc.get("pins") or {}
    for k in pins:
        v = pins[k]
        if isinstance(v, dict):
            for pk in v:
                p = v[pk]
                if isinstance(p, dict) and isinstance(p.get("point"), dict):
                    try:
                        out.append((float(p["point"]["x"]), float(p["point"]["y"])))
                    except Exception:
                        pass
    return out


def _bboxes_of(doc):
    out = []
    comps = doc.get("components") or {}
    for k in comps:
        bb = comps[k].get("bbox")
        if isinstance(bb, (list, tuple)) and len(bb) == 4:
            try:
                out.append(tuple(float(v) for v in bb))
            except Exception:
                pass
    return out


def _edges_of(doc):
    """[(p0, p1)] doc 坐标。"""
    out = []
    nets = doc.get("nets") or {}
    for nk in nets:
        edges = nets[nk].get("edges") or {}
        for ek in edges:
            pts = edges[ek]
            if isinstance(pts, list) and len(pts) >= 2:
                try:
                    out.append(((float(pts[0]["x"]), float(pts[0]["y"])),
                                (float(pts[-1]["x"]), float(pts[-1]["y"]))))
                except Exception:
                    pass
    return out


# ------------------------------------------------------------------ geometry

class Cloud:
    """一组候选锚点；查询「每个目标点到该集合的最小距离」。"""

    def __init__(self, pts, name):
        self.name = name
        a = np.asarray(pts, dtype=float)
        if a.ndim != 2 or a.shape[0] == 0:
            self.pts = np.zeros((0, 2))
        else:
            self.pts = a
        self._tree = None

    def __len__(self):
        return self.pts.shape[0]

    def dist(self, targets):
        n = targets.shape[0]
        if self.pts.shape[0] == 0 or n == 0:
            return np.full(n, 1e9)
        try:
            from scipy.spatial import cKDTree
            if self._tree is None:
                self._tree = cKDTree(self.pts)
            d, _ = self._tree.query(targets, k=1)
            return np.asarray(d, dtype=float)
        except Exception:
            out = np.empty(n)
            for i in range(n):
                out[i] = np.sqrt(((self.pts - targets[i]) ** 2).sum(axis=1).min())
            return out


def _pca_ends(pts):
    """连通块的最长轴两端点（pts 已是 (n,2)）。"""
    if pts.shape[0] < 2:
        return []
    c = pts.mean(axis=0)
    q = pts - c
    try:
        u, s, vt = np.linalg.svd(q, full_matrices=False)
        d = vt[0]
    except Exception:
        return [tuple(pts[0]), tuple(pts[-1])]
    t = q @ d
    return [tuple(pts[int(np.argmin(t))]), tuple(pts[int(np.argmax(t))])]


# ------------------------------------------------------------------ features

def _img_features(bgr, H, bboxes):
    """(b) (c) (d2) 三类图像特征，返回 doc 坐标点列。"""
    from scipy import ndimage
    b = bgr[:, :, 0].astype(np.int16)
    g = bgr[:, :, 1].astype(np.int16)
    r = bgr[:, :, 2].astype(np.int16)

    # 高饱和红 = 结点圆点（暗红引线 R 较低、绿导线 G 高）
    dot_red = (r > 140) & (r > g + 70) & (r > b + 60)
    maroon = (r > 60) & (r > g + 30) & (r > b + 20) & (g < 110)
    green = (g > r + 25) & (g > b + 20)

    dots = []
    if dot_red.sum() > 0:
        dt = ndimage.distance_transform_edt(dot_red)
        core = dt >= 2.0
        lab, n = ndimage.label(core)
        if n:
            sums = ndimage.sum(dot_red, lab, index=np.arange(1, n + 1))
            coms = ndimage.center_of_mass(dot_red, lab, index=np.arange(1, n + 1))
            for i in range(n):
                if sums[i] >= 15:
                    yy, xx = coms[i]
                    dots.append((float(xx), float(H - yy)))

    # 色转折点：绿像素的 5x5 邻域里有 maroon
    tr = []
    if green.sum() > 0 and maroon.sum() > 0:
        md = ndimage.binary_dilation(maroon, iterations=2)
        meet = green & md
        if meet.sum() > 0:
            lab, n = ndimage.label(meet)
            if n:
                coms = ndimage.center_of_mass(meet, lab, index=np.arange(1, n + 1))
                tr = [(float(xx), float(H - yy)) for yy, xx in coms]

    # 元件外伸引线段的自由端：bbox 外的 maroon 连通块，取主轴两端
    tips = []
    if maroon.sum() > 0:
        inside = np.zeros(maroon.shape, dtype=bool)
        for (x0, y0, x1, y1) in bboxes:
            ia = int(max(0, math.floor(H - max(y0, y1)) - 2))
            ib = int(min(H - 1, math.ceil(H - min(y0, y1)) + 2))
            ja = int(max(0, math.floor(min(x0, x1)) - 2))
            jb = int(min(maroon.shape[1] - 1, math.ceil(max(x0, x1)) + 2))
            if ib >= ia and jb >= ja:
                inside[ia:ib + 1, ja:jb + 1] = True
        out = maroon & ~inside
        lab, n = ndimage.label(out)
        if n:
            objs = ndimage.find_objects(lab)
            for i, sl in enumerate(objs):
                if sl is None:
                    continue
                m = (lab[sl] == (i + 1))
                area = int(m.sum())
                if area < 12 or area > 4000:
                    continue
                ys, xs = np.nonzero(m)
                pts = np.stack([xs + sl[1].start, ys + sl[0].start], axis=1).astype(float)
                for (xx, yy) in _pca_ends(pts):
                    tips.append((float(xx), float(H - yy)))
    return dots, tr, tips


def _corner_features(res, ang_deg=25.0):
    out = []
    ca = math.cos(math.radians(180.0 - ang_deg))
    for poly in res.polylines:
        if len(poly) < 3:
            continue
        for i in range(1, len(poly) - 1):
            ax, ay = poly[i - 1]
            bx, by = poly[i]
            cx, cy = poly[i + 1]
            v1 = (bx - ax, by - ay)
            v2 = (cx - bx, cy - by)
            n1 = math.hypot(*v1)
            n2 = math.hypot(*v2)
            if n1 < 1e-6 or n2 < 1e-6:
                continue
            c = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
            if c <= ca:
                out.append((float(bx), float(by)))
    return out


def _tee_features(edges, near=2.0, far=40.0):
    """我方边的轴交点型 T 点（不依赖墨宽）。"""
    out = []
    n = len(edges)
    dirs = []
    for (p0, p1) in edges:
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        L = math.hypot(dx, dy)
        dirs.append((dx / L, dy / L, L) if L > 1e-6 else (0.0, 0.0, 0.0))
    for i in range(n):
        a0, a1 = edges[i]
        ux, uy, La = dirs[i]
        if La <= 0:
            continue
        for j in range(n):
            if i == j:
                continue
            b0, b1 = edges[j]
            vx, vy, Lb = dirs[j]
            if Lb <= 0:
                continue
            cross = ux * vy - uy * vx
            if abs(cross) < math.sin(math.radians(30.0)):
                continue            # 近共线，不算 T
            wx, wy = b0[0] - a0[0], b0[1] - a0[1]
            t = (wx * vy - wy * vx) / cross          # 沿 A 的参数（px）
            s = (wx * uy - wy * ux) / (-cross)       # 沿 B 的参数（px）
            if not (-1.0 <= s <= Lb + 1.0):
                continue                              # 交点必须在 B 的跨度内
            d0 = abs(t)
            d1 = abs(t - La)
            dmin = min(d0, d1)
            if not (near < dmin <= far):
                continue
            out.append((float(a0[0] + t * ux), float(a0[1] + t * uy)))
    return out


def _bbox_outline(bboxes):
    out = []
    for (x0, y0, x1, y1) in bboxes:
        xa, xb = min(x0, x1), max(x0, x1)
        ya, yb = min(y0, y1), max(y0, y1)
        for x in np.arange(xa, xb + 1.0, 1.0):
            out.append((float(x), float(ya)))
            out.append((float(x), float(yb)))
        for y in np.arange(ya, yb + 1.0, 1.0):
            out.append((float(xa), float(y)))
            out.append((float(xb), float(y)))
    return out


# ------------------------------------------------------------------ main

def census(cases, run_dir, data_root, out_path=None, tag_extra=""):
    import pipeline.pcore as pcore
    import pipeline.pskel as pskel
    import pipeline.prun as prun

    index = {}
    for c in discover(data_root):
        index.setdefault(str(c.case_id), c)

    feats = ["node_raw", "node_solved", "dot", "colour_tr", "bbox", "lead_tip",
             "pin_our", "corner", "tee", "gt_pin", "gt_end"]
    # 累计「目标点 -> 各特征距离」
    acc = {"gt": {f: [] for f in feats}, "our": {f: [] for f in feats}}
    per_case = []
    extra_stash = []

    for cid in cases:
        case = index.get(str(cid))
        if case is None:
            print("!! case %s not found" % cid)
            continue
        pred_path = os.path.join(run_dir, "final_results", "%s_Pred.json" % cid)
        if not os.path.exists(pred_path):
            print("!! missing pred %s" % pred_path)
            continue
        gt_doc = json.load(io.open(case.json_path, encoding="utf-8"))
        pred_doc = json.load(io.open(pred_path, encoding="utf-8"))
        pc = pcore.find_case(str(cid))
        inp = prun.inputs_of(pc)
        H = inp.h
        bgr = inp.bgr

        old = os.environ.get("PSKEL_NODE_SOLVE")
        os.environ["PSKEL_NODE_SOLVE"] = "0"
        ex0 = pskel.WireExtractor(use_colour=False, close_ink=True, snap=False,
                                  ink_mode="adaptive")
        res0 = ex0.extract(bgr)
        os.environ["PSKEL_NODE_SOLVE"] = "1"
        ex1 = pskel.WireExtractor(use_colour=False, close_ink=True, snap=False,
                                  ink_mode="adaptive")
        res1 = ex1.extract(bgr)
        if old is None:
            os.environ.pop("PSKEL_NODE_SOLVE", None)
        else:
            os.environ["PSKEL_NODE_SOLVE"] = old

        def to_doc(p):
            return (float(p[0]), float(H - p[1]))

        raw_nodes = [to_doc(p) for p in res0.junctions]
        # 段端点（重解后的）
        e1 = []
        for (p0, p1) in res1.segments:
            e1.append((float(p0[0]), float(H - p0[1])))
            e1.append((float(p1[0]), float(H - p1[1])))
        E1 = np.asarray(e1, dtype=float) if e1 else np.zeros((0, 2))
        solved = []
        for j in raw_nodes:
            if E1.shape[0] == 0:
                solved.append(j)
                continue
            d = np.sqrt(((E1 - np.asarray(j, dtype=float)) ** 2).sum(axis=1))
            k = int(np.argmin(d))
            solved.append((float(E1[k, 0]), float(E1[k, 1])) if d[k] <= 20.0 else j)

        bboxes = _bboxes_of(pred_doc)
        dots, tr, tips = _img_features(bgr, H, bboxes)
        corners = [to_doc(p) for p in _corner_features(res1)]
        tees = _tee_features(_edges_of(pred_doc))

        clouds = {
            "node_raw": Cloud(raw_nodes, "node_raw"),
            "node_solved": Cloud(solved, "node_solved"),
            "dot": Cloud(dots, "dot"),
            "colour_tr": Cloud(tr, "colour_tr"),
            "bbox": Cloud(_bbox_outline(bboxes), "bbox"),
            "lead_tip": Cloud(tips, "lead_tip"),
            "pin_our": Cloud(_pins_of(pred_doc), "pin_our"),
            "corner": Cloud(corners, "corner"),
            "tee": Cloud(tees, "tee"),
            "gt_pin": Cloud(_pins_of(gt_doc), "gt_pin"),
            "gt_end": Cloud([(x, y) for (x, y) in _ends_of(gt_doc)], "gt_end"),
        }

        gt_ends = np.asarray(_ends_of(gt_doc), dtype=float)
        our_ends = np.asarray(_ends_of(pred_doc), dtype=float)
        row = dict(case=str(cid), n_gt=int(gt_ends.shape[0]), n_our=int(our_ends.shape[0]),
                   n_feat={f: len(clouds[f]) for f in feats})
        for f in feats:
            dg = clouds[f].dist(gt_ends)
            do = clouds[f].dist(our_ends)
            acc["gt"][f].append(dg)
            acc["our"][f].append(do)
            row[f] = dict(
                gt=[int((dg <= t).sum()) for t in (3.0, 5.0, 10.0)],
                our=[int((do <= t).sum()) for t in (3.0, 5.0, 10.0)])
        per_case.append(row)
        extra_stash.append(dict(case=str(cid), node_raw=raw_nodes, node_solved=solved,
                                dot=dots, colour_tr=tr, lead_tip=tips,
                                bbox=_bbox_outline(bboxes), pin_our=_pins_of(pred_doc),
                                corner=corners, tee=tees,
                                gt_ends=[list(p) for p in gt_ends],
                                our_ends=[list(p) for p in our_ends]))

    # ---- 汇总
    summary = {}
    for side in ("gt", "our"):
        summary[side] = {}
        for f in feats:
            if not acc[side][f]:
                continue
            d = np.concatenate(acc[side][f])
            n = d.shape[0]
            summary[side][f] = dict(n=n,
                                    le3=int((d <= 3.0).sum()),
                                    le5=int((d <= 5.0).sum()),
                                    le10=int((d <= 10.0).sum()),
                                    r3=float((d <= 3.0).mean()),
                                    r5=float((d <= 5.0).mean()),
                                    r10=float((d <= 10.0).mean()),
                                    med=float(np.median(d)))

    # ---- 缺口清单：GT 端被某特征 <=5px 覆盖，但我们 3px 内没有任何端点
    union_keys = [f for f in feats if f not in ("gt_end", "gt_pin")]
    gap = {}
    union_gap_ends = 0
    for ci, cid in enumerate([r["case"] for r in per_case]):
        st = extra_stash[ci]
        if st["case"] != cid:
            continue
        gt_ends = np.asarray(st["gt_ends"], dtype=float)
        our_ends = np.asarray(st["our_ends"], dtype=float)
        # 每个 GT 端到「我方最近端点」的距离（与特征无关）
        d_near_our = Cloud(our_ends, "our_ends").dist(gt_ends)
        covered_any = np.zeros(gt_ends.shape[0], dtype=bool)
        for f in union_keys:
            pts = st[f] if f in st else None
            if pts is None:
                continue
            cl = Cloud(pts, f)
            dg = cl.dist(gt_ends)
            # 该 GT 端被 f 覆盖（这里用 5px），但我们那边没有端点落在 3px 内
            hit = (dg <= 5.0) & (d_near_our > 3.0)
            if hit.any():
                gap.setdefault(f, []).append((cid, int(hit.sum()), int((dg <= 5.0).sum())))
            covered_any |= (dg <= 5.0)
        union_gap_ends += int((~covered_any).sum())

    print("=" * 108)
    print("切点普查：列=候选锚点（全部不看 GT 算出），行=GT 端 / 我方端 覆盖率")
    print("=" * 108)
    print("%-12s %8s | %7s %7s %7s | %7s %7s %7s" %
          ("feature", "n_pts", "GT<=3", "GT<=5", "GT<=10", "OUR<=3", "OUR<=5", "OUR<=10"))
    print("-" * 108)
    ngt = sum(r["n_gt"] for r in per_case)
    nour = sum(r["n_our"] for r in per_case)
    for f in feats:
        npts = sum(r["n_feat"].get(f, 0) for r in per_case)
        gt_s = summary["gt"].get(f, {})
        our_s = summary["our"].get(f, {})
        mark = " *" if f in ("gt_pin", "gt_end") else ""
        print("%-12s %8d | %6.3f %6.3f %6.3f | %6.3f %6.3f %6.3f%s" %
              (f, npts,
               gt_s.get("r3", 0), gt_s.get("r5", 0), gt_s.get("r10", 0),
               our_s.get("r3", 0), our_s.get("r5", 0), our_s.get("r10", 0), mark))
    print("-" * 108)
    print("GT 端总数 %d / 我方端总数 %d（* = 用到 GT 的上界参考，不算可用杠杆）" % (ngt, nour))

    print("\n逐案（GT<=5px 覆盖率）：")
    hdr = "case  nGT  nOur | " + " ".join("%-11s" % f[:11] for f in feats)
    print(hdr)
    for r in per_case:
        line = "%-5s %4d %5d | " % (r["case"], r["n_gt"], r["n_our"])
        for f in feats:
            v = r[f]["gt"]
            line += "%-11s " % ("%d/%d/%d" % tuple(v))
        print(line)

    print("\n" + "=" * 108)
    print("缺口清单：GT 端被特征 F 以 <=5px 覆盖、但我方端点 3px 内空着（可重划的候选切点）")
    print("=" * 108)
    rows = []
    for f, lst in gap.items():
        tot = sum(x[1] for x in lst)
        cand = sum(x[2] for x in lst)
        rows.append((tot, cand, f, lst))
    for tot, cand, f, lst in sorted(rows, reverse=True):
        print("%-12s 缺口 %4d 条 / 该特征覆盖 GT 端 %4d 条  逐案 %s" %
              (f, tot, cand, ", ".join("%s:%d/%d" % x for x in lst)))
    print("=> 任何候选特征都没能在 5px 内解释的 GT 端：%d 条（占 %d 的 %.1f%%）"
          % (union_gap_ends, ngt, 100.0 * union_gap_ends / max(1, ngt)))

    # ---- 联合可达上界：把若干特征并起来当候选切点集合，看能覆盖多少 GT 端 / 多少 GT 边两端
    combos = [
        ("node_raw", ["node_raw"]),
        ("node+tee", ["node_raw", "tee"]),
        ("node+tee+lead", ["node_raw", "tee", "lead_tip"]),
        ("node+tee+dot+lead", ["node_raw", "tee", "dot", "lead_tip"]),
        ("+colour_tr", ["node_raw", "tee", "dot", "lead_tip", "colour_tr"]),
        ("ALL9", union_keys),
    ]
    joint = {}
    print("\n" + "=" * 108)
    print("联合可达上界（并集当候选切点；「边两端<=5px」= 该 GT 边理论上可被切出来配上的条数）")
    print("=" * 108)
    ne_tot = 0
    for st in extra_stash:
        ne_tot += len(st["gt_ends"]) // 2
    for name, fs in combos:
        n_end5 = n_end3 = n_all = 0
        n_edge = 0
        for st in extra_stash:
            pts = []
            for f in fs:
                pts.extend(st.get(f) or [])
            cl = Cloud(pts, name)
            gt = np.asarray(st["gt_ends"], dtype=float)
            dg = cl.dist(gt)
            n_end5 += int((dg <= 5.0).sum())
            n_end3 += int((dg <= 3.0).sum())
            n_all += dg.shape[0]
            for i in range(0, dg.shape[0] - 1, 2):
                if dg[i] <= 5.0 and dg[i + 1] <= 5.0:
                    n_edge += 1
        joint[name] = dict(end5=n_end5, end3=n_end3, n=n_all, edge5=n_edge, ne=ne_tot)
        print("%-18s GT端<=5 %6.3f  GT端<=3 %6.3f | 边两端<=5 %4d / %4d = %.3f"
              % (name, n_end5 / max(1, n_all), n_end3 / max(1, n_all),
                 n_edge, ne_tot, n_edge / max(1, ne_tot)))
    print("（对照）我方端点自身对 GT 端的覆盖见上表 gt_end 行；line_tol=5.0")

    if out_path:
        doc = dict(summary=summary, per_case=per_case,
                   gap={k: v for k, v in gap.items()}, joint=joint,
                   union_gap_ends=union_gap_ends, n_gt=ngt, n_our=nour,
                   feats=feats, stash=extra_stash)
        io.open(out_path, "w", encoding="utf-8").write(
            json.dumps(doc, ensure_ascii=False))
        print("\nwrote %s" % out_path)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--run-dir", default=os.environ.get("NLCENSUS_RUN", DEFAULT_RUN))
    ap.add_argument("--data", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    cases = [x.strip() for x in a.cases.split(",") if x.strip()]
    census(cases, a.run_dir, a.data, a.out or None)


if __name__ == "__main__":
    main()
