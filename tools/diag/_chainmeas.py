# -*- coding: utf-8 -*-
"""`_chainmeas.py` —— 「导线 ↔ 元件 ↔ 引脚」链式单点失效的**量化**（用户 m42009 第二问）。

审计结论（子代理 ca575e9a）：**不存在硬环**，是单向流水线（`stage1/detect.py` /
`stage1/bigic.py` 对 ink/wire/segment 零引用，元件先定稿、之后没有任何代码用
wire/pin 回改元件）。但存在**同现象的链式单点失效**：

    导线短桩要「贴已检出元件框」才被豁免（`stage1/wires.py:534`）
      → 元件框被漏检 ⇒ 桩删（只有计数器 `n_comp_dropped`，无账本）
      → 该元件没有导线触点 ⇒ `stage1/pins.py:251/:279` `return []`（无账本）
      → `stage1/nets.py:242-243` 引脚数不足的网 `continue`（无账本）
      → `stage2/assemble.py:792-794` 只丢引脚、元件保留（观感：元件孤岛 + 网消失）

本探针量两跳的**上限**：
  1) 基线：最终 `segments` 覆盖多少 GT 导线边；
  2) 把 `stubs_latent`（被删短桩的保留池）并回去后多覆盖多少 GT 边
     ⇒ **第一跳能被救回的收益上限**（"用别的证据救桩"的天花板）；
  3) 被删桩的 `dbox` 分布（离最近框多远被删 ⇒ `rescue_pad` 要多大才豁免得到）；
  4) 第二跳：stage1 里 `pins` 为空、而 GT 导线端点落在该元件框内的元件数
     ⇒ **无触点元件的损失量**。

用法：
    python work\\tools\\dev\\_chainmeas.py --cases-file work\\_cases_w28.txt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"Y:\workspace\EDA")
EDA = ROOT / "eda-pcb"
for p in (str(EDA), str(EDA / "tools" / "eval"), str(ROOT / "work" / "tools" / "dev")):
    if p not in sys.path:
        sys.path.insert(0, p)

import cv2  # noqa: E402
from schematic_parser.data import discover, DEFAULT_DATA_ROOT  # noqa: E402
import stage1_wires_eval as SE  # noqa: E402
import _wireedge as WE  # noqa: E402
from stage1 import wires as W  # noqa: E402

# 本探针量的是**默认口径**，把 A/B 用的开关全部清掉
for _k in ("STAGE1_WIRE_EDGE", "STAGE1_WIRE_EDGE_NEAR", "STAGE1_WIRE_EDGE_REL",
           "STAGE1_WIRE_EDGE_MIN", "STAGE1_WIRE_INK", "STAGE1_WIRE_UNCONF",
           "STAGE1_WIRE_UNCONF_NEAR", "STAGE1_OCC_LOCAL"):
    os.environ.pop(_k, None)

BOX_BUCKETS = [(0, 2), (2, 5), (5, 10), (10, 20), (20, 50), (50, 1e9)]


def gt_edge_points(case, H: float):
    """GT 导线边的端点（翻成 PNG 帧）。"""
    gt = json.load(open(case.json_path, encoding="utf-8"))
    pts = []
    for net in gt["nets"].values():
        for e in (net.get("edges") or {}).values():
            for q in e:
                pts.append((float(q["x"]), H - float(q["y"])))
    return pts


def inside(px, py, bx, pad=6.0):
    x1, y1, x2, y2 = bx
    return (x1 - pad) <= px <= (x2 + pad) and (y1 - pad) <= py <= (y2 + pad)


def edge_cover(GA, PA, tol: float):
    """逐条 GT 边的覆盖布尔（判据与 `stage1_wires_eval` 一致：16 个采样点全在 tol 内）。"""
    n_g = GA.shape[0]
    if not n_g:
        return np.zeros(0, bool)
    pts, of = [], []
    for i in range(n_g):
        pts.append(SE.sample_polyline(GA[i, :2], GA[i, 2:], 16))
        of.append(np.full(16, i))
    if not PA.shape[0]:
        return np.zeros(n_g, bool)
    d = SE.dist_pts_to_segs(np.concatenate(pts, 0), PA)
    worst = np.full(n_g, 0.0)
    np.maximum.at(worst, np.concatenate(of), d)
    return worst <= tol


def edge_samples(GA, n: int = 16):
    """返回 (采样点数组 Nx2, 所属边索引)。"""
    pts, of = [], []
    for i in range(GA.shape[0]):
        pts.append(SE.sample_polyline(GA[i, :2], GA[i, 2:], n))
        of.append(np.full(n, i))
    if not pts:
        return np.zeros((0, 2)), np.zeros(0, int)
    return np.concatenate(pts, 0), np.concatenate(of)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases-file", default=str(ROOT / "work" / "_cases_w28.txt"))
    ap.add_argument("--stage1-dir", default=str(ROOT / "work" / "_full220" / "runs" / "stage1"))
    ap.add_argument("--png-dir", default=str(ROOT / "work" / "_submit_in220"))
    ap.add_argument("--tol", type=float, default=5.0)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    want = [s.strip() for s in Path(a.cases_file).read_text(encoding="utf-8-sig").split(",") if s.strip()]
    if a.limit:
        want = want[:a.limit]
    cases = {c.case_id: c for c in discover(DEFAULT_DATA_ROOT)}

    tot = {"n_g": 0, "cov_base": 0, "cov_lat": 0, "n_p": 0, "cov_p_base": 0,
           "cov_p_lat": 0, "n_lat": 0, "n_isl": 0, "n_isl_gtp": 0, "n_comp": 0,
           "n_in": 0, "cov_in": 0, "n_out": 0, "cov_out": 0}
    buckets = [0] * len(BOX_BUCKETS)
    rows = []
    for cid in want:
        s1p = Path(a.stage1_dir) / ("%s.stage1.json" % cid)
        png = Path(a.png_dir) / ("%s.png" % cid)
        case = cases.get(cid)
        if not s1p.exists() or not png.exists() or case is None:
            print("  跳过 %s" % cid)
            continue
        d = json.load(open(s1p, encoding="utf-8"))
        H = float(d["image"]["h"])
        bgr = WE.imread_unicode(png)
        comps = [c["bbox_img"] for c in d.get("components", [])]
        texts = [t["bbox_img"] for t in d.get("texts", [])]
        w, _t = W.extract_wires(bgr, occlude=comps, erase=texts, erase_pad=1.0,
                                drop_frame=True, max_axis_dev=12.0, min_extent=80.0,
                                rescue_pad=8.0)
        GA = SE.seg_arrays(WE.gt_segs(case, H))
        PA = SE.seg_arrays(w["segments"])
        cg0, ng, cp0, npd0 = WE.score(GA, PA, a.tol)
        lat = w.get("stubs_latent") or []
        PA2 = SE.seg_arrays(list(w["segments"]) + [[s["seg"][0], s["seg"][1]] for s in lat])
        cg1, _n2, cp1, npd1 = WE.score(GA, PA2, a.tol)

        # 第二跳：stage1 一条引脚都没有的元件，GT 导线端点是否落在它的框里
        pins = d.get("pins") or {}
        gpts = gt_edge_points(case, H)
        n_isl = n_isl_gt = 0
        for c in d.get("components", []):
            sid = c.get("id")
            if pins.get(sid):
                continue
            n_isl += 1
            bx = c.get("bbox_img")
            if bx and any(inside(px, py, bx) for px, py in gpts):
                n_isl_gt += 1

        # 第三跳（真正的互相等待，见文件头）：GT 导线边**穿过已检出元件框**的比例 ——
        # `occlude_boxes` 在提线前把框涂白，所以落在框内的导线证据 100% 注定丢失；
        # 而框本身又靠这些导线来确认引脚。这一步量「被自己的框删掉的导线天花板」。
        epts, eof = edge_samples(GA, 16)
        cov = edge_cover(GA, PA, a.tol)
        in_box = np.zeros(GA.shape[0], bool)
        for k in range(epts.shape[0]):
            px, py = float(epts[k][0]), float(epts[k][1])
            for bx in comps:
                if inside(px, py, bx, pad=2.0):
                    in_box[int(eof[k])] = True
                    break
        n_in = int(in_box.sum())
        n_out = int((~in_box).sum())
        tot["n_in"] += n_in
        tot["cov_in"] += int(cov[in_box].sum()) if n_in else 0
        tot["n_out"] += n_out
        tot["cov_out"] += int(cov[~in_box].sum()) if n_out else 0

        rows.append((cid, cg0, ng, cg1, len(lat), npd0, npd1, n_isl, n_isl_gt, n_in,
                     int(cov[in_box].sum()) if n_in else 0))
        tot["n_g"] += ng
        tot["cov_base"] += cg0
        tot["cov_lat"] += cg1
        tot["n_p"] += npd0
        tot["cov_p_base"] += cp0
        tot["cov_p_lat"] += cp1
        tot["n_lat"] += len(lat)
        tot["n_isl"] += n_isl
        tot["n_isl_gtp"] += n_isl_gt
        tot["n_comp"] += len(d.get("components", []))
        for s in lat:
            for i, (lo, hi) in enumerate(BOX_BUCKETS):
                if lo <= s["dbox"] < hi:
                    buckets[i] += 1
                    break

    print("\n=== 逐张：GT 边覆盖(基线→并回保留池) | 保留池桩数 | 预测段数 | 无引脚元件(其中GT有导线端点) ===")
    for r in rows:
        print("  %-14s %4d/%4d → %4d/%4d | lat %3d | segs %5d→%5d | isl %2d(%d)"
              % (r[0], r[1], r[2], r[3], r[2], r[4], r[5], r[6], r[7], r[8]))
    print("\n=== 合计 ===")
    R0 = tot["cov_base"] / max(1, tot["n_g"])
    R1 = tot["cov_lat"] / max(1, tot["n_g"])
    P0 = tot["cov_p_base"] / max(1, tot["n_p"])
    P1 = tot["cov_p_lat"] / max(1, max(1, tot["n_p"]) + 0)
    print("  GT 边：基线覆盖 %d/%d R=%.4f  →  并回保留池 %d/%d R=%.4f  (ΔR=%+.4f, 多 %d 条)"
          % (tot["cov_base"], tot["n_g"], R0, tot["cov_lat"], tot["n_g"], R1, R1 - R0,
             tot["cov_lat"] - tot["cov_base"]))
    print("  预测段精度：基线 %d/%d P=%.4f" % (tot["cov_p_base"], tot["n_p"], P0))
    print("  保留池桩数 %d（%.1f/张）" % (tot["n_lat"], tot["n_lat"] / max(1, len(rows))))
    print("  被删桩的 dbox 分布（离最近框多远）：" +
          "  ".join("%g-%g:%d" % (lo, hi if hi < 1e8 else 999, n)
                    for (lo, hi), n in zip(BOX_BUCKETS, buckets)))
    print("  第二跳：无引脚元件 %d/%d 颗，其中 GT 导线端点落在其框内的 **%d 颗**"
          % (tot["n_isl"], tot["n_comp"], tot["n_isl_gtp"]))
    print("  第三跳（真正的互相等待）：GT 导线边**穿过已检出元件框**的 %d/%d 条，被覆盖 %d 条"
          "；框外的 %d/%d 条，被覆盖 %d 条"
          % (tot["n_in"], tot["n_in"] + tot["n_out"], tot["cov_in"],
             tot["n_out"], tot["n_in"] + tot["n_out"], tot["cov_out"]))
    if tot["n_in"]:
        print("    ⇒ 框内 R=%.4f vs 框外 R=%.4f（框内导线在提线前就被涂掉，注定拿不到）"
              % (tot["cov_in"] / tot["n_in"], tot["cov_out"] / max(1, tot["n_out"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
