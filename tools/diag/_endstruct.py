# -*- coding: utf-8 -*-
"""导线端点可行性（只读）：GT 的导线端点**钉在什么结构上**，我们的图像证据够不够。

背景：§5.90 把导线列收口成「端点纵向长度错」，§5.104 量出端点神谕 +1.28~+1.30、
§5.112 又否掉三种「统一把端点往某个锚点挪」的规则（`work/_endtrue.py`：160 条 GT 端点里
48 在 bbox 边框、22 在 GT 脚点、**90 处两者都不是**）。
⇒ 剩下的问题不是「规则怎么写」，而是**目标位置上到底有没有可观测的结构**。
本探针把每个 GT 导线端点对下列结构量最近距离，并做**首次命中分类**：

| 结构 | 说法 | 来源 |
| --- | --- | --- |
| `pin` | 落在某个 GT 脚点上（我们就照 GT 的引脚定义看） | `inp.truth.pins` |
| `bbox` | 落在某个元件框的边框上 | `inp.truth.comps[*].box` |
| `wire` | 落在**别的**导线上（T 形搭接） | GT `nets[*].edges` |
| `skel` | 落在图像骨架的**端点**上 | `prun.skeleton_of(...).endpoints` |
| `junc` | 落在图像骨架的**结点**上 | 同上 `.junctions` |
| `none` | 以上都没有（⇒ 该端点只能靠「把线画到哪就是哪」） | — |

同时量：GT 端点**是否落在我们骨架的像素上**（`d_skel_px`）—— 这是「图像里有没有这根线」
的硬证据；以及我们的 pred 端点离 GT 端点多远（`d_pred`）。

用法：
    python -X utf8 work\\_endstruct.py                       # 官方 10 张，交付产物 sjndef
    python -X utf8 work\\_endstruct.py --run gsd9 --tol 5
    python -X utf8 work\\_endstruct.py --cases 1,8 --detail  # 打逐端点明细
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]                       # Y:\workspace\EDA
EDA = ROOT / "eda-pcb"
sys.path.insert(0, str(EDA))
sys.path.insert(0, str(EDA / "pipeline"))
sys.path.insert(0, str(EDA / "tools" / "eval"))

from schematic_parser.data import DEFAULT_DATA_ROOT, load_json            # noqa: E402
from schematic_parser.metrics import _edge_points, normalise_doc          # noqa: E402
from stage2_eval import build_index, resolve                              # noqa: E402

from pipeline import pcore, prun  # noqa: E402  (eda-pcb/pipeline/{pcore,prun}.py)

sys.stdout.reconfigure(encoding="utf-8")
OFFICIAL = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]


def _pts(arr):
    return np.asarray(arr, dtype=float).reshape(-1, 2)


def d_point_to_segments(p, segs):
    """点到线段集合的最短距离（segs: list[(x0,y0,x1,y1)]）。"""
    if not segs:
        return float("inf")
    a = np.asarray([[s[0], s[1]] for s in segs], dtype=float)
    b = np.asarray([[s[2], s[3]] for s in segs], dtype=float)
    ab = b - a
    L2 = (ab ** 2).sum(axis=1)
    L2[L2 <= 1e-12] = 1e-12
    t = ((p - a) * ab).sum(axis=1) / L2
    t = np.clip(t, 0.0, 1.0)
    proj = a + ab * t[:, None]
    return float(np.hypot(*(proj - p).T).min())


def d_point_to_points(p, arr):
    if arr is None or len(arr) == 0:
        return float("inf")
    return float(np.hypot(*(arr - p).T).min())


def rect_segments(box):
    x0, y0, x1, y1 = box
    return [(x0, y0, x1, y0), (x1, y0, x1, y1), (x1, y1, x0, y1), (x0, y1, x0, y0)]


def gt_wire_endpoints(gt_doc, h):
    """GT 导线端点（图像坐标）。GT 文档 y 向上 ⇒ y_img = h - y。"""
    doc = normalise_doc(gt_doc)
    out = []
    for nkey, net in (doc.get("nets") or {}).items():
        for ekey, edge in (net.get("edges") or {}).items():
            pts = _edge_points(edge)
            if not pts or len(pts) < 2:
                continue
            for which, p in (("a", pts[0]), ("b", pts[-1])):
                out.append({"net": nkey, "edge": ekey, "which": which,
                            "p": np.array([float(p[0]), h - float(p[1])]),
                            "poly": [(float(q[0]), h - float(q[1])) for q in pts]})
    return out


def pred_wire_endpoints(pred_doc, h):
    doc = normalise_doc(pred_doc)
    out = []
    for net in (doc.get("nets") or {}).values():
        for edge in (net.get("edges") or {}).values():
            pts = _edge_points(edge)
            if not pts or len(pts) < 2:
                continue
            for p in (pts[0], pts[-1]):
                out.append(np.array([float(p[0]), h - float(p[1])]))
    return np.asarray(out, dtype=float) if out else np.zeros((0, 2))


def pred_path(outdir, cid):
    """交付产物的文件名：`<cid>_Pred.json`，再退到没有前导零/带下划线的写法。"""
    names = [cid + "_Pred.json", cid + ".json",
             (cid.lstrip("0") or "0") + "_Pred.json", (cid.lstrip("0") or "0") + ".json"]
    for nm in names:
        p = outdir / "final_results" / nm
        if p.exists():
            return p
    return None


def gt_edges(gt_doc, h):
    """GT 的导线折线（图像坐标）。"""
    doc = normalise_doc(gt_doc)
    out = []
    for nkey, net in (doc.get("nets") or {}).items():
        for ekey, edge in (net.get("edges") or {}).items():
            pts = _edge_points(edge)
            if not pts or len(pts) < 2:
                continue
            out.append({"net": nkey, "edge": ekey,
                        "poly": [(float(q[0]), h - float(q[1])) for q in pts]})
    return out


def seg_match(gt_doc, pred_doc, h, tol):
    """复刻判分器的线段匹配：每条 edge 只比**前两点**，unordered。

    判分器 `net_line_f1` 里是 `_seg_distance(pe[:2], ge[:2], cfg.line_unordered)`：
    只有折线的头两点参与比较。这里据此给出三分类：
      `tp`            —— 前两点在 tol 内配上（判分器算对）；
      `cov_not_match` —— **没配上，但整条 GT 折线被我们的折线像素覆盖**（⇒ 分割/端点问题，不是没画）；
      `not_cov`       —— 覆盖也差（⇒ 真的没画到那里）。
    """
    g = gt_edges(gt_doc, h)
    if pred_doc is None:
        return None
    doc = normalise_doc(pred_doc)
    p = []
    for net in (doc.get("nets") or {}).values():
        for edge in (net.get("edges") or {}).values():
            pts = _edge_points(edge)
            if pts and len(pts) >= 2:
                p.append([(float(q[0]), h - float(q[1])) for q in pts])
    if not p:
        return {"gt": len(g), "tp": 0, "cov_not_match": 0, "not_cov": len(g),
                "pts_per_edge": 0.0}
    pred_segs = [(pl[0], pl[1]) for pl in p if len(pl) >= 2]
    pred_all = []
    for pl in p:
        for i in range(len(pl) - 1):
            pred_all.append((pl[i][0], pl[i][1], pl[i + 1][0], pl[i + 1][1]))
    tp = cov = bad = 0
    for e in g:
        a, b = e["poly"][0], e["poly"][1]
        ok = False
        for c, d in pred_segs:
            for (u, v) in ((c, d), (d, c)):
                if (max(abs(a[0] - u[0]), abs(a[1] - u[1])) <= tol
                        and max(abs(b[0] - v[0]), abs(b[1] - v[1])) <= tol):
                    ok = True
                    break
            if ok:
                break
        if ok:
            tp += 1
            continue
        # 覆盖检查：沿 GT 折线采样，看是否落在我们折线的 2px 内
        pl = np.asarray(e["poly"], dtype=float)
        samples = []
        for i in range(len(pl) - 1):
            seg = pl[i + 1] - pl[i]
            L = float(np.hypot(*seg))
            if L <= 1e-9:
                continue
            k = max(2, int(L / 3.0) + 1)
            for t in np.linspace(0.0, 1.0, k):
                samples.append(pl[i] + seg * t)
        if samples and pred_all:
            arr = np.asarray(samples, dtype=float)
            hit = 0
            for s in arr:
                if d_point_to_segments(s, pred_all) <= 2.0:
                    hit += 1
            frac = hit / float(len(arr))
        else:
            frac = 0.0
        if frac >= 0.9:
            cov += 1
        else:
            bad += 1
    npts = float(np.mean([len(e["poly"]) for e in g])) if g else 0.0
    return {"gt": len(g), "tp": tp, "cov_not_match": cov, "not_cov": bad,
            "pts_per_edge": npts}


def case_of(cid):
    case = pcore.find_case(cid)
    if case is not None:
        return case
    exact, digits = build_index(DEFAULT_DATA_ROOT)
    dcase, _ = resolve(cid, exact, digits)
    if dcase is None:
        dcase, _ = resolve(cid.lstrip("0") or "0", exact, digits)
    if dcase is None:
        return None
    return pcore.find_case(dcase.case_id) or pcore.find_case(cid.lstrip("0") or "0")


def main(argv=None):
    ap = argparse.ArgumentParser(description="导线端点结构归属（只读）")
    ap.add_argument("--cases", default=",".join(OFFICIAL))
    ap.add_argument("--run", default="sjndef", help="产物后缀 work/_fast_out_<run>")
    ap.add_argument("--tol", type=float, default=5.0)
    ap.add_argument("--skel-tol", type=float, default=2.0, help="判定「端点在骨架像素上」的容差")
    ap.add_argument("--detail", action="store_true")
    ap.add_argument("--no-segmode", dest="segmode", action="store_false",
                    help="跳过线段层三分类（默认做）")
    a = ap.parse_args(argv)

    cases = [c.strip() for c in a.cases.split(",") if c.strip()]
    outdir = ROOT / "work" / ("_fast_out_" + a.run.strip().rstrip("/\\"))
    print("== 导线端点结构归属（只读）==")
    print("case（%d）：%s  产物：%s  tol=%.1fpx" % (len(cases), " ".join(cases), outdir, a.tol))

    keys = ["pin", "bbox", "wire", "skel", "junc", "none"]
    tot = {k: 0 for k in keys}
    d_skel_all, d_pred_all = [], []
    per_case = []
    seg_tot = {"gt": 0, "tp": 0, "cov_not_match": 0, "not_cov": 0}
    seg_cases = []
    for cid in cases:
        case = case_of(cid)
        if case is None:
            print("  ! 找不到 case %s" % cid)
            continue
        inp = prun.inputs_of(case)
        h = inp.h
        res = prun.skeleton_of(inp, "adaptive")
        cid_out = case.case_id
        pins = _pts([[p.x, p.y] for p in inp.truth.pins.values()])
        bbox_segs = []
        for c in inp.truth.comps.values():
            bbox_segs += rect_segments(c.box)
        skel_pts = np.argwhere(res.skel > 0)
        skel_xy = np.stack([skel_pts[:, 1], skel_pts[:, 0]], axis=1).astype(float) \
            if len(skel_pts) else np.zeros((0, 2))
        skel_ends = _pts(res.endpoints) if len(res.endpoints) else np.zeros((0, 2))
        juncs = _pts(res.junctions) if len(res.junctions) else np.zeros((0, 2))
        pp = pred_path(outdir, cid)
        pred_doc = load_json(pp) if pp else None
        pred_ends = pred_wire_endpoints(pred_doc, h) if pred_doc else np.zeros((0, 2))
        if pp is None:
            print("  ! case %s 没有交付产物（%s）⇒ 只量结构，不量 pred" % (cid, outdir))
        if a.segmode:
            sm = seg_match(inp.gt_doc, pred_doc, h, a.tol)
            if sm:
                for k in ("gt", "tp", "cov_not_match", "not_cov"):
                    seg_tot[k] += sm[k]
                seg_cases.append((cid, sm))

        eps = gt_wire_endpoints(inp.gt_doc, h)
        cnt = {k: 0 for k in keys}
        for e in eps:
            p = e["p"]
            others = []
            for q in eps:
                if q is e or (q["net"], q["edge"]) == (e["net"], e["edge"]):
                    continue
                pl = q["poly"]
                for i in range(len(pl) - 1):
                    others.append((pl[i][0], pl[i][1], pl[i + 1][0], pl[i + 1][1]))
            d_pin = d_point_to_points(p, pins)
            d_bbox = d_point_to_segments(p, bbox_segs)
            d_wire = d_point_to_segments(p, others)
            d_ske = d_point_to_points(p, skel_ends)
            d_jun = d_point_to_points(p, juncs)
            d_skel_px = d_point_to_points(p, skel_xy)
            d_pred = d_point_to_points(p, pred_ends)
            d_skel_all.append(d_skel_px)
            d_pred_all.append(d_pred)
            if d_pin <= a.tol:
                k = "pin"
            elif d_bbox <= a.tol:
                k = "bbox"
            elif d_wire <= a.tol:
                k = "wire"
            elif d_ske <= a.tol:
                k = "skel"
            elif d_jun <= a.tol:
                k = "junc"
            else:
                k = "none"
            cnt[k] += 1
            tot[k] += 1
            if a.detail:
                print("   %-4s net=%-8s %s (%.0f,%.0f) pin %.1f bbox %.1f wire %.1f skelE %.1f "
                      "junc %.1f skelPx %.1f pred %.1f => %s"
                      % (case.case_id, e["net"], e["which"], p[0], p[1], d_pin, d_bbox, d_wire,
                         d_ske, d_jun, d_skel_px, d_pred, k))
        n = len(eps)
        per_case.append((case.case_id, n, cnt,
                         float(np.median(d_skel_all[-n:])) if n else 0.0,
                         float(np.median(d_pred_all[-n:])) if n else 0.0))
        print("  case %-3s 端点 %3d  " % (case.case_id, n) +
              "  ".join("%s %d" % (k, cnt[k]) for k in keys) +
              "   中位 d_skelPx %.1f  d_pred %.1f" % (per_case[-1][3], per_case[-1][4]))

    n = sum(tot.values())
    print("\n-- 合计 %d 个 GT 导线端点 --" % n)
    if n:
        for k in keys:
            print("   %-5s %4d  %5.1f%%   %s" % (k, tot[k], 100.0 * tot[k] / n,
                  {"pin": "钉在 GT 脚点上", "bbox": "钉在元件框边框上",
                   "wire": "搭在别的导线上（T 接）", "skel": "落在图像骨架端点上",
                   "junc": "落在图像骨架结点上",
                   "none": "以上都没有 ⇒ 只能靠线段本身的几何"}[k]))
        sk = np.asarray(d_skel_all)
        pr = np.asarray(d_pred_all)
        print("   骨架像素距离：中位 %.2f  p90 %.2f  ≤%.0fpx 占 %.1f%%" % (
            float(np.median(sk)), float(np.percentile(sk, 90)), a.skel_tol,
            100.0 * float((sk <= a.skel_tol).mean())))
        print("   我们的 pred 端点距 GT 端点：中位 %.2f  p90 %.2f  ≤%.0fpx 占 %.1f%%" % (
            float(np.median(pr)), float(np.percentile(pr, 90)), a.tol,
            100.0 * float((pr <= a.tol).mean())))
        reach = n - tot["none"]
        print("   ⇒ 图像侧**可判定**的上界：%d/%d = %.1f%%（`none` 那 %d 个没有可观测结构可对齐）"
              % (reach, n, 100.0 * reach / n, tot["none"]))
    if a.segmode and seg_tot["gt"]:
        g = seg_tot["gt"]
        print("\n-- 线段层（复刻判分器：每条 edge 只比前两点、unordered、tol %.1fpx）--" % a.tol)
        for cid, sm in seg_cases:
            print("   case %-3s GT edge %3d  配上 %3d  覆盖但没配上 %3d  没覆盖 %3d  （GT 折线平均 %.1f 点）"
                  % (cid, sm["gt"], sm["tp"], sm["cov_not_match"], sm["not_cov"], sm["pts_per_edge"]))
        print("   合计 GT edge %d：配上 %d（%.1f%%）｜**覆盖但没配上 %d（%.1f%%）**｜没覆盖 %d（%.1f%%）"
              % (g, seg_tot["tp"], 100.0 * seg_tot["tp"] / g,
                 seg_tot["cov_not_match"], 100.0 * seg_tot["cov_not_match"] / g,
                 seg_tot["not_cov"], 100.0 * seg_tot["not_cov"] / g))
        print("   ⇒ 「覆盖但没配上」是**分割/端点**问题（线画到了、头两点不对），"
              "只有「没覆盖」才是真缺线")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
