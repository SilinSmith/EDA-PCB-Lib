#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""引脚锚点的「恒定偏移」可估性探针（§5.130 第 2 步）。只做可观测性调查，不实装逻辑。

关键修正：偏移不是「每元件」恒定，而是**每元件的每一条边（每列/每行引脚）**恒定。
case 1 U1 实测两个引脚列：左列 19 脚跟 GT 差 (-2.0,+0.5)，右列 7 脚跟 GT 差 (+8.0,+1.4)，
各自 dev≈1px —— 按元件合并就会得到 dev=10.0 的假结论。

  Part A（GT 只作评估基准）  逐 (元件, 边) 求 d = gt - ours 的中位与最大偏离。
  Part B（完全 GT-blind）    从我方锚点沿「背离元件框」的射线走去，找 maroon->green
                            的色转折点（新口径的「红绿导线交点」就是这个），得到估计 e。
  Part C（GT 只作评估）      把 e 施加回我方锚点，报命中率与神谕 Δpin_f1。

用法：
  python -X utf8 tools/diag/_pinanchor.py [--cases 1,2,...] [--ray 1] [--show]
"""
import argparse
import io
import json
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
EDA = os.path.dirname(os.path.dirname(HERE))
if EDA not in sys.path:
    sys.path.insert(0, EDA)

from schematic_parser.data import DEFAULT_DATA_ROOT, discover          # noqa: E402
from schematic_parser.metrics import (                                  # noqa: E402
    EvalConfig, assign_components, rewrite_keys, pin_f1, canon_type,
)

CASES10 = [str(i) for i in range(1, 11)]
DEFAULT_RUN = "Y:/workspace/EDA/work/_fast_out_delivery_default"

BOX_TYPES = {"box", "ic", "boxic"}
RAY_MAX = 26            # 射线最长搜索 px
STABLE_DEV = 2.0        # 每边偏移「恒定」的最大偏离
STABLE_MIN = 2          # 至少几根脚才算数


def box_comps(section):
    out = {}
    for k, c in (section.get("components") or {}).items():
        if canon_type(c.get("type")) in BOX_TYPES:
            out[k] = c
    return out


def _pin_map(section, comp_key):
    out = {}
    for k, v in (section.get("pins") or {}).get(comp_key, {}).items():
        num = str(k).split("_")[-1]
        pt = v.get("point") or {}
        out[num] = (v.get("pinname") or "",
                    (float(pt.get("x", 0.0)), float(pt.get("y", 0.0))))
    return out


def _edge_of(pt, bbox):
    x0, y0, x1, y1 = [float(v) for v in bbox]
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    dx, dy = pt[0] - cx, pt[1] - cy
    if abs(dx) >= abs(dy):
        return "R" if dx > 0 else "L"
    return "T" if dy > 0 else "B"


def _outward_img(edge):
    """doc 坐标的边 -> 图像坐标的背离方向（y 翻转）。"""
    return {"R": (1.0, 0.0), "L": (-1.0, 0.0),
            "T": (0.0, -1.0), "B": (0.0, 1.0)}[edge]


def _classify(img, x, y):
    if x < 0 or y < 0 or y >= img.shape[0] or x >= img.shape[1]:
        return "."
    r, g, b = int(img[y, x, 0]), int(img[y, x, 1]), int(img[y, x, 2])
    if g > r + 30 and g > b + 30:
        return "g"
    if r > g + 30 and r > b + 30:
        return "m"
    return "."


def _ray_transition(img, q, u):
    """从 q 沿 u 走，返回「maroon->green 的转折点」相对 q 的有符号距离（px）。

    转折点 = 最后一个 maroon 与第一个持续 green 的中点。找不到返回 None。
    """
    last_m = None
    first_g = None
    run = 0
    for t in range(0, RAY_MAX + 1):
        x = int(round(q[0] + u[0] * t))
        y = int(round(q[1] + u[1] * t))
        c = _classify(img, x, y)
        if c == "m":
            last_m = t
            run = 0
        elif c == "g":
            run += 1
            if run >= 2 and first_g is None:
                first_g = t - 1
                break
        else:
            run = 0
    if first_g is None:
        return None
    base = last_m if last_m is not None else 0
    return 0.5 * (base + first_g)


def analyse_case(cid, case, run_dir, cfg, ray=True):
    pp = os.path.join(run_dir, "final_results", "%s_Pred.json" % cid)
    if not os.path.exists(pp):
        return None
    gt_doc = json.load(io.open(case.json_path, encoding="utf-8"))
    pred_doc = json.load(io.open(pp, encoding="utf-8"))
    comp, keymap = assign_components(gt_doc, pred_doc, cfg, {})
    pred2 = rewrite_keys(pred_doc, keymap)

    img = np.array(Image.open(case.png_path).convert("RGB"))
    h = img.shape[0]
    gbox = box_comps(gt_doc)
    groups = {}
    for ck in sorted(gbox.keys()):
        gp = _pin_map(gt_doc, ck)
        op = _pin_map(pred2, ck)
        bbox = (pred2.get("components") or {}).get(ck, {}).get("bbox") \
            or gbox[ck].get("bbox")
        if not gp or not op or not bbox:
            continue
        gbyname = {}
        for num, (nm, pt) in gp.items():
            if nm:
                gbyname.setdefault(nm.upper(), []).append((num, pt))
        for num, (nm, pt) in sorted(op.items()):
            edge = _edge_of(pt, bbox)
            key = (ck, edge)
            g = groups.setdefault(key, dict(td=[], ed=[], bbox=bbox))
            cand = gbyname.get((nm or "").upper()) or []
            gnum = None
            if num in gp:                      # 号就是 pin_f1 的连接键，优先按号配
                gnum = num
            elif len(cand) == 1:
                gnum = cand[0][0]
            if gnum is not None:
                gpt = gp[gnum][1]
                g["td"].append((num, pt, gpt, (gpt[0] - pt[0], gpt[1] - pt[1])))
            if ray:
                q = (pt[0], h - pt[1])
                t = _ray_transition(img, q, _outward_img(edge))
                if t is not None:
                    u = _outward_img(edge)
                    g["ed"].append((num, pt, (u[0] * t, -u[1] * t)))
    return dict(case=cid, groups=groups)


def _med(v):
    return float(np.median(v)) if v else 0.0


def _axis_dev(ds):
    if not ds:
        return 0.0
    mx = _med([d[0] for d in ds])
    my = _med([d[1] for d in ds])
    return max([max(abs(d[0] - mx), abs(d[1] - my)) for d in ds])


def _shift_pins(pred_doc, groups, field, use_est, strict=False):
    """按 (元件,边) 的中位偏移整体平移该边上的我方锚点。返回新文档。

    strict=True 时只动「偏移确实恒定且非零」的边（dev<=1.5、n>=3、|中位|>=2.5px），
    避开「同一行里一半精确一半偏 8px」的双峰组 —— 那种组整体平移只会把精确的弄坏。
    """
    doc = json.loads(json.dumps(pred_doc))
    out = doc.setdefault("pins", {})
    shifts = {}
    for (ck, edge), g in groups.items():
        src = g["ed"] if use_est else g["td"]
        if use_est:
            ds = [e[2] for e in src]
        else:
            ds = [d for (_, _, _, d) in src]
        if len(ds) < STABLE_MIN:
            continue
        mx, my = _med([d[0] for d in ds]), _med([d[1] for d in ds])
        if strict:
            if _axis_dev(ds) > 1.5 or len(ds) < 3:
                continue
            if max(abs(mx), abs(my)) < 2.5:
                continue
        if abs(mx) < 0.5 and abs(my) < 0.5:
            continue
        for row in src:
            shifts[(ck, row[0])] = (mx, my)
    for (ck, num), (mx, my) in shifts.items():
        k = "pin_%s" % num
        cell = out.get(ck, {}).get(k)
        if not cell:
            continue
        p = cell.get("point") or {}
        cell["point"] = {"x": float(p.get("x", 0.0)) + mx,
                         "y": float(p.get("y", 0.0)) + my}
    return doc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=DEFAULT_RUN)
    ap.add_argument("--cases", default=",".join(CASES10))
    ap.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    ap.add_argument("--ray", type=int, default=1)
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()

    index = {str(c.case_id): c for c in discover(a.data_root)}
    cfg = EvalConfig()

    n_edge = n_true_const = n_est = n_agree = 0
    oracle_rows = []
    oracle_rows_g = []
    print("== 逐 (元件, 边)：真实偏移 vs 图像（射线色转折）估计 ==")
    print("case comp 边  nA  true(dx,dy)   dev  |  nB  est(dx,dy)     dev   一致")
    for cid in [s.strip() for s in a.cases.split(",") if s.strip()]:
        case = index.get(cid)
        if case is None:
            continue
        r = analyse_case(cid, case, a.run_dir, cfg, bool(a.ray))
        if r is None:
            continue
        # ---- Part C：三种锚点下的 pin_f1 ----
        gt_doc = json.load(io.open(case.json_path, encoding="utf-8"))
        pred_doc = json.load(io.open(
            os.path.join(a.run_dir, "final_results", "%s_Pred.json" % cid),
            encoding="utf-8"))
        _, km = assign_components(gt_doc, pred_doc, cfg, {})
        base = pin_f1(gt_doc, rewrite_keys(pred_doc, km), cfg, {}).f1
        o_true = pin_f1(gt_doc, _shift_pins(rewrite_keys(pred_doc, km), r["groups"], "td", False),
                        cfg, {}).f1
        o_str = pin_f1(gt_doc, _shift_pins(rewrite_keys(pred_doc, km), r["groups"], "td", False,
                                           strict=True), cfg, {}).f1
        o_est = pin_f1(gt_doc, _shift_pins(rewrite_keys(pred_doc, km), r["groups"], "ed", True),
                       cfg, {}).f1
        oracle_rows.append((cid, base, o_true, o_est, o_str))
        oracle_rows_g.append(r)
        for (ck, edge), g in sorted(r["groups"].items()):
            td, ed = g["td"], g["ed"]
            if not td and not ed:
                continue
            n_edge += 1
            tdev = _axis_dev([d for (_, _, _, d) in td])
            tdx = _med([d[0] for (_, _, _, d) in td])
            tdy = _med([d[1] for (_, _, _, d) in td])
            if tdev <= STABLE_DEV and len(td) >= STABLE_MIN:
                n_true_const += 1
            edx = _med([e[2][0] for e in ed])
            edy = _med([e[2][1] for e in ed])
            edev = _axis_dev([e[2] for e in ed])
            agree = "-"
            if len(ed) >= STABLE_MIN:
                n_est += 1
                agree = "YES" if max(abs(edx - tdx), abs(edy - tdy)) <= 2.0 else "no"
                if agree == "YES":
                    n_agree += 1
            print("%-4s %-5s %-2s %3d  %6.1f,%6.1f %5.1f | %3d  %6.1f,%6.1f %6.1f  %s"
                  % (cid, ck, edge, len(td), tdx, tdy, tdev,
                     len(ed), edx, edy, edev, agree))
            if a.show:
                for (num, pt, gpt, d) in td:
                    print("       true %-4s ours=(%.1f,%.1f) gt=(%.1f,%.1f) d=(%.1f,%.1f)"
                          % (num, pt[0], pt[1], gpt[0], gpt[1], d[0], d[1]))
                for e in ed:
                    print("       est  %-4s e=(%.1f,%.1f)" % (e[0], e[2][0], e[2][1]))
    print()
    print("(元件,边) 合计 %d；真实偏移恒定(dev<=%.1f, n>=%d) %d (%.1f%%)；"
          "图像估出(n>=%d) %d；估值与真值一致 %d"
          % (n_edge, STABLE_DEV, STABLE_MIN, n_true_const,
             100.0 * n_true_const / max(1, n_edge), STABLE_MIN, n_est, n_agree))
    print()
    print("== Part C：(元件,边) 恒定偏移修正的 pin_f1 ==")
    print("case   base   真实偏移(全部边)  真实偏移(只动恒定边)  图像估计(GT-blind)")
    nb = nt = ne = ns = 0.0
    for (cid, base, o_true, o_est, o_str) in oracle_rows:
        nb += base
        nt += o_true
        ne += o_est
        ns += o_str
        print("%-5s %7.4f  %7.4f (%+7.4f)  %7.4f (%+7.4f)  %7.4f (%+7.4f)"
              % (cid, base, o_true, o_true - base, o_str, o_str - base,
                 o_est, o_est - base))
    n = max(1, len(oracle_rows))
    print("均值  %7.4f  %7.4f (%+7.4f)  %7.4f (%+7.4f)  %7.4f (%+7.4f)"
          % (nb / n, nt / n, (nt - nb) / n, ns / n, (ns - nb) / n,
             ne / n, (ne - nb) / n))
    print("分数  [真实全部 %+.2f | 真实恒定边 %+.2f | 图像估计 %+.2f]   （×0.25 权重）"
          % (25.0 * (nt - nb) / n, 25.0 * (ns - nb) / n, 25.0 * (ne - nb) / n))
    # ---- 估计 vs 真值的相关（只用两边都有 n>=2 的组） ----
    pairs = []
    for r in (oracle_rows_g or []):
        for (ck, edge), g in r["groups"].items():
            if len(g["td"]) >= STABLE_MIN and len(g["ed"]) >= STABLE_MIN:
                td = [d for (_, _, _, d) in g["td"]]
                ed = [e[2] for e in g["ed"]]
                pairs.append((_med([d[0] for d in td]), _med([e[0] for e in ed]),
                              _med([d[1] for d in td]), _med([e[1] for e in ed])))
    if pairs:
        tx = np.array([p[0] for p in pairs])
        ex = np.array([p[1] for p in pairs])
        ty = np.array([p[2] for p in pairs])
        ey = np.array([p[3] for p in pairs])
        rx = float(np.corrcoef(tx, ex)[0, 1]) if tx.std() > 1e-9 and ex.std() > 1e-9 else float("nan")
        ry = float(np.corrcoef(ty, ey)[0, 1]) if ty.std() > 1e-9 and ey.std() > 1e-9 else float("nan")
        print()
        print("估计 vs 真值（%d 组同时有两者）：相关 r_x=%s r_y=%s；"
              "平均绝对差 MAE_x=%.1fpx MAE_y=%.1fpx；命中(|差|<=2px) %d/%d"
              % (len(pairs), ("%.2f" % rx) if rx == rx else "n/a",
                 ("%.2f" % ry) if ry == ry else "n/a",
                 float(np.mean(np.abs(tx - ex))), float(np.mean(np.abs(ty - ey))),
                 int(np.sum((np.abs(tx - ex) <= 2.0) & (np.abs(ty - ey) <= 2.0))), len(pairs)))
        print("真值 x 范围 [%.1f, %.1f]；估计 x 范围 [%.1f, %.1f]"
              % (tx.min(), tx.max(), ex.min(), ex.max()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
