# -*- coding: utf-8 -*-
"""第一阶段：网络标签**候选**的验收（不涉及文字内容）。

判据（只用官方 GT，不依赖任何非官方文件）：
  - GT 每条 `net_line` 的两个端点是「走线的端点」。其中落在某个 `pins[*][*].point`
    附近的，是**引脚端**；其余的就是**标签端**（网络标签画在那里）。
  - 我的 `stage1.labels.label_candidates` 给出「文字框 + 它的自由端」。
  - 一对一贪心配对：候选的自由端与某个 GT 标签端距离 <= `MATCH_TOL` 即 TP。
  - R = TP / GT 标签端数（**标签定位的召回**，也就是这条线能救回多少走线端点）；
    P = TP / 候选数。

同时报两个中间量，便于判断精度损失来自哪一环：
  `落脚` = 候选的自由端落在**任意** GT 走线端点上（说明这个自由端是真的走线端）；
  `贴引脚` = 候选的自由端落在某个 GT 引脚点上（那本该是引脚连接点，不是标签）。

用法：
    python Y:\workspace\EDA\work\stage1_label_eval.py --stage1 Y:\workspace\EDA\work\stage1_all13
"""
import argparse
import json
import os
import sys
import time

import numpy as np

for _d in ("eval", "diag", "data", "gnd", "viz"):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), _d))
sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")
from schematic_parser.data import discover, DEFAULT_DATA_ROOT  # noqa: E402
from stage1.labels import label_candidates, _as_segs  # noqa: E402
from stage1_conn_eval import _dist_pts_to_segs, gt_pins_and_nets  # noqa: E402

PIN_TOL = 6.0      # 「这个走线端点是不是引脚端」的距离门限
MATCH_TOL = 6.0    # 候选自由端 <-> GT 标签端 的配对距离
UNIQ_TOL = 3.0     # 两端点距离小于它就算同一个端点（GT 相邻走线共享端点）


def _xy(p):
    if isinstance(p, dict):
        return float(p["x"]), float(p["y"])
    return float(p[0]), float(p[1])


def gt_label_ends(doc, H, pin_pts, pin_tol=PIN_TOL):
    """GT 走线端点里**不是引脚端**的那些 -> (K,2) 以及总共的端点数。"""
    ends = []
    for net in (doc.get("nets") or {}).values():
        for e in (net.get("edges") or {}).values():
            if len(e) < 2:
                continue
            for p in (e[0], e[-1]):
                x, y = _xy(p)
                ends.append((x, H - y))
    if not ends:
        return np.zeros((0, 2), np.float64), 0, 0
    E = np.asarray(ends, np.float64)
    # 去重（同一坐标被多条边走共用）
    keep = []
    for i in range(len(E)):
        dup = False
        for j in keep:
            if abs(E[i][0] - E[j][0]) <= UNIQ_TOL and abs(E[i][1] - E[j][1]) <= UNIQ_TOL:
                dup = True
                break
        if not dup:
            keep.append(i)
    U = E[keep]
    if not len(pin_pts):
        return U, len(U), len(E)
    PP = np.asarray(pin_pts, np.float64).reshape(-1, 2)
    D = np.hypot(U[:, 0][:, None] - PP[:, 0][None, :], U[:, 1][:, None] - PP[:, 1][None, :])
    dmin = D.min(axis=1) if D.size else np.full(len(U), 1e9)
    lab = U[dmin > pin_tol]
    return lab, len(lab), len(U)


def greedy_match(a, b, tol):
    """a (M,2) 与 b (K,2) 一对一贪心配对，返回配对数。"""
    a = np.asarray(a, np.float64).reshape(-1, 2)
    b = np.asarray(b, np.float64).reshape(-1, 2)
    if not len(a) or not len(b):
        return 0, set(), set()
    D = np.hypot(a[:, 0][:, None] - b[:, 0][None, :], a[:, 1][:, None] - b[:, 1][None, :])
    cand = []
    for i in range(len(a)):
        for j in np.nonzero(D[i] <= tol)[0]:
            cand.append((float(D[i][j]), i, int(j)))
    cand.sort(key=lambda t: t[0])
    ua, ub, n = set(), set(), 0
    for _, i, j in cand:
        if i in ua or j in ub:
            continue
        ua.add(i)
        ub.add(j)
        n += 1
    return n, ua, ub


def source_of(path):
    base = os.path.basename(path)
    for tag in ("KiCad", "Altium", "Datasheet", "jlc"):
        if tag.lower() in base.lower():
            return tag
    return "other"


def run(stage1_dir, data_root, limit=0, verbose=0, skeleton_only=False,
        min_group_len=0.0, group_tol=1.5):
    cases = sorted(discover(data_root), key=lambda c: c.case_id)
    if limit:
        cases = cases[:limit]
    tot = dict(n_lab=0, n_end=0, n_cand=0, tp=0, on_end=0, on_pin=0)
    by_src = {}
    rows = []
    dd_hist = []
    t_wall = time.time()
    n_case = 0
    for case in cases:
        sp = os.path.join(stage1_dir, case.case_id + ".stage1.json")
        if not os.path.exists(sp):
            continue
        sd = json.load(open(sp, encoding="utf-8"))
        H = float(sd["image"]["h"])
        gdoc = json.load(open(case.json_path, encoding="utf-8"))
        segs = _as_segs(sd["wire_detail"]["segments"])
        if skeleton_only:
            n_ray = int((sd["wire_detail"] or {}).get("n_rays") or 0)
            if n_ray:
                segs = segs[:len(segs) - n_ray]
        if not len(segs):
            continue
        texts = sd.get("texts") or []
        comps = sd.get("components") or []
        cands = label_candidates(segs, texts, boxes=comps, min_group_len=min_group_len,
                                 tol=group_tol)

        pins, _nets = gt_pins_and_nets(gdoc, H)
        pin_pts = np.asarray([p for _, p in pins], np.float64).reshape(-1, 2)
        lab, n_lab, n_end = gt_label_ends(gdoc, H, pin_pts)

        cfe = np.asarray([c["free_end"] for c in cands], np.float64).reshape(-1, 2)
        tp, _, _ = greedy_match(cfe, lab, MATCH_TOL)
        on_end = 0
        gsegs = None
        if len(cfe) and n_end:
            all_ends = []
            for net in (gdoc.get("nets") or {}).values():
                for e in (net.get("edges") or {}).values():
                    if len(e) < 2:
                        continue
                    for p in (e[0], e[-1]):
                        x, y = _xy(p)
                        all_ends.append((x, H - y))
            AE = np.asarray(all_ends, np.float64)
            D = np.hypot(cfe[:, 0][:, None] - AE[:, 0][None, :],
                         cfe[:, 1][:, None] - AE[:, 1][None, :])
            on_end = int(np.sum(D.min(axis=1) <= MATCH_TOL))
            # 自由端到「GT 整条走线」的距离：分得清「停在走线中段」还是「停在空白」
            gsegs = []
            for net in (gdoc.get("nets") or {}).values():
                for e in (net.get("edges") or {}).values():
                    if len(e) < 2:
                        continue
                    x1, y1 = _xy(e[0])
                    x2, y2 = _xy(e[-1])
                    gsegs.append((x1, H - y1, x2, H - y2))
            gsegs = np.asarray(gsegs, np.float64).reshape(-1, 4)
        on_pin = 0
        if len(cfe) and len(pin_pts):
            Dp = np.hypot(cfe[:, 0][:, None] - pin_pts[:, 0][None, :],
                          cfe[:, 1][:, None] - pin_pts[:, 1][None, :])
            on_pin = int(np.sum(Dp.min(axis=1) <= MATCH_TOL))

        if cfe is not None and len(cfe) and gsegs is not None and len(gsegs):
            Dg = _dist_pts_to_segs(cfe, gsegs).min(axis=1)
            dd_hist.extend(Dg.tolist())

        tot["n_lab"] += len(lab)
        tot["n_end"] += n_end
        tot["n_cand"] += len(cands)
        tot["tp"] += tp
        tot["on_end"] += on_end
        tot["on_pin"] += on_pin
        s = source_of(case.json_path)
        d = by_src.setdefault(s, dict(n_lab=0, n_cand=0, tp=0))
        d["n_lab"] += len(lab)
        d["n_cand"] += len(cands)
        d["tp"] += tp
        rows.append((case.case_id, s, n_lab, len(cands), tp, on_end, on_pin))
        n_case += 1
        if verbose and n_case <= verbose:
            print("  {:<10} {:<10} 标签端 {:>4}  候选 {:>4}  TP {:>4}  落脚 {:>4}  贴引脚 {:>4}"
                  .format(case.case_id, s, n_lab, len(cands), tp, on_end, on_pin))
    dt = time.time() - t_wall

    n_lab, n_cand, tp = tot["n_lab"], tot["n_cand"], tot["tp"]
    P = tp / n_cand if n_cand else 0.0
    R = tp / n_lab if n_lab else 0.0
    F = 2 * P * R / (P + R) if (P + R) else 0.0
    print("\n=== 网络标签候选验收（{} 张，官方 GT，配对 {}px）===".format(n_case, MATCH_TOL))
    print("GT 走线端点数 {}   其中标签端 {}（{:.1f}%）".format(
        tot["n_end"], n_lab, 100.0 * n_lab / max(1, tot["n_end"])))
    print("我的候选数 {}（每张 {:.1f}）".format(n_cand, n_cand / max(1, n_case)))
    print("TP {}   P {:.4f}   R {:.4f}   F1 {:.4f}".format(tp, P, R, F))
    print("候选自由端落在真走线端点上 {}/{} = {:.1f}%".format(
        tot["on_end"], n_cand, 100.0 * tot["on_end"] / max(1, n_cand)))
    print("候选自由端落在 GT 引脚点上 {}/{} = {:.1f}%（本该是引脚连接，不是标签）".format(
        tot["on_pin"], n_cand, 100.0 * tot["on_pin"] / max(1, n_cand)))
    print("耗时 {:.1f}s = {:.1f} ms/张".format(dt, dt / max(1, n_case) * 1000))
    if dd_hist:
        a = np.asarray(dd_hist, np.float64)
        print("候选自由端到最近 GT 走线的距离：p10 {:.1f} / p25 {:.1f} / p50 {:.1f} / "
              "p75 {:.1f} / p90 {:.1f} px".format(*[float(np.percentile(a, q))
                                                    for q in (10, 25, 50, 75, 90)]))
        print("  <=2px（就在走线上、只是不在端点）{:.1f}%   <=6px {:.1f}%   >20px（落在空白）{:.1f}%"
              .format(100.0 * float(np.mean(a <= 2)), 100.0 * float(np.mean(a <= 6)),
                      100.0 * float(np.mean(a > 20))))
    print("\n按来源：")
    for s, d in sorted(by_src.items()):
        p = d["tp"] / d["n_cand"] if d["n_cand"] else 0.0
        r = d["tp"] / d["n_lab"] if d["n_lab"] else 0.0
        print("  {:<10} 标签端 {:>5}  候选 {:>5}  TP {:>5}  P {:.4f}  R {:.4f}".format(
            s, d["n_lab"], d["n_cand"], d["tp"], p, r))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", default=r"Y:\workspace\EDA\work\stage1_all13")
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--verbose", type=int, default=0)
    ap.add_argument("--skeleton-only", action="store_true",
                    help="只用骨架段做标签候选（射线在末尾，按 n_rays 切掉）")
    ap.add_argument("--min-group-len", type=float, default=0.0,
                    help="分量总长小于它的当作碎片，其端点不算走线端")
    ap.add_argument("--group-tol", type=float, default=1.5,
                    help="段并连通分量的端点容差（桥接断口）")
    a = ap.parse_args()
    run(a.stage1, a.data_root, limit=a.limit, verbose=a.verbose,
        skeleton_only=a.skeleton_only, min_group_len=a.min_group_len,
        group_tol=a.group_tol)


if __name__ == "__main__":
    main()
