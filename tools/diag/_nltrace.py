# -*- coding: utf-8 -*-
"""§5.126 5.1 归因：回答「基线里那些配不上 GT 的**端点**，到底是 stage2 端点链的
**哪一步**把它搬到当前位置的」。

与 `_nlcrop.py`（那里长什么样）互补：这个工具回答**谁搬的**。

做法：
1. 用 `metrics` 的官方口径拿 net 配对 `pairs`，对每对 (pred_net, gt_net) 按 `line_tol`
   求最大基数匹配（同 `net_line_f1`），得到**配不上的我方边**。
2. 这些边的每一端，找 GT 同网最近段端，> 5px 即「坏端」。
3. 取该端在每个 `STAGE2_TRACE_DUMP` 快照里的位置（同 net key + 同 edge 序号，
   快照按 `edges` 插入序写），量相邻两步之间的位移与方向（正 = 外推，负 = 回缩）。

用法：
  $env:STAGE2_TRACE_DUMP="Y:\\workspace\\EDA\\work\\_trace_all"
  python -X utf8 tools/eval/fast.py --cases 1,...,10 --no-vlm --extra "-offline -vlm-pin off" --out all
  Remove-Item Env:\\STAGE2_TRACE_DUMP
  python -X utf8 tools/diag/_nltrace.py --tracedir ..\\work\\_trace_all
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import math
import os
import sys
from typing import Dict, List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from schematic_parser import metrics as M                      # noqa: E402
from schematic_parser.data import discover, load_json, normalise_doc  # noqa: E402

TOL = 5.0
LONG = 12.0   # 单步位移超过这个值记入 long（可能是端点身份串台，读的时候要警惕）
_DBG_N = [0]
STAGES = ("pre", "snap3", "split", "trim", "extfree", "extany",
          "ext", "pinsnap_box", "overlap", "final")
OPS = [
    ("pre", "snap3", "snap_edge_ends(EDGE_SNAP)"),
    ("snap3", "split", "split_dedup+at_pins+at_xnet"),
    ("split", "trim", "trim_free_ends(EDGE_TRIM=3)"),
    ("trim", "extfree", "snap_ends_to_crossings(CORNER=12)"),
    ("extfree", "extany", "extend_free_ends(EXT_FREE=0)"),
    ("extany", "ext", "extend_ends_to_evidence(EXT_ANY=25)"),
    ("ext", "pinsnap_box", "extend_edge_ends(EDGE_EXT=20)"),
    ("pinsnap_box", "overlap", "snap_ends_to_pins+extend_ends_to_box"),
    ("overlap", "final", "merge_overlapping_edges(OVERLAP=10)"),
]


def _d(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _tag_of(fname: str) -> str:
    parts = os.path.basename(fname).split(".")
    return parts[-2] if len(parts) >= 4 else ""


def _load_stages(tracedir: str, cid: str):
    """同名 tag 取**最后一次**（出口那一趟）。返回 ({tag: rows}, H)。"""
    out, H = {}, 0.0
    for fp in sorted(glob.glob(os.path.join(tracedir, "%s.*.json" % cid))):
        tag = _tag_of(fp)
        if not tag:
            continue
        try:
            with io.open(fp, encoding="utf-8") as fh:
                d = json.load(fh)
        except Exception:
            continue
        out[tag] = d.get("edges") or []
        H = float(d.get("h") or H)
    return out, H


def _iter_edges(rows):
    """按写入序展开成 [(netkey, eidx, p0, p1)]，eidx 用全局递增序号（与快照写入一致）。"""
    cur = {}
    for r in rows:
        nk = r[0]
        i = cur.get(nk, 0)
        cur[nk] = i + 1
        yield nk, i, r[2], r[3]


def _stage_ends(rows, netkey, eidx, H):
    for nk, i, p0, p1 in _iter_edges(rows):
        if nk == netkey and i == eidx:
            return ((float(p0[0]), H - float(p0[1])),
                    (float(p1[0]), H - float(p1[1])))
    return None


def _greedy_match(cost_rows, tol):
    cand = []
    for i, row in enumerate(cost_rows):
        for j, c in enumerate(row):
            if c <= tol:
                cand.append((c, i, j))
    cand.sort(key=lambda z: z[0])
    ui, uj = set(), set()
    for _c, i, j in cand:
        if i in ui or j in uj:
            continue
        ui.add(i)
        uj.add(j)
    return sorted(ui)


def _all_ends(rows, H=None):
    """展开成 [(netkey, p0, p1)]，保留写入序。

    ★ 快照里写的就是**出口文档坐标**（与 `*_Pred.json` 逐位相同，已实测），
    不需要任何翻转 —— `H` 只在文件名/记录里出现，这里不用。
    """
    out = []
    for r in rows:
        out.append((r[0],
                    (float(r[2][0]), float(r[2][1])),
                    (float(r[3][0]), float(r[3][1]))))
    return out


def _nearest(q, prev_nk, cand, maxd, crossd=25.0):
    """找离 q 最近的端点。**同 net key 优先**（跨 net 只在 `crossd` 内才允许，避免串到
    别人的线上）；同 net 内允许放到 `maxd`。返回 (dist, nk, endidx)。"""
    best = None
    for nk, p0, p1 in cand:
        for ei, p in ((0, p0), (1, p1)):
            d = _d(q, p)
            same = (nk == prev_nk)
            if d > (maxd if same else crossd):
                continue
            key = (0 if same else 1, d)
            if best is None or key < best[0]:
                best = (key, d, nk, ei)
    return None if best is None else (best[1], best[2], best[3])


def _dbg_gap(rec, final_ends):
    """诊断：坏端在 `final` 快照里到底差多远（只打前几条）。"""
    if _DBG_N[0] >= 6:
        return
    _DBG_N[0] += 1
    best = None
    for nk, p0, p1 in final_ends:
        for ei, p in ((0, p0), (1, p1)):
            d = _d(rec["q"], p)
            if best is None or d < best[0]:
                best = (d, nk, ei, p)
    print("   [诊断] 坏端 q=%s net=%s → final 快照最近端 %s (d=%.2f)"
          % (rec["q"], rec["pnk"], best[3] if best else None,
             best[0] if best else -1))


def _other_end(rows, H, q, nk):
    """同一条边（含 q 这个端点、net key 匹配）的另一端；用来定轴方向。"""
    for n2, p0, p1 in _all_ends(rows, H):
        if n2 != nk:
            continue
        if _d(p0, q) <= 1.0 or _d(p1, q) <= 1.0:
            other = p1 if _d(p0, q) <= 1.0 else p0
            if _d(other, q) < 50.0:
                return other
    return None


def analyse(cid, data_root, run_dir, tracedir, verbose=False):
    idx = {}
    for c in discover(data_root):
        idx.setdefault(c.case_id, c)
    case = idx.get(cid)
    if case is None:
        return None, "no GT"
    fp = os.path.join(run_dir, "final_results", "%s_Pred.json" % cid)
    if not os.path.exists(fp):
        return None, "no pred %s" % fp
    stages, H = _load_stages(tracedir, cid)
    if "final" not in stages or H <= 0:
        return None, "tracedir 缺 %s 或 h" % cid
    cfg = M.EvalConfig()
    cfg.line_tol = TOL
    gt, pred = normalise_doc(load_json(case.json_path)), normalise_doc(load_json(fp))
    _c, keymap = M.assign_components(gt, pred, cfg, None)
    pred2 = M.rewrite_keys(pred, keymap)
    _n, pairs = M.net_hypergraph_f1(gt, pred2, cfg, None)
    g_nets, p_nets = list(gt["nets"].items()), list(pred2["nets"].items())
    bad = []
    for i, j in pairs:
        pnk, pv = p_nets[i]
        _gnk, gv = g_nets[j]
        p_edges = [M._edge_points(v) for v in (pv.get("edges") or {}).values()]
        g_edges = [M._edge_points(v) for v in (gv.get("edges") or {}).values()]
        p_edges = [e for e in p_edges if e]
        g_edges = [e for e in g_edges if e]
        if not p_edges or not g_edges:
            continue
        cost = []
        for pe in p_edges:
            row = []
            for ge in g_edges:
                if len(pe) < 2 or len(ge) < 2:
                    row.append(float("inf"))
                    continue
                d = M._seg_distance(pe[:2], ge[:2], cfg.line_unordered)
                row.append(d if d <= cfg.line_tol else float("inf"))
            cost.append(row)
        up = set(_greedy_match(cost, cfg.line_tol))
        for pi, pe in enumerate(p_edges):
            if pi in up:
                continue
            for ei in (0, 1):
                q = tuple(float(x) for x in pe[0 if ei == 0 else 1])
                bg = None
                for ge in g_edges:
                    for gp in ge[:2]:
                        dd = _d(q, gp)
                        if bg is None or dd < bg[0]:
                            bg = (dd, tuple(float(x) for x in gp))
                if bg is None or bg[0] <= TOL:
                    continue
                bad.append(dict(pnk=pnk, eidx=pi, ei=ei,
                                q=(float(pe[0][0]), float(pe[0][1])) if ei == 0
                                else (float(pe[-1][0]), float(pe[-1][1])),
                                gt=bg[1], err=bg[0]))
    stat = {name: dict(n=0, moved=0, s=0.0, pos=0, neg=0, mx=0.0, hist={},
                       long=0)
            for _a, _b, name in OPS}
    traced = 0
    lost = 0
    MAXD = 60.0
    dist_by_stage = {}
    # 出口文档里每个坏端，先在 `final` 快照里找回它的坐标，再从那里贪心跨 stage 追回去。
    # 只在**同一 net key** 内追（跨 net 串台会得到毫无意义的轨迹）。
    final_ends = _all_ends(stages.get("final") or [], H)
    for rec in bad:
        # 坏端在 `final` 快照里的位置 = 离它最近的端点；若最近也超过 1px，说明这条边
        # 不在快照里（网被合并/改名），直接放弃这个端。
        start = _nearest(rec["q"], rec["pnk"], final_ends, 1.0, crossd=1.0)
        if start is None:
            _dbg_gap(rec, final_ends)
            lost += 1
            continue
        d_seed, nk, ei = start
        if d_seed > 1.0:
            lost += 1
            continue
        q = None
        for n2, p0, p1 in final_ends:
            if n2 == nk:
                q = p0 if ei == 0 else p1
                break
        if q is None:
            lost += 1
            continue
        # 逆着 chain 往回追：final -> pre
        rev = []
        okall = True
        prev_nk = nk
        pt = q
        for t in reversed(STAGES):
            cand = _all_ends(stages.get(t) or [], H)
            hit = _nearest(pt, prev_nk, cand, MAXD, crossd=LONG)
            if hit is None:
                okall = False
                break
            _dd, nk2, ei2 = hit
            q2 = None
            for n2, p0, p1 in cand:
                if n2 == nk2:
                    q2 = p0 if ei2 == 0 else p1
                    break
            if q2 is None:
                okall = False
                break
            rev.append((t, q2, nk2))
            pt = q2
            prev_nk = nk2
        if not okall:
            lost += 1
            continue
        seq = list(reversed(rev))
        traced += 1
        for k, (t, q, _nk) in enumerate(seq):
            ds = dist_by_stage.setdefault(t, [])
            ds.append(_d(q, rec["gt"]))
        for k, (_a, _b, name) in enumerate(OPS):
            t0, q0, nk0 = seq[k]
            _t1, q1, _nk1 = seq[k + 1]
            mv = _d(q0, q1)
            st = stat[name]
            st["n"] += 1
            if mv > 0.05 and mv <= LONG:
                st["moved"] += 1
                st["s"] += mv
                st["mx"] = max(st["mx"], mv)
                # 「另一端」用来定轴方向（取同一 stage 里同一条边）
                o0 = _other_end(stages[t0], H, q0, nk0)
                o1 = _other_end(stages[seq[k + 1][0]], H, q1, seq[k + 1][2])
                if o0 and o1:
                    vx, vy = o1[0] - q1[0], o1[1] - q1[1]
                    L = math.hypot(vx, vy) or 1.0
                    proj = ((q1[0] - q0[0]) * vx + (q1[1] - q0[1]) * vy) / L
                    if proj > 0:
                        st["pos"] += 1
                    else:
                        st["neg"] += 1
                b = int(mv // 2) * 2
                st["hist"][b] = st["hist"].get(b, 0) + 1
            elif mv > LONG:
                # 大跳：多半是端点身份串台（被切分/合并换了宿主），不计入位移统计
                st["long"] += 1
        if verbose and traced <= 10:
            print("   [样] net=%s err=%.1f 反向追到端点 %s"
                  % (rec["pnk"], rec["err"],
                     " ".join("%s(%.1f,%.1f)" % (t, q[0], q[1]) for t, q, _n in seq)))
    return dict(cid=cid, n_bad=len(bad), traced=traced, lost=lost, stat=stat,
                dist_by_stage=dist_by_stage), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracedir", required=True)
    ap.add_argument("--run-dir",
                    default=r"Y:\workspace\EDA\work\_fast_out_deterministic_default")
    ap.add_argument("--cases", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    data = None
    tot = {name: dict(n=0, moved=0, s=0.0, pos=0, neg=0, mx=0.0, hist={},
                      long=0) for _a, _b, name in OPS}
    nb = nt = nl = 0
    all_dist = {}
    for cid in [c.strip() for c in a.cases.split(",") if c.strip()]:
        rec, err = analyse(cid, data or __import__("schematic_parser.data",
                                                  fromlist=["x"]).DEFAULT_DATA_ROOT,
                           a.run_dir, a.tracedir, a.verbose)
        if rec is None:
            print("case %s: %s" % (cid, err))
            continue
        print("case %2s: 配不上的边端点 %3d，可追溯 %3d（追丢 %3d）"
              % (rec["cid"], rec["n_bad"], rec["traced"], rec["lost"]))
        nb += rec["n_bad"]
        nt += rec["traced"]
        nl += rec["lost"]
        for t, ds in rec["dist_by_stage"].items():
            all_dist.setdefault(t, []).extend(ds)
        for _a, _b, name in OPS:
            s = rec["stat"][name]
            tot[name]["n"] += s["n"]
            tot[name]["moved"] += s["moved"]
            tot[name]["s"] += s["s"]
            tot[name]["pos"] += s["pos"]
            tot[name]["neg"] += s["neg"]
            tot[name]["mx"] = max(tot[name]["mx"], s["mx"])
            tot[name]["long"] += s["long"]
            for k, v in s["hist"].items():
                tot[name]["hist"][k] = tot[name]["hist"].get(k, 0) + v
    print("\n合计：配不上的边端点 %d，可追溯 %d（追丢 %d）" % (nb, nt, nl))
    print("%-42s %5s %6s %8s %6s %6s %7s %6s"
          % ("算子", "端数", "动过", "均位移", "外推", "回缩", "最大", "大跳"))
    for _a, _b, name in OPS:
        s = tot[name]
        print("%-42s %5d %6d %8.2f %6d %6d %7.1f %6d"
              % (name, s["n"], s["moved"], s["s"] / max(s["moved"], 1),
                 s["pos"], s["neg"], s["mx"], s["long"]))
    print("\n反向追溯到的端点**离目标 GT 端的距离**（px，逐 stage 演化）：")
    print("  %-14s %5s %8s %8s %8s" % ("stage", "端数", "中位", "均值", "≤5px"))
    prev = None
    for t in STAGES:
        ds = all_dist.get(t)
        if not ds:
            continue
        ds2 = sorted(ds)
        med = ds2[len(ds2) // 2]
        ok = sum(1 for d in ds if d <= TOL)
        arrow = ""
        if prev is not None:
            arrow = "  (相对上一 stage %+.2f)" % (sum(ds) / len(ds) - prev)
        print("  %-14s %5d %8.1f %8.2f %8d%s"
              % (t, len(ds), med, sum(ds) / len(ds), ok, arrow))
        prev = sum(ds) / len(ds)
    print("\n位移直方图（|位移| 分箱 2px，仅动过的端）：")
    for _a, _b, name in OPS:
        h = tot[name]["hist"]
        if not h:
            continue
        print("  %-42s %s" % (name, " ".join("%d:%d" % (k, h[k]) for k in sorted(h))))


if __name__ == "__main__":
    main()
