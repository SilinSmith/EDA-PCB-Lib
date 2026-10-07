# -*- coding: utf-8 -*-
"""stage1 连通性评测（用户 m04360 第三条）。

为什么要这个：`stage1_wires_eval.py` 用的是**逐段邻近**尺子（每条 GT 段附近有没有
预测段），它完全不测**连通性** —— 两条挨在一起的真导线在我这边分成两个分量，
逐段邻近照样满分。用户的原话是「部分明明挨在一起的导线居然没有识别成连通的」。

这里改测真正要交付的东西：**同一 GT net 里的引脚，在我的导线图里是否属于同一个
连通分量**。做法：
  1. GT 侧：`pins[c][p] = point`（json 左下原点 -> 图像左上原点），
     `nets[*].hyperGraph` 给出每个 net 的引脚集合 => GT 的引脚划分。
  2. 我的侧：把所有预测段用并查集按**端点距离 <= tol** 并成连通分量；
     每个 GT 引脚找 tol_pin 内最近的段，取该段所在分量当这个引脚的归属；
     找不到段的引脚自成一个单例分量。
  3. 比两个划分：同一 GT net 的引脚对，在我的侧同分量的比例 = 连通召回；
     不同 GT net 的引脚对，在我侧不同分量的比例 = 连通精度。

扫 tol = 1/2/4/8/16：如果加大 tol 就能大幅回血，说明是**几个像素的缝隙**没接上
（也就是用户说的「明明挨在一起」）；如果回不动，就是那一截根本没提出来。

用法：
    python Y:\workspace\EDA\work\stage1_conn_eval.py --stage1 Y:\workspace\EDA\work\stage1_all11
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

for _d in ("eval", "diag", "data", "gnd", "viz"):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), _d))
sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")
from schematic_parser.data import (DEFAULT_DATA_ROOT, discover,  # noqa: E402
                                   parse_hypergraph, pin_key_to_number)

CELL = 16.0


def seg_arrays(detail_segments):
    """把 wire_detail['segments'] 摊成 N x 4 的 (x1,y1,x2,y2)。"""
    out = []
    for s in detail_segments:
        a, b = s
        out.append((float(a[0]), float(a[1]), float(b[0]), float(b[1])))
    return np.asarray(out, dtype=np.float64).reshape(-1, 4)


def pt_seg_d2(px, py, x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    L2 = dx * dx + dy * dy
    if L2 < 1e-12:
        return (px - x1) ** 2 + (py - y1) ** 2
    t = ((px - x1) * dx + (py - y1) * dy) / L2
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    qx, qy = x1 + t * dx, y1 + t * dy
    return (px - qx) ** 2 + (py - qy) ** 2


def _dist_pts_to_segs(pts, segs):
    """pts (M,2) 到 segs (N,4) 的距离矩阵 (M,N)，向量化暴力版。

    **不要用 SegIndex 代替它**：网格索引在真实数据上静默漏找，把连通 R 压低约 6 倍
    （`_conn_check.py` 实测：同一份产物，暴力 0.7889 vs 网格 0.1297）。
    """
    p = np.asarray(pts, np.float64).reshape(-1, 2)
    s = np.asarray(segs, np.float64).reshape(-1, 4)
    if not len(s) or not len(p):
        return np.zeros((len(p), len(s)), np.float64)
    x1, y1, x2, y2 = s[:, 0], s[:, 1], s[:, 2], s[:, 3]
    dx, dy = x2 - x1, y2 - y1
    L2 = dx * dx + dy * dy
    L2 = np.where(L2 < 1e-12, 1e-12, L2)
    t = ((p[:, 0][:, None] - x1[None, :]) * dx[None, :]
         + (p[:, 1][:, None] - y1[None, :]) * dy[None, :]) / L2[None, :]
    t = np.clip(t, 0.0, 1.0)
    return np.hypot(p[:, 0][:, None] - (x1[None, :] + t * dx[None, :]),
                    p[:, 1][:, None] - (y1[None, :] + t * dy[None, :]))


class SegIndex(object):
    """把线段塞进边长 CELL 的网格，好按点查附近线段。"""

    def __init__(self, segs, pad):
        self.segs = segs
        self.pad = pad
        self.grid = {}
        for i in range(len(segs)):
            x1, y1, x2, y2 = segs[i]
            gx0 = int(math.floor((min(x1, x2) - pad) / CELL))
            gx1 = int(math.floor((max(x1, x2) + pad) / CELL))
            gy0 = int(math.floor((min(y1, y2) - pad) / CELL))
            gy1 = int(math.floor((max(y1, y2) + pad) / CELL))
            for gx in range(gx0, gx1 + 1):
                for gy in range(gy0, gy1 + 1):
                    self.grid.setdefault((gx, gy), []).append(i)

    def nearest(self, px, py):
        gx = int(math.floor(px / CELL))
        gy = int(math.floor(py / CELL))
        r = int(math.ceil(self.pad / CELL))
        best_i, best_d2 = -1, float("inf")
        for ax in range(gx - r, gx + r + 1):
            for ay in range(gy - r, gy + r + 1):
                for i in self.grid.get((ax, ay), ()):
                    x1, y1, x2, y2 = self.segs[i]
                    d2 = pt_seg_d2(px, py, x1, y1, x2, y2)
                    if d2 < best_d2:
                        best_d2, best_i = d2, i
        return best_i, (math.sqrt(best_d2) if best_i >= 0 else None)


class UF(object):
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        p = self.p
        while p[a] != a:
            p[a] = p[p[a]]
            a = p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


WELD = 0.0          # 方向性桥接距离；0 = 关闭。见 stage1/labels.py 的 _weld_unions
EXT = 0.0           # 端点沿外法线朝前延伸距离；0 = 关闭。见 _extend_unions
EXT_PERP = 2.0
EXT_COS = 1.0       # <1 时要求被撞的段与射线大致垂直（T 型）
EXT_NEAREST = True  # True = 每个端点只接正前方最近的一条


def build_groups(segs, tol, weld=None, ext=None, ext_perp=None, ext_cos=None,
                 ext_nearest=None):
    """把段并成连通分量。返回 {seg_idx: group_root}。

    **现在委托给 `stage1.labels.wire_groups`** —— 全项目只留一份图定义。之前这里
    自己实现了一份，两边的 bug 各自独立地把我误导了半天：这里踩的是
    `np.concatenate(axis=0)` 把第 i 段与自己的**序号后继**并起来（整张图塌成一个
    分量，连通 P 塌到 0.01），`labels.py` 那边踩的是 `dist_pts_to_segs` 返回
    `(M,)` 而当成 `(M,N)` 用（所有引脚绑到第 0 条段）。合成自检见
    `work\_conn_comp.py` 的 `selftest()`。

    合并规则（详见 `wire_groups`）：端点-端点 <= tol、端点-线段 <= tol（T 型结点）、
    以及 `weld > 0` 时的**方向性桥接**（只接沿彼此延长线、方向相对的两个端点；
    纯距离门槛能把连通 R 从 0.157 抬到 0.534，但同时把精度从 0.566 砸到 0.121）。
    """
    from stage1.labels import wire_groups
    n = len(segs)
    if tol <= 0 or not n:
        return {i: i for i in range(n)}
    arr = segs if isinstance(segs, np.ndarray) else np.asarray(segs, np.float64)
    roots = wire_groups(arr, tol=float(tol),
                        weld=float(WELD if weld is None else weld),
                        ext=float(EXT if ext is None else ext),
                        ext_perp=float(EXT_PERP if ext_perp is None else ext_perp),
                        ext_cos=float(EXT_COS if ext_cos is None else ext_cos),
                        ext_nearest=bool(EXT_NEAREST if ext_nearest is None
                                         else ext_nearest))
    return {i: int(roots[i]) for i in range(n)}


def gt_pins_and_nets(doc, H):
    """返回 [(pin_key, (x_img, y_img)), ...] 与 [(net_id, [pin_key, ...]), ...]。

    注意两边的键不同名：`pins[c]` 的键是 `pin_<number>`，而 `hyperGraph` 用的是
    `(comp, <number>)`。统一归一化成 `(comp, number)`（用 data.pin_key_to_number）。
    """
    pins = []
    for comp, plist in (doc.get("pins") or {}).items():
        for pkey, rec in plist.items():
            pt = rec.get("point") or {}
            x, y = pt.get("x"), pt.get("y")
            if x is None or y is None:
                continue
            key = (comp, pin_key_to_number(pkey))
            pins.append((key, (float(x), H - float(y))))
    nets = []
    for nid, net in (doc.get("nets") or {}).items():
        keys = parse_hypergraph(net.get("hyperGraph") or "")
        if keys:
            nets.append((nid, [tuple(k) for k in keys]))
    return pins, nets


def pair_sets(keys):
    out = set()
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            out.add((a, b) if a <= b else (b, a))
    return out


def gt_edge_segs(doc, H):
    """GT 自己的 net_line 线段（json 左下原点 -> 图像左上原点）。

    这是本评测的**对照组**：如果连 GT 自己的走线几何都连不上同一 net 的引脚，
    说明那些引脚是靠 net label 逻辑相连、纸上根本没有画线 —— 那就不是提取的问题。
    """
    out = []
    for nid, net in (doc.get("nets") or {}).items():
        for e in (net.get("edges") or {}).values():
            if len(e) < 2:
                continue
            def _xy(p):
                if isinstance(p, dict):
                    return float(p["x"]), float(p["y"])
                return float(p[0]), float(p[1])
            x1, y1 = _xy(e[0])
            x2, y2 = _xy(e[1])
            out.append((x1, H - y1, x2, H - y2))
    return np.asarray(out, dtype=np.float64).reshape(-1, 4)


def run(stage1_dir, data_root, tol_list, pin_tol, limit=0, split=None, source="stage1",
        weld_list=None, ext_list=None, ext_perp=None, ext_cos=None, ext_nearest=None):
    cases = sorted(discover(data_root), key=lambda c: c.case_id)
    if split:
        cases = [c for c in cases if c.split == split]
    if limit:
        cases = cases[:limit]

    weld_list = [float(w) for w in (weld_list or [0.0])]
    ext_list = [float(e) for e in (ext_list or [EXT])]
    ext_perp = float(EXT_PERP if ext_perp is None else ext_perp)
    ext_cos = float(EXT_COS if ext_cos is None else ext_cos)
    ext_nearest = bool(EXT_NEAREST if ext_nearest is None else ext_nearest)
    grid = [(float(t), w, e) for t in tol_list for w in weld_list for e in ext_list]
    acc = {k: dict(tp=0, fp=0, fn=0) for k in grid}
    diag_tols = set(tol_list[:2])
    n_case = 0
    n_pin_pair_pos = 0
    n_pin_pair_neg = 0
    n_unbound = 0
    n_pins = 0
    t_wall = time.time()
    for case in cases:
        sp = os.path.join(stage1_dir, case.case_id + ".stage1.json")
        if not os.path.exists(sp):
            continue
        sd = json.load(open(sp, encoding="utf-8"))
        H = float(sd["image"]["h"])
        gdoc = json.load(open(case.json_path, encoding="utf-8"))
        if source == "gt":
            segs = gt_edge_segs(gdoc, H)
        else:
            segs = seg_arrays(sd["wire_detail"]["segments"])
        pins, nets = gt_pins_and_nets(gdoc, H)
        if not pins or not nets:
            continue
        n_case += 1
        key2pt = dict(pins)
        key2gid = {}
        for nid, keys in nets:
            for k in keys:
                key2gid.setdefault(k, nid)
        # 每个引脚 -> 最近的段（暴力最近段，不用 SegIndex，理由见 _dist_pts_to_segs）
        pin2seg = {}
        keys_all = [k for k, _ in pins]
        pts_all = np.asarray([pt for _, pt in pins], np.float64).reshape(-1, 2)
        if len(segs) and len(keys_all):
            D = _dist_pts_to_segs(pts_all, segs)
            jmin = np.argmin(D, axis=1)
            dmin = D[np.arange(len(keys_all)), jmin]
            for k, j, d in zip(keys_all, jmin, dmin):
                n_pins += 1
                if float(d) <= pin_tol:
                    pin2seg[k] = int(j)
                else:
                    n_unbound += 1
        else:
            n_unbound += len(keys_all)
            n_pins += len(keys_all)

        # 正样本对：同一 net 的引脚对（只需在两侧都存在的 pin 上算）
        pos_pairs, neg_pairs = set(), set()
        for nid, keys in nets:
            ks = [k for k in keys if k in key2pt]
            pos_pairs |= pair_sets(ks)
        allks = [k for k in key2pt]
        # 负样本对太多，做确定性采样：每个 net 抽固定几个代表
        groups = {}
        for k in allks:
            groups.setdefault(key2gid.get(k, "?"), []).append(k)
        reps = {g: v[:6] for g, v in groups.items()}
        gl = list(reps)
        for i in range(len(gl)):
            for j in range(i + 1, len(gl)):
                for a in reps[gl[i]]:
                    for b in reps[gl[j]]:
                        neg_pairs.add((a, b) if a <= b else (b, a))
        n_pin_pair_pos += len(pos_pairs)
        n_pin_pair_neg += len(neg_pairs)

        for (t, wl, ex) in grid:
            g_of = build_groups(segs, t, wl, ex, ext_perp, ext_cos, ext_nearest) \
                if len(segs) else {}
            pin2grp = {}
            for key in key2pt:
                si = pin2seg.get(key)
                pin2grp[key] = g_of.get(si, ("solo", si)) if si is not None else ("solo", key)
            a = acc[(t, wl, ex)]
            # 判定：预测「这两根引脚同分量」。
            # 正样本 = 同一 GT net 的引脚对，负样本 = 不同 GT net 的引脚对。
            # TP/FP/FN 只来自这两类，绝不能把「负样本正确分开」也记进 TP
            # （第一版就是这么错的，R 被 1.67M 个负样本抬到 0.99 的假象）。
            for (u, v) in pos_pairs:
                if pin2grp.get(u) == pin2grp.get(v):
                    a["tp"] += 1
                else:
                    a["fn"] += 1
                    if u not in pin2seg or v not in pin2seg:
                        a["fn_unbound"] = a.get("fn_unbound", 0) + 1
                    else:
                        a["fn_split"] = a.get("fn_split", 0) + 1
            for (u, v) in neg_pairs:
                if pin2grp.get(u) == pin2grp.get(v):
                    a["fp"] += 1
            # net 粒度：同一 net 的引脚在我这边跨了几个分量
            for nid, keys in nets:
                ks = [k for k in keys if k in key2pt]
                if len(ks) < 2:
                    continue
                span = len(set(pin2grp[k] for k in ks))
                a["nets"] = a.get("nets", 0) + 1
                a["net1"] = a.get("net1", 0) + (1 if span == 1 else 0)
                a["span_sum"] = a.get("span_sum", 0) + span
                a["span_max"] = max(a.get("span_max", 0), span)

            # 诊断：两边都接上了、却分属不同分量的失败对，两个分量之间到底离多远？
            # 近距离 => 只是「没并起来」（小缝隙 / T 型结点没接）；远距离 => 中间那段
            # 根本就没提出来。两者要分开治。
            if t in diag_tols and wl == 0.0 and ex == 0.0:
                cpts = {}
                for i in range(len(segs)):
                    r = g_of.get(i)
                    if r is None:
                        continue
                    bucket = cpts.setdefault(r, [])
                    if len(bucket) < 240:
                        bucket.append((segs[i][0], segs[i][1]))
                        bucket.append((segs[i][2], segs[i][3]))
                arrs = {r: np.asarray(v, dtype=np.float64) for r, v in cpts.items()}
                for (u, v) in pos_pairs:
                    if pin2grp.get(u) == pin2grp.get(v):
                        continue
                    if u not in pin2seg or v not in pin2seg:
                        continue
                    ra, rb = g_of.get(pin2seg[u]), g_of.get(pin2seg[v])
                    A, B = arrs.get(ra), arrs.get(rb)
                    if A is None or B is None or not len(A) or not len(B):
                        continue
                    As = A[:: max(1, len(A) // 40)][:40]
                    Bs = B[:: max(1, len(B) // 40)][:40]
                    d = np.sqrt(((As[:, None, :] - Bs[None, :, :]) ** 2).sum(-1)).min()
                    a.setdefault("gaps", []).append(float(d))

    print("cases %d / pins %d / 未绑定到任何段的引脚 %d (%.1f%%)"
          % (n_case, n_pins, n_unbound, 100.0 * n_unbound / max(1, n_pins)))
    print("正样本对 %d / 负样本对 %d / 墙钟 %.1fs"
          % (n_pin_pair_pos, n_pin_pair_neg, time.time() - t_wall))
    print("")
    print("| tol | weld | ext | 同 net 连上(R) | 异 net 分开(P) | F1 | net 完整率 | 平均跨度 | FN=引脚没接上 | FN=接上了但没连通 |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for (t, wl, ex) in grid:
        a = acc[(t, wl, ex)]
        tp, fp, fn = a["tp"], a["fp"], a["fn"]
        R = tp / max(1, tp + fn)
        P = tp / max(1, tp + fp)
        F1 = 2 * P * R / max(1e-9, P + R)
        nn = max(1, a.get("nets", 0))
        print("| %g | %g | %g | %.4f | %.4f | **%.4f** | %.4f | %.3f | %d | %d |"
              % (t, wl, ex, R, P, F1, a.get("net1", 0) / nn, a.get("span_sum", 0) / nn,
                 a.get("fn_unbound", 0), a.get("fn_split", 0)))
    for (t, wl, ex) in grid:
        g = acc[(t, wl, ex)].get("gaps") or []
        if not g:
            continue
        g = np.asarray(g)
        print("")
        print("tol=%g weld=%g ext=%g 失败对的两个分量最近距离（n=%d）：" % (t, wl, ex, len(g)))
        for q in (10, 25, 50, 75, 90):
            print("  p%-3d %6.1f px" % (q, float(np.percentile(g, q))))
        for thr in (2, 4, 8, 16, 40):
            print("  <= %2dpx 占 %.1f%%" % (thr, 100.0 * float((g <= thr).mean())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", default=r"Y:\workspace\EDA\work\stage1_all11")
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--tol", default="1,2,4,8,16",
                    help="端点合并容差（可逗号分隔）。**必须带小数点写 1.5**，int() 会截断。")
    ap.add_argument("--weld", default="0",
                    help="方向性桥接距离（可逗号分隔）；0 = 关闭。见 labels._weld_unions")
    ap.add_argument("--ext", default="0",
                    help="端点沿外法线朝前延伸距离（可逗号分隔）；0 = 关闭。见 labels._extend_unions")
    ap.add_argument("--ext-perp", type=float, default=2.0)
    ap.add_argument("--ext-cos", type=float, default=1.0,
                    help="<1 时要求被撞的段与射线大致垂直（T 型）")
    ap.add_argument("--ext-all", action="store_true",
                    help="关闭「只接正前方最近一条」，改为撞到几条接几条")
    ap.add_argument("--pin-tol", type=float, default=6.0)
    ap.add_argument("--split", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--source", default="stage1", choices=("stage1", "gt"),
                    help="gt = 对照组，用 GT 自己的 net_line 当线段")
    a = ap.parse_args()
    tols = [float(x) for x in a.tol.split(",") if x.strip()]
    welds = [float(x) for x in (a.weld or "0").split(",") if x.strip()]
    exts = [float(x) for x in (a.ext or "0").split(",") if x.strip()]
    run(a.stage1, a.data_root, tols, a.pin_tol, a.limit, a.split, a.source,
        welds, exts, a.ext_perp, a.ext_cos, not a.ext_all)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()
