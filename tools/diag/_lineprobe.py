# -*- coding: utf-8 -*-
"""为什么 NetLineF1 只有 0.019？

官方口径（`eda-pcb/schematic_parser/metrics.py`）：

* `net_line_f1`（`:522`）**只在 `match_nets` 配出来的对子上比边**。`match_nets`
  （`:477`）用「引脚集合交集」当相似度做匈牙利匹配，`_assign` 会返回**完整的**
  指派 —— 包括交集为 0 的瞎配。所以 `matched_*_nets` 几乎等于「全部网络」，
  配错的那些对子把双方所有边都算成 fp/fn。
* 边的比较（`:540-547`）：`_seg_distance(pe[:2], ge[:2], unordered=True)`，
  即**两条边的两个端点各自都要在 `line_tol=5.0` 像素内**（unordered 允许反向）。
  只取每条边的前两个点。

本脚本回答两个问题：

1. **配对的成色**：多少对子是真的（引脚交集 > 0）、多少是瞎配的，各贡献多少边。
2. **几何本身对不对**：把 GT 每条边拿去和**全图**我们所有边比（不限网络），
   最近距离的分布是什么样。如果这里也很差，那问题在 stage1 提线；
   如果这里很好、只是配对差，那问题在网络结构。

用法：
    python work\\tools\\diag\\_lineprobe.py --case 0001
    python work\\tools\\diag\\_lineprobe.py --case 0001 --pred work\\stage2_v14 --examples 3
"""
import argparse
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
EDA = r"Y:\workspace\EDA"
sys.path.insert(0, os.path.join(EDA, "eda-pcb"))

from schematic_parser import metrics as M  # noqa: E402
from schematic_parser.data import find_case, load_json  # noqa: E402


def _load_gt(case):
    c = find_case(case)
    if c is None or not c.json_path:
        raise SystemExit(f"找不到 GT: {case}")
    return load_json(c.json_path)


def _edges_of(net):
    out = []
    for v in (net.get("edges") or {}).values():
        p = M._edge_points(v)
        if p is not None and len(p) >= 2:
            out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="0001")
    ap.add_argument("--pred", default=os.path.join(EDA, "work", "stage2_v14"))
    ap.add_argument("--examples", type=int, default=0)
    a = ap.parse_args()

    gt = _load_gt(a.case)
    pred = load_json(os.path.join(a.pred, a.case + ".json"))
    if gt is None or pred is None:
        print("load failed", a.pred, a.case)
        return 1

    g_nets, p_nets = M._net_sets(gt["nets"]), M._net_sets(pred["nets"])
    pairs = M.match_nets(g_nets, p_nets)
    print(f"case {a.case}  gt nets {len(g_nets)}  pred nets {len(p_nets)}  pairs {len(pairs)}")

    # --- 1. 配对成色 ---------------------------------------------------------
    real = [0, 0, 0]      # (真实对数, 真实对里的 pred 边数, 真实对里的 gt 边数)
    blind = [0, 0, 0]
    for i, j in pairs:
        inter = len(p_nets[i][1] & g_nets[j][1])
        pe = _edges_of(pred["nets"][p_nets[i][0]])
        ge = _edges_of(gt["nets"][g_nets[j][0]])
        tgt = real if inter > 0 else blind
        tgt[0] += 1
        tgt[1] += len(pe)
        tgt[2] += len(ge)
    print(f"  真的配上（引脚有交集）: {real[0]:5d} 对, 携带 pred 边 {real[1]:6d} / gt 边 {real[2]:6d}")
    print(f"  瞎配（交集为 0）    : {blind[0]:5d} 对, 携带 pred 边 {blind[1]:6d} / gt 边 {blind[2]:6d}")

    # --- 2. 同对子内边的匹配质量 --------------------------------------------
    import math
    hit_in_pair = 0
    tot_in_pair = 0
    for i, j in pairs:
        if not len(p_nets[i][1] & g_nets[j][1]):
            continue
        pe = _edges_of(pred["nets"][p_nets[i][0]])
        ge = _edges_of(gt["nets"][g_nets[j][0]])
        if not pe or not ge:
            continue
        for g in ge:
            tot_in_pair += 1
            if any(M._seg_distance(p[:2], g[:2], True) <= 5.0 for p in pe):
                hit_in_pair += 1
    if tot_in_pair:
        print(f"  真实对子里 gt 边被命中: {hit_in_pair}/{tot_in_pair} = {hit_in_pair/tot_in_pair:.3f}")

    # --- 3. 几何本身（不限网络，拿全图所有 pred 边比） -----------------------
    all_pe = []
    for _, net in pred["nets"].items():
        all_pe.extend(_edges_of(net))
    all_ge = []
    for _, net in gt["nets"].items():
        all_ge.extend(_edges_of(net))
    print(f"  全图 pred 边 {len(all_pe)}  gt 边 {len(all_ge)}")
    dists = []
    for g in all_ge:
        best = float("inf")
        for p in all_pe:
            d = M._seg_distance(p[:2], g[:2], True)
            if d < best:
                best = d
        dists.append(best)
    if dists:
        ds = sorted(d for d in dists if d != float("inf"))
        n = len(ds)
        hit5 = sum(1 for d in ds if d <= 5.0)
        print(f"  不限网络、最近边距: <=5px {hit5}/{len(dists)} = {hit5/len(dists):.3f}"
              f"   中位 {ds[n//2]:.1f}  p25 {ds[n//4]:.1f}  p75 {ds[3*n//4]:.1f}  max {ds[-1]:.1f}")
        hist = Counter()
        for d in ds:
            hist[min(7, int(d // 10))] += 1
        print("     直方（每格 10px，最后一格 >=70）: "
              + "  ".join(f"{k*10}+:{hist.get(k,0)}" for k in range(8)))

    # --- 4. 双边样例 ---------------------------------------------------------
    if a.examples:
        print("\n  ---- 真实对子样例（gt 边 -> 最接近的 pred 边） ----")
        shown = 0
        for i, j in pairs:
            if not len(p_nets[i][1] & g_nets[j][1]):
                continue
            pe = _edges_of(pred["nets"][p_nets[i][0]])
            ge = _edges_of(gt["nets"][g_nets[j][0]])
            if not pe or not ge:
                continue
            # 只挑「配得上但对不上」的
            for g in ge:
                if any(M._seg_distance(p[:2], g[:2], True) <= 5.0 for p in pe):
                    continue
                cand = sorted(pe, key=lambda p: M._seg_distance(p[:2], g[:2], True))[:2]
                fs = " | ".join(
                    f"({c[0][0]:.0f},{c[0][1]:.0f})->({c[1][0]:.0f},{c[1][1]:.0f})"
                    f" d={M._seg_distance(c[:2], g[:2], True):.1f}" for c in cand)
                print(f"    {g_nets[j][0]:>12s} pins={sorted(g_nets[j][1])[:4]}"
                      f"  GT ({g[0][0]:.0f},{g[0][1]:.0f})->({g[1][0]:.0f},{g[1][1]:.0f})"
                      f"   pred: {fs}")
                shown += 1
                break
            if shown >= a.examples:
                break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
