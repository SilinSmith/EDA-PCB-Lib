# -*- coding: utf-8 -*-
"""_nethloss.py —— netH（`net_hypergraph_f1`，权重 0.35）的**亏损分解 + 可达上界**（§5.128 第 0 步）。

动机（父 agent m01378）：仓库里有一条旧结论说 netH 从 35x0.1790~6.27 的表观亏空里
「只有 ~+0.0112（~+0.39 分）是真实可达的」。本工具用**官方判分器**把它做成可达性上界。

官方口径（`schematic_parser/metrics.py:547-585`，已逐行核对）：
  * `_net_sets(nets, components, cfg.net_pin_collapse_nonbox=False)` —— 每条网的
    `(comp, pin)` 字面集合（**不**折算非 box 引脚号）。
  * `match_nets(g, p, soft=True)` —— 以 Dice `2|P&G|/(|P|+|G|)` 为目标最大化的
    **一对一匈牙利**。
  * `soft_tp = sum(Dice)`；`f1 = 2*soft_tp/(N_pred + N_gt)`。
  ★ 由最后一式可知：**分母只数网条数，分子却是 Dice 之和** ⇒ 「网被切碎」会被罚两次
    （N_pred 涨 + soft_tp 跌）；而**网的配对已由判分器自己最优化** ⇒ 「网络级配对失败」
    在 netH 里**不是独立可修的错误类**，它只是成员集合错误的症状（本工具会实测确认）。

分类（父 agent 指定的五类）：
  (1) 不可修      GT 成员落在**我方任何网里都没有**的 `V \\ U`（看不出来的成员）
  (2) 网络合并失败 多条我方网都被同一条 GT 网吸附（`argmax` 相同）——该并的没并
  (3) 网络切分失败 一条我方网同时压在 >=2 条 GT 网上——该分的没分
  (4) 成员错      我方网里的**凭空成员** `P_i \\ V`（不属于任何 GT 网）
  (5) 配对失败     匈牙利给出的对不是 `argmax`；以及两边各自剩下的未配对网

每类给出**条数**与**只修这一类的神谕增益**（在已交付产物上离线重算，不跑管线）。

可达上界（判据写死：< +0.02 就止损）：
  * `O_full`     —— 神谕：我方网 = GT 网（成员 = GT 成员），**上界中的上界**。
  * `O_full_U`   —— 神谕但**只能重排已有成员**（成员池 `U = ∪ 我方网成员`），
                    与 `O_full` 的差 = 「多拿到成员」的价值（那属于 pin 通道）。
  * `O_merge_same` —— **完全不用 GT**：把**同名**的我方网并起来（唯一的现成 GT-free 信号）。
  * `O_merge_orc` / `O_split_orc` / `O_member` —— 各自的单类神谕。

用法：
  python -X utf8 tools/diag/_nethloss.py --run-dir ..\\work\\_fast_out_delivery_default
  python -X utf8 tools/diag/_nethloss.py --run-dir ..\\work\\_fast_out_deterministic_default --out ..\\work\\_nethloss_det.json
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from schematic_parser.data import (discover, DEFAULT_DATA_ROOT,          # noqa: E402
                                   pin_key_to_number)
from schematic_parser.metrics import (EvalConfig, _net_sets,            # noqa: E402
                                      _net_sim, assign_components,
                                      match_nets, normalise_doc,
                                      rewrite_keys)

DEFAULT_RUN = "Y:/workspace/EDA/work/_fast_out_delivery_default"
CASES10 = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]


# ------------------------------------------------------------------ 打分核

def sc(psets, g_nets, compact=False):
    """给定我方网集合列表，按官方口径算 `(f1, soft, pairs, n_pred)`。

    `psets` = [(name, set), ...]；`compact=True` 时丢掉成员集合为空的网
    （官方保留空网 ⇒ 只多扣分母，所以只有神谕变体才 compact）。
    """
    pn = [(n, frozenset(s)) for n, s in psets]
    if compact:
        pn = [(n, s) for n, s in pn if s]
    pairs = match_nets(g_nets, pn, soft=True)
    soft = sum(_net_sim(pn[i][1], g_nets[j][1], True) for i, j in pairs)
    f1 = (2.0 * soft / (len(pn) + len(g_nets))) if (pn or g_nets) else 0.0
    return f1, soft, pairs, len(pn)


def dice(a, b):
    u = len(a) + len(b)
    return (2.0 * len(a & b) / u) if u else 0.0


# ------------------------------------------------------------------ 主分析

def analyse_case(gt_doc, pred_doc, cfg):
    gt, pred = normalise_doc(gt_doc), normalise_doc(pred_doc)
    comp, keymap = assign_components(gt, pred, cfg, {})
    pred2 = rewrite_keys(pred, keymap)

    g_nets = _net_sets(gt["nets"], gt.get("components"), cfg.net_pin_collapse_nonbox)
    p_nets = _net_sets(pred2["nets"], pred2.get("components"), cfg.net_pin_collapse_nonbox)
    NG, NP = len(g_nets), len(p_nets)
    if not NG or not NP:
        return None

    base_f1, base_soft, pairs, _ = sc([(n, s) for n, s in p_nets], g_nets)

    U = set()                      # 我方成员池：出现在**任何**我方网里的成员
    for _, s in p_nets:
        U |= s
    V = set()                      # GT 成员池
    for _, s in g_nets:
        V |= s

    # argmax：每条我方网最像哪条 GT 网；touched：它压在哪些 GT 网上
    argmax, dmax = {}, {}
    for i, (_, ps) in enumerate(p_nets):
        best, bd = -1, 0.0
        for j, (_, gs) in enumerate(g_nets):
            d = dice(ps, gs)
            if d > bd:
                best, bd = j, d
        argmax[i], dmax[i] = best, bd
    touched = [(i, {j for j, (_, gs) in enumerate(g_nets) if (ps & gs)})
               for i, (_, ps) in enumerate(p_nets)]

    pair_of_pred = {i: j for i, j in pairs}
    paired_pred, paired_gt = set(pair_of_pred), {j for _, j in pairs}

    # ---- 五类计数
    unfix = V - U                                        # (1)
    merge_group = {}
    for i in range(NP):
        if argmax[i] >= 0:
            merge_group.setdefault(argmax[i], []).append(i)
    merge_groups = [v for v in merge_group.values() if len(v) >= 2]
    merge_extra_nets = sum(len(v) - 1 for v in merge_groups)
    merge_members = sum(sum(len(p_nets[i][1]) for i in v[1:]) for v in merge_groups)
    merge_same_name = [v for v in merge_groups
                       if len({(p_nets[i][0] or "") for i in v}) == 1]
    merge_anydup_name = [v for v in merge_groups
                         if len({(p_nets[i][0] or "") for i in v}) < len(v)]
    merge_detail = [{"gt_net": g_nets[j][0],
                     "gt_pins": sorted("%s.%s" % (c, p) for c, p in g_nets[j][1]),
                     "pred": [{"name": p_nets[i][0], "n": len(p_nets[i][1]),
                               "pins": sorted("%s.%s" % (c, p) for c, p in p_nets[i][1])}
                              for i in v]}
                    for j, v in merge_group.items() if len(v) >= 2]
    split_nets = [i for i, t in touched if len(t) >= 2]
    split_members = sum(len(p_nets[i][1]) for i in split_nets)
    split_detail = [{"pred": p_nets[i][0],
                     "n": len(p_nets[i][1]),
                     "pins": sorted("%s.%s" % (c, p) for c, p in p_nets[i][1]),
                     "gt": [{"net": g_nets[j][0], "live": len(p_nets[i][1] & g_nets[j][1])}
                            for j in sorted(touched[i][1])]}
                    for i in split_nets]
    extra_members = sum(len(p_nets[i][1] - V) for i in range(NP))
    nonargmax = [(i, j) for i, j in pairs if argmax[i] != j]
    pred_unmatched = [i for i in range(NP) if i not in paired_pred]
    gt_unmatched = [j for j in range(NG) if j not in paired_gt]
    gt_unm_reach = [j for j in gt_unmatched
                    if any(dice(p_nets[i][1], g_nets[j][1]) > 0 for i in range(NP))]
    # 我方**检出**的引脚全域（不管有没有进网）——修正引脚键命名空间
    W = set(U)
    for compk, grp in (pred2.get("pins") or {}).items():
        if isinstance(grp, dict):
            for pk in grp:
                W.add((compk, pin_key_to_number(pk)))
    orphan_pins = W - U
    # GT 里成员集合为空的网（永远拿不到 Dice）
    gt_empty = [j for j, (_, s) in enumerate(g_nets) if not s]
    # (1) 的关键切分：缺的成员里「我方可检出的引脚」vs「我方压根没有的引脚」
    miss_detected = (V - U) & W
    miss_undetected = (V - U) - W

    # ---- 单类神谕
    orc = {}
    orc["O_full"] = sc([(n, s) for n, s in g_nets], g_nets, compact=True)[0]
    orc["O_full_U"] = sc([(g_nets[j][0], g_nets[j][1] & U) for j in range(NG)],
                         g_nets, compact=True)[0]
    # ★ 中间档：成员池放宽到「我方检出的所有引脚」W —— 即「引脚检测不动、只把网重挂对」
    orc["O_full_W"] = sc([(g_nets[j][0], g_nets[j][1] & W) for j in range(NG)],
                         g_nets, compact=True)[0]
    # 只用「同名」信号的并网（完全 GT-free）
    byname = {}
    for i, (n, s) in enumerate(p_nets):
        byname.setdefault(n or "", []).append(s)
    merged_same = [(n, set().union(*v)) for n, v in byname.items()]
    orc["O_merge_same"] = sc(merged_same, g_nets)[0]
    # 神谕并网：同一个 argmax 组的并成一条；**单例组与 argmax<0 的网原样保留**
    mg = dict(merge_group)
    _neg = [i for i in range(NP) if argmax[i] < 0]
    if _neg:                       # ★ 必须判空：setdefault(-1, []) 会凭空造一条空网 +1 分母
        mg[-1] = _neg
    orc["O_merge_orc"] = sc([(p_nets[v[0]][0] if len(v) == 1 else "*",
                              set(p_nets[v[0]][1]) if len(v) == 1
                              else set().union(*[p_nets[i][1] for i in v]))
                             for v in mg.values()],
                            g_nets)[0]
    # ★ 另一类独立错误：**凭空网**（成员全部不属于任何 GT 网）——「该删的没删」
    junk_nets = [i for i in range(NP) if p_nets[i][1] and not (p_nets[i][1] & V)]
    orc["O_drop_junk"] = sc([(n, s) for i, (n, s) in enumerate(p_nets)
                             if i not in set(junk_nets)], g_nets)[0]
    # 神谕切分：每条我方网按 GT 成员切；剩渣单独成网
    split_psets = []
    for i, (n, ps) in enumerate(p_nets):
        cut = False
        for j, (_, gs) in enumerate(g_nets):
            piece = ps & gs
            if piece and piece != ps:
                split_psets.append((n, piece))
                cut = True
        if cut:
            junk = ps - V
            if junk:
                split_psets.append((n, junk))
        else:
            split_psets.append((n, ps))
    orc["O_split_orc"] = sc(split_psets, g_nets)[0]
    # 神谕成员：配对上的网成员直接换成 GT 集合；未配对的网原样留着（保守）
    mem_psets = []
    for i, (n, ps) in enumerate(p_nets):
        mem_psets.append((n, g_nets[pair_of_pred[i]][1] if i in pair_of_pred else ps))
    orc["O_member"] = sc(mem_psets, g_nets)[0]

    return {
        "case": "", "NG": NG, "NP": NP, "base_f1": base_f1, "base_soft": base_soft,
        "n_pairs": len(pairs),
        "c1_unfix_members": len(unfix),
        "c1_gt_members": len(V),
        "c1_miss_detected": len(miss_detected),
        "c1_miss_undetected": len(miss_undetected),
        "w_pins": len(W),
        "c1_nets_unobservable": sum(1 for _, s in g_nets if not (s & U)),
        "c2_merge_groups": len(merge_groups),
        "c2_merge_extra_nets": merge_extra_nets,
        "c2_merge_members": merge_members,
        "c2_same_name_groups": len(merge_same_name),
        "c2_anydup_name_groups": len(merge_anydup_name),
        "merge_detail": merge_detail,
        "split_detail": split_detail,
        "c3_split_nets": len(split_nets),
        "c3_split_members": split_members,
        "c4_extra_members": extra_members,
        "c4_junk_nets": len(junk_nets),
        "junk_detail": [{"name": p_nets[i][0], "n": len(p_nets[i][1]),
                         "pins": sorted("%s.%s" % (c, p) for c, p in p_nets[i][1])}
                        for i in junk_nets],
        "c5_nonargmax_pairs": len(nonargmax),
        "c5_pred_unmatched": len(pred_unmatched),
        "c5_gt_unmatched": len(gt_unmatched),
        "c5_gt_unmatched_reachable": len(gt_unm_reach),
        "orphan_pins": len(orphan_pins),
        "gt_empty_nets": len(gt_empty),
        "orc": orc,
    }


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=DEFAULT_RUN)
    ap.add_argument("--cases", default=",".join(CASES10))
    ap.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    ap.add_argument("--out", default="")
    ap.add_argument("--show", action="store_true", help="打印该并/该分的网明细（判 GT-free 信号）")
    a = ap.parse_args()

    index = {}
    for c in discover(a.data_root):
        index.setdefault(str(c.case_id), c)

    cfg = EvalConfig()
    rows = []
    for cid in [s.strip() for s in a.cases.split(",") if s.strip()]:
        case = index.get(str(cid))
        if case is None:
            print("!! case %s not found" % cid)
            continue
        pp = os.path.join(a.run_dir, "final_results", "%s_Pred.json" % cid)
        if not os.path.exists(pp):
            print("!! missing %s" % pp)
            continue
        gt_doc = json.load(io.open(case.json_path, encoding="utf-8"))
        pred_doc = json.load(io.open(pp, encoding="utf-8"))
        r = analyse_case(gt_doc, pred_doc, cfg)
        if r is None:
            continue
        r["case"] = cid
        rows.append(r)

    if not rows:
        print("no rows")
        return 1

    n = len(rows)

    def avg(k):
        return sum(r[k] for r in rows) / float(n) if n else 0.0

    print("run-dir = %s   cases = %d" % (a.run_dir, n))
    print()
    print("== 基线（官方口径，逐案均值） ==")
    print("netH f1 = %.4f   soft_tp = %.4f   N_gt = %.1f   N_pred = %.1f   配对 = %.1f"
          % (avg("base_f1"), avg("base_soft"), avg("NG"), avg("NP"), avg("n_pairs")))
    print()

    hdr = ("case  netH     NG   NP  pairs |  c1unf c1det c1und  gtmemb | c2grp c2same"
           " c2net c2mem | c3net c3mem | c4ext | c5na c5pun c5gun |"
           " orphan  wpin gtmpty")
    print("== 五类计数（逐案；c1det=缺的成员里我方**检出过**的引脚，c1und=压根没检出） ==")
    print(hdr)
    for r in rows:
        print("%-5s %5.4f %4d %4d %5d | %6d %5d %5d %7d | %5d %6d %5d %5d | %5d %5d |"
              " %5d | %4d %5d %5d | %6d %5d %6d"
              % (r["case"], r["base_f1"], r["NG"], r["NP"], r["n_pairs"],
                 r["c1_unfix_members"], r["c1_miss_detected"], r["c1_miss_undetected"],
                 r["c1_gt_members"],
                 r["c2_merge_groups"], r["c2_same_name_groups"],
                 r["c2_merge_extra_nets"], r["c2_merge_members"],
                 r["c3_split_nets"], r["c3_split_members"], r["c4_extra_members"],
                 r["c5_nonargmax_pairs"], r["c5_pred_unmatched"], r["c5_gt_unmatched"],
                 r["orphan_pins"], r["w_pins"], r["gt_empty_nets"]),
              end="")
        print("  junk=%d" % r["c4_junk_nets"])
    print("合计/均值  %5.4f %4.1f %4.1f %5.1f | %6.1f %5.1f %5.1f %7.1f | %5.1f %6.1f"
          " %5.1f %5.1f | %5.1f %5.1f | %5.1f | %4.1f %5.1f %5.1f | %6.1f %5.1f %6.1f"
          % (avg("base_f1"), avg("NG"), avg("NP"), avg("n_pairs"),
             avg("c1_unfix_members"), avg("c1_miss_detected"), avg("c1_miss_undetected"),
             avg("c1_gt_members"),
             avg("c2_merge_groups"), avg("c2_same_name_groups"),
             avg("c2_merge_extra_nets"), avg("c2_merge_members"),
             avg("c3_split_nets"), avg("c3_split_members"), avg("c4_extra_members"),
             avg("c5_nonargmax_pairs"), avg("c5_pred_unmatched"), avg("c5_gt_unmatched"),
             avg("orphan_pins"), avg("w_pins"), avg("gt_empty_nets")))
    print()

    print("== 单类神谕（逐案均值 f1，以及增益 d = O - base） ==")
    keys = ["O_full", "O_full_W", "O_full_U", "O_merge_same", "O_merge_orc",
            "O_split_orc", "O_drop_junk", "O_member"]
    print("case  base   " + "".join("%12s" % k for k in keys))
    for r in rows:
        print("%-5s %6.4f " % (r["case"], r["base_f1"])
              + "".join("%+12.4f" % (r["orc"][k] - r["base_f1"]) for k in keys))
    print("%-5s %6.4f " % ("MEAN", avg("base_f1"))
          + "".join("%+12.4f" % (sum(r["orc"][k] for r in rows) / n - avg("base_f1"))
                    for k in keys))
    print()
    print("绝对值：")
    for k in keys:
        print("  %-16s %.4f" % (k, sum(r["orc"][k] for r in rows) / n))
    print()
    d_full = sum(r["orc"]["O_full"] for r in rows) / n - avg("base_f1")
    d_fullW = sum(r["orc"]["O_full_W"] for r in rows) / n - avg("base_f1")
    d_fullU = sum(r["orc"]["O_full_U"] for r in rows) / n - avg("base_f1")
    d_nm = sum(r["orc"]["O_merge_same"] for r in rows) / n - avg("base_f1")
    d_mg = sum(r["orc"]["O_merge_orc"] for r in rows) / n - avg("base_f1")
    d_sp = sum(r["orc"]["O_split_orc"] for r in rows) / n - avg("base_f1")
    d_jk = sum(r["orc"]["O_drop_junk"] for r in rows) / n - avg("base_f1")
    print("★ 可达性判据（父 agent 写死：< +0.02 即止损）")
    print("  神谕全上界 O_full        d = %+.4f  (%.2f 分, 权重 0.35)" % (d_full, d_full * 35))
    print("  + 引脚检出不动(池=W)     d = %+.4f  (%.2f 分)  <- 净挂接可达" % (d_fullW, d_fullW * 35))
    print("  + 只重排已有网成员(U)    d = %+.4f  (%.2f 分)  <- 纯重路由可达" % (d_fullU, d_fullU * 35))
    print("  ---- 单类神谕（互不重叠地分类） ----")
    print("  ② 该并的神谕并网         d = %+.4f  (%.2f 分)" % (d_mg, d_mg * 35))
    print("  ③ 该分的神谕切分         d = %+.4f  (%.2f 分)" % (d_sp, d_sp * 35))
    print("  ④ 删凭空网               d = %+.4f  (%.2f 分)" % (d_jk, d_jk * 35))
    print("  GT-free 同名并网         d = %+.4f  (%.2f 分)" % (d_nm, d_nm * 35))

    if a.out:
        io.open(a.out, "w", encoding="utf-8").write(
            json.dumps(rows, ensure_ascii=False, indent=1))
        print("\njson -> %s" % a.out)

    if a.show:
        print("\n== 该并/该分明细（判「不看 GT 能不能判」） ==")
        for r in rows:
            if not r["merge_detail"] and not r["split_detail"]:
                continue
            print("\n--- case %s (netH %.4f) ---" % (r["case"], r["base_f1"]))
            for m in r["merge_detail"]:
                print("  [该并] GT网 %r 成员 %s" % (m["gt_net"], m["gt_pins"]))
                for p in m["pred"]:
                    print("         我方网 %-10r n=%d  %s" % (p["name"], p["n"], p["pins"]))
            for s in r["split_detail"]:
                print("  [该分] 我方网 %-10r n=%d  %s" % (s["pred"], s["n"], s["pins"]))
                for g in s["gt"]:
                    print("         压住 GT网 %-10r 重合 %d" % (g["net"], g["live"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
