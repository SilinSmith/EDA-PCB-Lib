# -*- coding: utf-8 -*-
"""网络分（`net_hypergraph_f1`，权重 **0.35** —— 四项里最重）到底卡在哪。

官方口径（`eda-pcb\\schematic_parser\\metrics.py:497-519` + `data.py:147-149`）：

  `net_pin_set(net)` = `parse_hypergraph(net['hyperGraph'])`，即网络成员**只看
  `hyperGraph` 这个字符串**（`"(U1.7,U1.8,U3.1)"`），**与 `pins` 段无关**。
  `match_nets` 对全部 pred/GT 网络做一次匈牙利匹配，然后
      `tp = Σ |p_set ∩ g_set|`（配对上的网络**共享的引用条数**）
      `fp = Σ|p_set| - tp`      `fn = Σ|g_set| - tp`

所以网络分只由两件事决定：**① 我们发出的 `(元件, 引脚号)` 引用有多少条是对的；
② 我们发出的条数够不够（召回）**。

本脚本把 GT 的每一条引用分成四类，决定下一步该修哪里：

  A. 我们**发出来了而且对**（= tp）
  B. 我方元件和引脚**都在**，只是这条引用**没被放进任何网络** ⇒ 网络归组/连边问题
  C. 我方元件在、但**没有这个引脚号** ⇒ 引脚识别问题（②关）
  D. 我方**根本没有这个元件键** ⇒ 位号/元件匹配问题

用法（workdir = Y:\\workspace\\EDA）：
    python work\\tools\\diag\\_netcover.py
    python work\\tools\\diag\\_netcover.py --pred work\\_fusion_out --top 12
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(r"Y:\workspace\EDA")
sys.path.insert(0, str(ROOT / "eda-pcb"))
sys.path.insert(0, str(ROOT / "work" / "tools" / "eval"))

sys.stdout.reconfigure(encoding="utf-8")

from schematic_parser.data import (DEFAULT_DATA_ROOT, load_json,          # noqa: E402
                                   net_pin_set, parse_hypergraph,
                                   pin_key_to_number)
from schematic_parser.metrics import (EvalConfig, assign_components,     # noqa: E402
                                      match_nets, rewrite_keys, _net_sets)
import stage2_eval as SE                                                 # noqa: E402

RUN = ROOT / "work" / "_submit_out220r36" / "runs"


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=str(RUN / "stage2"))
    ap.add_argument("--split", default="")
    ap.add_argument("--top", type=int, default=10)
    a = ap.parse_args(argv)

    exact, digits = SE.build_index(DEFAULT_DATA_ROOT)
    cfg = EvalConfig()
    files = [f for f in sorted(Path(a.pred).glob("*.json"))
             if not f.name.startswith("_")]

    st = collections.Counter()
    per_case = []
    samples = collections.defaultdict(list)
    refcnt_g = collections.Counter()
    refcnt_p = collections.Counter()

    for f in files:
        case, _how = SE.resolve(f.stem, exact, digits)
        if case is None or (a.split and case.split != a.split):
            continue
        gt = load_json(case.json_path)
        pr = json.loads(f.read_text(encoding="utf-8"))
        try:
            _f1, k2g = assign_components(gt, pr, cfg, {})
        except Exception:                                    # noqa: BLE001
            continue
        pr2 = rewrite_keys(pr, k2g)

        # 我方「元件键 -> 引脚号集合」（用 pin 段，不是 hyperGraph）
        padj = collections.defaultdict(set)
        for comp, grp in (pr2.get("pins") or {}).items():
            if isinstance(grp, dict):
                for pk in grp:
                    padj[comp].add(pin_key_to_number(pk))
        pcomp = set((pr2.get("components") or {}).keys())

        g_nets = list((gt.get("nets") or {}).items())
        p_nets = list((pr2.get("nets") or {}).items())
        g_sets = [(n, net_pin_set(v if isinstance(v, dict) else {})) for n, v in g_nets]
        p_sets = [(n, net_pin_set(v if isinstance(v, dict) else {})) for n, v in p_nets]
        pairs = match_nets(g_sets, p_sets)
        gt_in_pairs = {j for _i, j in pairs}

        # tp / fp / fn（复刻官方）
        tp = sum(len(p_sets[i][1] & g_sets[j][1]) for i, j in pairs)
        total_p = sum(len(s) for _n, s in p_sets)
        total_g = sum(len(s) for _n, s in g_sets)
        st["tp"] += tp
        st["fp"] += total_p - tp
        st["fn"] += total_g - tp
        st["pred_pins"] += total_p
        st["gt_pins"] += total_g
        st["pred_nets"] += len(p_sets)
        st["gt_nets"] += len(g_sets)
        st["paired"] += len(pairs)
        for _n, s in g_sets:
            refcnt_g[min(len(s), 6)] += 1
        for _n, s in p_sets:
            refcnt_p[min(len(s), 6)] += 1

        # 我方发出去、但和 GT 对不上的引用（精确 FP 来源）
        for i, j in pairs:
            for r in (p_sets[i][1] - g_sets[j][1]):
                key = "配对网络里多出来的引用"
                st[key] += 1
                if len(samples["fp"]) < a.top:
                    samples["fp"].append((case.case_id, p_sets[i][0], r,
                                          sorted(g_sets[j][1])[:5]))

        # ★ GT 的每一条引用按「我方有没有这个元件键 / 这个引脚号」四分类
        miss_comp = collections.Counter()
        for j, (n, s) in enumerate(g_sets):
            if j not in gt_in_pairs:
                st["GT网络没被配对"] += 1
            for comp, num in s:
                st["gt_refs"] += 1
                if comp not in pcomp:
                    st["D_元件键都没有"] += 1
                    miss_comp[comp] += 1
                elif num not in padj.get(comp, ()):
                    st["C_有元件没这个引脚号"] += 1
                elif j in gt_in_pairs:
                    pi = [i for i, jj in pairs if jj == j]
                    if pi and (comp, num) in p_sets[pi[0]][1]:
                        st["A_发出来了而且对"] += 1
                    else:
                        st["B_料齐了但没进这条网络"] += 1
                        if len(samples["B"]) < a.top:
                            samples["B"].append((case.case_id, n, (comp, num),
                                                 p_sets[pi[0]][0] if pi else None,
                                                 sorted(p_sets[pi[0]][1])[:6] if pi else []))
                else:
                    st["B_料齐了但网络没配上"] += 1

    def pct(n, d):
        return "%6.1f%%" % (100.0 * n / max(1, d))

    print("== 网络规模 ==")
    print("  GT   %5d 条网络 / %5d 条引用（均 %.1f）"
          % (st["gt_nets"], st["gt_pins"], st["gt_pins"] / max(1, st["gt_nets"])))
    print("  我方 %5d 条网络 / %5d 条引用（均 %.1f）"
          % (st["pred_nets"], st["pred_pins"], st["pred_pins"] / max(1, st["pred_nets"])))
    print("  匈牙利配对 %d 对" % st["paired"])
    print("\n== 官方口径复刻 ==")
    print("  tp %5d  fp %5d  fn %5d  ⇒ F1 %.4f"
          % (st["tp"], st["fp"], st["fn"],
             2.0 * st["tp"] / max(1, 2 * st["tp"] + st["fp"] + st["fn"])))
    print("\n== GT 每网络引用条数分布（0/1/2/3/4/5/6+） ==")
    print("  ", [refcnt_g[k] for k in range(7)])
    print("== 我方每网络引用条数分布 ==")
    print("  ", [refcnt_p[k] for k in range(7)])

    print("\n== GT 的 %d 条引用，我方处在哪一步 ==" % st["gt_refs"])
    print("  A 发出来了而且对          %5d  %s" % (st["A_发出来了而且对"],
                                                 pct(st["A_发出来了而且对"], st["gt_refs"])))
    print("  B 料齐了但没进这条网络    %5d  %s" % (st["B_料齐了但没进这条网络"],
                                                 pct(st["B_料齐了但没进这条网络"], st["gt_refs"])))
    print("  B' 料齐了但网络没配上     %5d  %s" % (st["B_料齐了但网络没配上"],
                                                 pct(st["B_料齐了但网络没配上"], st["gt_refs"])))
    print("  C 有元件、没这个引脚号    %5d  %s" % (st["C_有元件没这个引脚号"],
                                                 pct(st["C_有元件没这个引脚号"], st["gt_refs"])))
    print("  D 连元件键都没有          %5d  %s" % (st["D_元件键都没有"],
                                                 pct(st["D_元件键都没有"], st["gt_refs"])))
    print("     （其中 GT 网络根本没被配对 %d 条）" % st["GT网络没被配对"])
    print("  我方配对网络里多出来的引用 %5d" % st["配对网络里多出来的引用"])
    # ★ 把 B 和 B' 加起来就是「只差网络归组」的上界
    b = st["B_料齐了但没进这条网络"] + st["B_料齐了但网络没配上"]
    print("\n  ★ 只修「网络归组」的天花板：tp %d → %d，F1 %.4f → %.4f"
          % (st["tp"], st["tp"] + b,
             2.0 * st["tp"] / max(1, 2 * st["tp"] + st["fp"] + st["fn"]),
             2.0 * (st["tp"] + b) / max(1, 2 * (st["tp"] + b) + st["fp"] + st["fn"])))
    print("  ★ 连引脚一起修（B+C）的天花板：tp %d → %d，F1 %.4f → %.4f"
          % (st["tp"], st["tp"] + b + st["C_有元件没这个引脚号"],
             2.0 * st["tp"] / max(1, 2 * st["tp"] + st["fp"] + st["fn"]),
             2.0 * (st["tp"] + b + st["C_有元件没这个引脚号"])
             / max(1, 2 * (st["tp"] + b + st["C_有元件没这个引脚号"]) + st["fp"] + st["fn"])))

    for k, hdr in (("B", "B 类样例（case, GT网络, 缺的引用, 我方配上的网络, 我方该网络里的引用）"),
                   ("fp", "FP 类样例（case, 我方网络, 多出来的引用, GT配对网络的引用）")):
        if samples[k]:
            print("\n== %s ==" % hdr)
            for s in samples[k]:
                print("   %s" % (s,))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
