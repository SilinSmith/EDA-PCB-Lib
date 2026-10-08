# -*- coding: utf-8 -*-
"""_wirezero.py —— 为什么「网配上了但预测侧零边」：把那些网对挖出来看。

对 GT9 逐 case：
  * 评测器配对后，pred 网 edges 数为 0 的网对 —— 它的 hyperGraph 引脚、GT 侧边数
  * 全预测文档里 edges 为空的网有几个、它们的 hyperGraph 是什么
  * 对照 stage1 里对应网（按引脚集合找）的 n_seg
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))
from schematic_parser.data import (DEFAULT_DATA_ROOT, SPLITS, discover, load_json,  # noqa: E402
                                   normalise_doc, net_pin_set, parse_hypergraph)
from schematic_parser.metrics import (EvalConfig, assign_components, match_nets,  # noqa: E402
                                      net_hypergraph_f1, rewrite_keys, _edge_points)

cfg = EvalConfig()
exact = {}
for c in discover(DEFAULT_DATA_ROOT, SPLITS):
    if c.json_path:
        exact.setdefault(c.case_id, c)

W = ROOT / "work"


def ne(net):
    return len([1 for v in (net.get("edges") or {}).values() if _edge_points(v) is not None])


for cid in ["1", "2", "4", "7", "9", "10", "3", "5", "8"]:
    gt = normalise_doc(load_json(exact[cid].json_path))
    pr_raw = json.loads((W / "_fast_out_reg9" / "final_results" / ("%s_Pred.json" % cid))
                        .read_text(encoding="utf-8"))
    pr = normalise_doc(pr_raw)
    _c, keymap = assign_components(gt, pr, cfg, {})
    pr2 = rewrite_keys(pr, keymap)
    _n, pairs = net_hypergraph_f1(gt, pr2, cfg, {})
    g_nets = list(gt["nets"].items())
    p_nets = list(pr2["nets"].items())
    empty_p = [(i, n) for i, (_k, n) in enumerate(p_nets) if ne(n) == 0]
    zero_pairs = 0
    print("=== case %s  预测网 %d（其中 0 边 %d） 配对 %d"
          % (cid, len(p_nets), len(empty_p), len(pairs)))
    for i, j in pairs:
        if ne(p_nets[i][1]) == 0:
            zero_pairs += 1
            gp = net_pin_set(g_nets[j][1])
            pp = net_pin_set(p_nets[i][1])
            print("   零边网对: gt_net %-8s(边%3d 脚%2d) ← pred net_%s(脚%2d) 交叠脚 %d"
                  % (g_nets[j][0], ne(g_nets[j][1]), len(gp),
                     p_nets[i][1].get("hyperGraph", "?") and p_nets[i][0], len(pp),
                     len(gp & pp)))
    # 预测里 0 边的网（无论有没有配上）
    for i, n in empty_p[:6]:
        print("   空边预测网 %-8s hyperGraph=%s" % (p_nets[i][0], n.get("hyperGraph")))
    # stage1 对应网
    s1 = json.loads((W / "_full220" / "runs" / "stage1" / ("%s.stage1.json" % cid))
                    .read_text(encoding="utf-8"))
    s1nets = s1.get("nets") or []
    nz = [n for n in s1nets if len(n.get("seg_index") or []) > 0]
    z = [n for n in s1nets if len(n.get("seg_index") or []) == 0]
    print("   stage1 网 %d（0 段 %d） 网内段合计 %d；输出文档 预测网 %d / 有边网 %d"
          % (len(s1nets), len(z), sum(len(n.get("seg_index") or []) for n in s1nets),
             len(pr_raw.get("nets") or {}),
             len([1 for n in (pr_raw.get("nets") or {}).values() if n.get("edges")])))
