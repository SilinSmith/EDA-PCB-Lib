# -*- coding: utf-8 -*-
"""_wiremiss.py —— 逐 case 追「边到底在哪一步消失」+ 6 张爆 0 拆解。

阶段（全部用现成产物，不跑 stage1）：
  A stage1 线段数 / 网内段
  B stage1 网（有引脚）
  C stage2 网（assemble 的 `if not refs: continue` 在这一步丢网）
  D fusion / final_results 的网与边
  E 官方口径 net_line 的 tp/fp/fn，并把 fn 拆成：
      E1 GT 网在预测里没有对应网（网没配上）
      E2 网配上了、但该 GT 网上没有任何预测边
      E3 网配上了、也有预测边，但端点距离 > 5px（几何对不上）
"""
import json
import math
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))
from schematic_parser.data import (DEFAULT_DATA_ROOT, SPLITS, discover,  # noqa: E402
                                   load_json, normalise_doc, net_pin_set)
from schematic_parser.metrics import (EvalConfig, _edge_points,  # noqa: E402
                                      _seg_distance, _assign, assign_components,
                                      match_nets, net_hypergraph_f1, rewrite_keys)

W = ROOT / "work"
cfg = EvalConfig()
exact = {}
for c in discover(DEFAULT_DATA_ROOT, SPLITS):
    if c.json_path:
        exact.setdefault(c.case_id, c)


def n_edges_in(doc_nets):
    return sum(len([1 for v in (n.get("edges") or {}).values()
                    if _edge_points(v) is not None]) for n in doc_nets.values())


def edges_of(net):
    return [e for e in (_edge_points(v) for v in (net.get("edges") or {}).values())
            if e is not None]


def bucket(cid, pred_path, s1_path, s2_path, out_path):
    gt_raw = load_json(exact[cid].json_path)
    pred_raw = json.loads(Path(out_path).read_text(encoding="utf-8"))
    gt, pred = normalise_doc(gt_raw), normalise_doc(pred_raw)
    comp, keymap = assign_components(gt, pred, cfg, {})
    pred2 = rewrite_keys(pred, keymap)
    _net, pairs = net_hypergraph_f1(gt, pred2, cfg, {})
    g_nets = list(gt["nets"].items())
    p_nets = list(pred2["nets"].items())
    matched_g = {j for _i, j in pairs}
    matched_p = {i for i, _j in pairs}

    e1 = e2 = e3 = 0
    for j, (_n, gnet) in enumerate(g_nets):
        ge = edges_of(gnet)
        if j not in matched_g:
            e1 += len(ge)
            continue
        i = [i for i, jj in pairs if jj == j][0]
        pe = edges_of(p_nets[i][1])
        if not pe:
            e2 += len(ge)
            continue
        # 该 GT 网内部：几何对不上的边数（用与评测器同样的分配）
        INF = float("inf")
        cost = [[(min(_seg_distance(p[:2], g[:2], True) for p in pe) if False else
                  (_seg_distance(p[:2], g[:2], True)
                   if _seg_distance(p[:2], g[:2], True) <= cfg.line_tol else INF))
                 for g in ge] for p in pe]
        got = set(_assign(cost))
        e3 += len(ge) - len(got)

    s1 = json.loads(Path(s1_path).read_text(encoding="utf-8"))
    s2 = json.loads(Path(s2_path).read_text(encoding="utf-8"))
    s1nets = s1.get("nets") or []
    s2nets = s2.get("nets") or {}

    # stage2 丢网的原因统计
    cids_in_s1 = {c for n in s1nets for c, _ in (n.get("pins") or [])}
    cids_mapped = sum(1 for c in cids_in_s1 if c in keymap)
    return {
        "case": cid,
        "gt_nets": len(g_nets), "gt_edges": n_edges_in(gt["nets"]),
        "pred_nets": len(p_nets), "pred_edges": n_edges_in(pred2["nets"]),
        "pairs": len(pairs),
        "s1_nets": len(s1nets), "s1_segIn": sum(len(n.get("seg_index") or []) for n in s1nets),
        "s2_nets": len(s2nets), "s2_edges": n_edges_in(s2nets),
        "out_nets": len(pred_raw.get("nets") or {}), "out_edges": n_edges_in(pred_raw.get("nets") or {}),
        "s1_cids": len(cids_in_s1), "s1_cids_keymapped": cids_mapped,
        "comp_pred": len(pred_raw.get("components") or {}),
        "comp_keyhit": len([k for k in (pred_raw.get("components") or {}) if k in
                            set((gt_raw.get("components") or {}).keys())]),
        "fn_e1_unmatched_net": e1,
        "fn_e2_matched_net_no_pred_edge": e2,
        "fn_e3_geometric": e3,
    }


GT9 = ["1", "2", "3", "4", "5", "7", "8", "9", "10"]
D19 = ["0092", "0048", "0019", "0024", "0086", "0042",
       "0075", "0083", "0090", "0057", "0062", "0032", "0028",
       "0175", "0198", "0105", "0106", "0036", "0120"]

rows = []
for cid, grp, base in ([("1", "GT9", W / "_fast_out_reg9")] +
                       [(c, "D19", W / "_full220") for c in D19]):
    pass

jobs = []
for c in GT9:
    jobs.append((c, "GT9", W / "_fast_out_reg9" / "runs" / "stage1" / ("%s.stage1.json" % c),
                 W / "_fast_out_reg9" / "runs" / "stage2" / ("%s.json" % c),
                 W / "_fast_out_reg9" / "final_results" / ("%s_Pred.json" % c)))
for c in D19:
    jobs.append((c, "D19", W / "_full220" / "runs" / "stage1" / ("%s.stage1.json" % c),
                 W / "_full220" / "runs" / "stage2" / ("%s.json" % c),
                 W / "_f220_A" / ("%s.json" % c)))

for cid, grp, s1p, s2p, op in jobs:
    if cid not in exact or not op.exists():
        print("skip", cid)
        continue
    try:
        r = bucket(cid, None, s1p, s2p, op)
    except Exception as e:                                    # noqa: BLE001
        import traceback
        print("ERR", cid, e)
        traceback.print_exc()
        continue
    r["grp"] = grp
    rows.append(r)

hdr = ("%-6s %-4s | %4s %5s | %4s %5s | %4s %5s | %4s %5s | %5s | %6s %6s %6s | %5s/%5s"
       % ("case", "grp", "gtN", "gtE", "prN", "prE", "s1N", "segIn", "s2N", "s2E",
          "outE", "fnNet", "fnNoEdge", "fnGeom", "keys", "cids"))
lines = [hdr, "-" * len(hdr)]
for r in rows:
    lines.append("%-6s %-4s | %4d %5d | %4d %5d | %4d %5d | %4d %5d | %5d | %6d %6d %6d | %5d/%5d"
                 % (r["case"], r["grp"], r["gt_nets"], r["gt_edges"], r["pred_nets"],
                    r["pred_edges"], r["s1_nets"], r["s1_segIn"], r["s2_nets"], r["s2_edges"],
                    r["s2_edges"] if not r["out_edges"] else r["out_edges"],
                    r["fn_e1_unmatched_net"], r["fn_e2_matched_net_no_pred_edge"],
                    r["fn_e3_geometric"], r["comp_keyhit"], r["comp_pred"]))
txt = "\n".join(lines)
(ROOT / "work" / "_wiremiss_out.txt").write_text(txt, encoding="utf-8")
(ROOT / "work" / "_wiremiss.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1),
                                              encoding="utf-8")
print(txt)
print()
for grp in ("GT9", "D19"):
    sub = [r for r in rows if r["grp"] == grp]
    if not sub:
        continue
    tot = {k: sum(r[k] for r in sub) for k in
           ("gt_edges", "pred_edges", "fn_e1_unmatched_net",
            "fn_e2_matched_net_no_pred_edge", "fn_e3_geometric", "s1_segIn", "s2_edges", "out_edges")}
    s = sum(sub[0].keys() and [r["fn_e1_unmatched_net"] + r["fn_e2_matched_net_no_pred_edge"]
                               + r["fn_e3_geometric"] for r in sub])
    print("[%s] GT 边 %d ｜ 输出边 %d（%.2fx）｜ s1 网内段 %d → s2 边 %d → 输出边 %d"
          % (grp, tot["gt_edges"], tot["pred_edges"], tot["pred_edges"] / max(1, tot["gt_edges"]),
             tot["s1_segIn"], tot["s2_edges"], tot["out_edges"]))
    print("     fn 三段拆分：网没配上 %d ｜ 网配上但预测零边 %d ｜ 几何 >5px %d（合计 %d）"
          % (tot["fn_e1_unmatched_net"], tot["fn_e2_matched_net_no_pred_edge"],
             tot["fn_e3_geometric"], s))
    print("     元件键命中：%d/%d" % (sum(r["comp_keyhit"] for r in sub), sum(r["comp_pred"] for r in sub)))
