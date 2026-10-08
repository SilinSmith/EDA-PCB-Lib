# -*- coding: utf-8 -*-
"""_wirestage.py —— 导线失分按阶段拆解（只读现成产物，不跑 stage1）。

用法：
    python work\\tools\\diag\\_wirestage.py

产物：
    work\\_wirestage_out.txt   人类可读表
    work\\_wirestage.json      逐 case 明细

做三件事：
  A. 逐 case 打 net_line 的 tp/fp/fn + 结构（GT 边数 / 预测边数 / GT 网数 / 预测网数
     / 配上几对网 / 有多少 GT 边因为「网没配上」直接算 fn）。
  B. 把「配上的网对」里的失配边做**最近端点距离直方图**（1/2/5/10/20/40/∞ px），
     用来区分「长度差/端点桩偏移」和「整条错连」。
  C. 从 stage1 产物（`work\\_fast_out_reg9\\runs\\stage1\\*.stage1.json`、
     `work\\_full220\\runs\\stage1\\*.stage1.json`）读 `_wire_stat`（若存在）与
     net 清单，统计「有段但没引脚的网」——那批段的边永远进不了 edges。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[3]          # Y:\workspace\EDA
sys.path.insert(0, str(ROOT / "eda-pcb"))

from schematic_parser.data import DEFAULT_DATA_ROOT, SPLITS, discover, load_json  # noqa: E402
from schematic_parser.metrics import EvalConfig, _edge_points, _seg_distance, evaluate_case  # noqa: E402

EVAL = ROOT / "work" / "tools" / "eval"
GT9 = ["1", "2", "3", "4", "5", "7", "8", "9", "10"]
D19 = ["0092", "0048", "0019", "0024", "0086", "0042",
       "0075", "0083", "0090", "0057", "0062", "0032", "0028",
       "0175", "0198", "0105", "0106", "0036", "0120"]


def build_index(data_root):
    exact, digits = {}, {}
    for c in discover(data_root, SPLITS):
        if not c.json_path:
            continue
        exact.setdefault(c.case_id, c)
        d = re.sub(r"\D", "", c.case_id)
        if d:
            digits.setdefault(d, []).append(c)
    return exact, digits


def resolve(stem, exact, digits):
    if stem in exact:
        return exact[stem]
    if stem.endswith("_Pred") and stem[:-5] in exact:
        return exact[stem[:-5]]
    d = re.sub(r"\D", "", stem) or stem
    cands = digits.get(d) or []
    if len(cands) == 1:
        return cands[0]
    pure = [c for c in cands if c.case_id.isdigit()]
    if len(pure) == 1:
        return pure[0]
    return None


def strict_dist(pe, ge):
    """两份代码里同一判据：line_unordered 时两个端点取向取 min。"""
    return _seg_distance(pe[:2], ge[:2], True)


def analyse(case, pred_path, cfg):
    gt = load_json(case.json_path)
    pred = json.loads(Path(pred_path).read_text(encoding="utf-8"))
    s = evaluate_case(gt, pred, case.case_id, cfg)
    line = s.net_line

    # --- 复算一遍结构，拿 matched / unmatched 的分流 ---
    from schematic_parser.metrics import (match_nets, normalise_doc,
                                          assign_components, rewrite_keys, _net_sets)
    gtn, prd = normalise_doc(gt), normalise_doc(pred)
    _comp, keymap = assign_components(gtn, prd, cfg, {})
    prd2 = rewrite_keys(prd, keymap)
    g_nets = list(gtn["nets"].items())
    p_nets = list(prd2["nets"].items())
    pairs = match_nets(_net_sets(gtn["nets"]), _net_sets(prd2["nets"]))

    def n_edges(net):
        return len([v for v in (net.get("edges") or {}).values()
                    if _edge_points(v) is not None])

    gt_e = sum(n_edges(n) for _, n in g_nets)
    pr_e = sum(n_edges(n) for _, n in p_nets)
    matched_g = {j for _i, j in pairs}
    matched_p = {i for i, _j in pairs}
    fn_unmatched_net = sum(n_edges(g_nets[j][1]) for j in range(len(g_nets))
                           if j not in matched_g)
    fp_unmatched_net = sum(n_edges(p_nets[i][1]) for i in range(len(p_nets))
                           if i not in matched_p)

    # --- 直方图 + 全局最近距离（在已配对的网对内部） ---
    hist = {"<=1": 0, "<=2": 0, "<=5": 0, "<=10": 0, "<=20": 0, "<=40": 0, ">40": 0}
    tp_pairs = fp_pairs = fn_pairs = 0
    zero_net_pairs = 0
    for i, j in pairs:
        pe = [e for e in (_edge_points(v) for v in (p_nets[i][1].get("edges") or {}).values())
              if e is not None]
        ge = [e for e in (_edge_points(v) for v in (g_nets[j][1].get("edges") or {}).values())
              if e is not None]
        if not pe or not ge:
            zero_net_pairs += 1
            fp_pairs += len(pe)
            fn_pairs += len(ge)
            continue
        INF = float("inf")
        cost = [[(strict_dist(p, g) if strict_dist(p, g) <= cfg.line_tol else INF)
                 for g in ge] for p in pe]
        from schematic_parser.metrics import _assign
        got = set(_assign(cost))
        tp_pairs += len(got)
        fp_pairs += len(pe) - len(got)
        fn_pairs += len(ge) - len(got)
        # 直方图：每条 GT 边到最近预测边的最短严格距离（不设阈值）
        for gi, g in enumerate(ge):
            best = min((strict_dist(p, g) for p in pe), default=INF)
            for lim in (1, 2, 5, 10, 20, 40):
                if best <= lim:
                    hist["<=%d" % lim] += 1
                    break
            else:
                hist[">40"] += 1

    return {
        "case": case.case_id,
        "score": round(s.score, 4),
        "lineF1": round(line.f1, 4),
        "line_tp": line.tp, "line_fp": line.fp, "line_fn": line.fn,
        "compF1": round(s.component.f1, 4),
        "pinF1": round(s.pin.f1, 4),
        "netHF1": round(s.net_hypergraph.f1, 4),
        "gt_edges": gt_e, "pred_edges": pr_e,
        "gt_nets": len(g_nets), "pred_nets": len(p_nets),
        "matched_nets": len(pairs),
        "gt_nets_unmatched": len(g_nets) - len(matched_g),
        "pred_nets_unmatched": len(p_nets) - len(matched_p),
        "fn_from_unmatched_net": fn_unmatched_net,
        "fp_from_unmatched_net": fp_unmatched_net,
        "fn_inside_paired_net": fn_pairs,
        "fp_inside_paired_net": fp_pairs,
        "zero_edge_net_pairs": zero_net_pairs,
        "nearest_gt_edge_hist": hist,
    }


def wire_stat_from_stage1(stage1_dir, ids):
    """读 stage1 产物里的 `_wire_stat`（若版本带）+ net 清单。"""
    out = {}
    for i in ids:
        p = Path(stage1_dir) / ("%s.stage1.json" % i)
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        nets = d.get("nets") or []
        ws = d.get("_wire_stat") or d.get("wire_stat") or {}
        seg = d.get("_segments") or []
        out[i] = {
            "n_nets": len(nets),
            "n_nets_no_pin": sum(1 for n in nets if not (n.get("pins") or [])),
            "n_seg_in_nets": sum(len(n.get("seg_index") or []) for n in nets),
            "n_segments_field": len(seg),
            "wire_stat": ws,
            "keys": sorted(d.keys()),
        }
    return out


def main():
    cfg = EvalConfig()
    exact, digits = build_index(DEFAULT_DATA_ROOT)
    reg9 = ROOT / "work" / "_fast_out_reg9" / "final_results"
    f220 = ROOT / "work" / "_f220_A"

    rows = []
    for grp, ids, d in (("GT9", GT9, reg9), ("D19", D19, f220)):
        for i in ids:
            case = resolve(i, exact, digits)
            pj = Path(d) / ("%s_Pred.json" % i if grp == "GT9" else "%s.json" % i)
            if case is None or not pj.exists():
                rows.append({"case": i, "group": grp, "error": "no gt/pred: %s" % pj})
                continue
            r = analyse(case, pj, cfg)
            r["group"] = grp
            r["pred"] = str(pj)
            rows.append(r)

    # ---------- 打印 ----------
    lines = []
    hdr = ("%-6s %-5s %6s %6s %4s %4s %4s %6s %6s %5s %5s %5s %6s %6s %8s"
           % ("case", "grp", "score", "lineF1", "tp", "fp", "fn",
              "gtE", "prE", "gtN", "prN", "mtN", "fnNet", "fpNet", "zeroPair"))
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for r in rows:
        if "error" in r:
            lines.append("%-6s %-5s  ERROR %s" % (r["case"], r["group"], r["error"]))
            continue
        lines.append("%-6s %-5s %6.2f %6.4f %4d %4d %4d %6d %6d %5d %5d %5d %6d %6d %8d"
                     % (r["case"], r["group"], r["score"], r["lineF1"],
                        r["line_tp"], r["line_fp"], r["line_fn"],
                        r["gt_edges"], r["pred_edges"], r["gt_nets"], r["pred_nets"],
                        r["matched_nets"], r["fn_from_unmatched_net"],
                        r["fp_from_unmatched_net"], r["zero_edge_net_pairs"]))
    # 汇总
    for grp in ("GT9", "D19"):
        sub = [r for r in rows if r.get("group") == grp and "error" not in r]
        if not sub:
            continue
        tp = sum(r["line_tp"] for r in sub); fp = sum(r["line_fp"] for r in sub)
        fn = sum(r["line_fn"] for r in sub)
        f1 = 2 * tp / max(1e-9, 2 * tp + fp + fn)
        a = sum(r["fn_from_unmatched_net"] for r in sub)
        b = sum(r["fn_inside_paired_net"] for r in sub)
        c = sum(r["fp_from_unmatched_net"] for r in sub)
        dd = sum(r["fp_inside_paired_net"] for r in sub)
        ge = sum(r["gt_edges"] for r in sub); pe = sum(r["pred_edges"] for r in sub)
        lines.append("")
        lines.append("[%s] n=%d  net_line F1 %.4f  tp %d fp %d fn %d  | GT 边 %d 预测边 %d（%.2fx）"
                     % (grp, len(sub), f1, tp, fp, fn, ge, pe, pe / max(1, ge)))
        lines.append("      fn 归属：网没配上 %d（%.1f%%）｜网配上但边对不上 %d（%.1f%%）"
                     % (a, 100.0 * a / max(1, a + b), b, 100.0 * b / max(1, a + b)))
        lines.append("      fp 归属：网没配上 %d（%.1f%%）｜网配上但边多余 %d（%.1f%%）"
                     % (c, 100.0 * c / max(1, c + dd), dd, 100.0 * dd / max(1, c + dd)))
        hist = {}
        for r in sub:
            for k, v in r["nearest_gt_edge_hist"].items():
                hist[k] = hist.get(k, 0) + v
        tot = sum(hist.values()) or 1
        lines.append("      已配对网内 GT 边的「到最近预测边」距离分布: "
                     + "  ".join("%s=%d(%.0f%%)" % (k, hist.get(k, 0), 100.0 * hist.get(k, 0) / tot)
                                 for k in ("<=1", "<=2", "<=5", "<=10", "<=20", "<=40", ">40")))

    # ---------- stage1 侧 ----------
    s1a = wire_stat_from_stage1(ROOT / "work" / "_fast_out_reg9" / "runs" / "stage1", GT9)
    s1b = wire_stat_from_stage1(ROOT / "work" / "_full220" / "runs" / "stage1", D19)
    lines.append("")
    lines.append("=== stage1 产物侧（段/网） ===")
    for tag, s1 in (("GT9", s1a), ("D19", s1b)):
        for k, v in s1.items():
            lines.append("%-6s %-6s nets=%-4d 无引脚网=%-4d 网内段=%-5d _segments=%-5d ws=%s keys=%s"
                         % (tag, k, v["n_nets"], v["n_nets_no_pin"], v["n_seg_in_nets"],
                            v["n_segments_field"], json.dumps(v["wire_stat"], ensure_ascii=False)[:160],
                            ",".join(v["keys"])[:200]))

    txt = "\n".join(lines)
    (ROOT / "work" / "_wirestage_out.txt").write_text(txt, encoding="utf-8")
    (ROOT / "work" / "_wirestage.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
