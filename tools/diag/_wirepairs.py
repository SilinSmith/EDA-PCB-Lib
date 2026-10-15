# -*- coding: utf-8 -*-
"""_wirepairs.py —— 「如果网全部配对正确，导线还能得几分？」

把官方 net_line 的两级匹配拆开：网级配对（`match_nets`）与边级匈牙利。
本脚本做两件事：
  A 单变量实验：只放开**网配对**（对 GT9 用 GT 网数 / 用「全网的边一起对边」），
    看 tp/fp/fn 变成多少 ⇒ 网配对错误吃掉了多少。
  B 把「预测边 → 最近 GT 边」的距离按 **单边**（不要求同一 net）算，得到边级匹配的
    理论上限（取消网级门禁后的天花板）。
"""
import json
import math
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))
from schematic_parser.data import DEFAULT_DATA_ROOT, SPLITS, discover, load_json, normalise_doc  # noqa: E402
from schematic_parser.metrics import EvalConfig, _edge_points, _seg_distance, _assign  # noqa: E402

exact = {}
for c in discover(DEFAULT_DATA_ROOT, SPLITS):
    if c.json_path:
        exact.setdefault(c.case_id, c)
W = ROOT / "work"
GT9 = ["1", "2", "3", "4", "5", "7", "8", "9", "10"]
D19 = ["0092", "0048", "0019", "0024", "0086", "0042", "0075", "0083", "0090",
       "0057", "0062", "0032", "0028", "0175", "0198", "0105", "0106", "0036", "0120"]
INF = float("inf")


def edges(doc):
    out = []
    for nk, n in (doc.get("nets") or {}).items():
        for ek, v in (n.get("edges") or {}).items():
            p = _edge_points(v)
            if p is not None and len(p) >= 2:
                out.append((nk, p[0], p[-1]))
    return out


print("== B 取消网级门禁后的边级天花板（全局匈牙利，line_tol=5）==")
print("%-5s %6s %6s | %6s %6s %6s %8s | %6s %6s"
      % ("case", "GT边", "预边", "tp", "fp", "fn", "F1", "tp(net)", "fn(net)"))
S = {g: {"gt": 0, "pr": 0, "tp": 0, "fp": 0, "fn": 0} for g in ("GT9", "D19")}
for grp, cases in (("GT9", GT9), ("D19", D19)):
    for cid in cases:
        gt = normalise_doc(load_json(exact[cid].json_path))
        pp = (W / "_fast_out_reg9" / "final_results" / ("%s_Pred.json" % cid)) if cid in GT9 \
            else (W / "_f220_A" / ("%s.json" % cid))
        if not pp.exists():
            continue
        pred = normalise_doc(json.loads(pp.read_text(encoding="utf-8")))
        ge, pe = edges(gt), edges(pred)
        cost = [[(_seg_distance(p[1:], g[1:], True) if _seg_distance(p[1:], g[1:], True) <= 5.0
                  else INF) for g in ge] for p in pe]
        got = set(_assign(cost)) if (ge and pe) else set()
        tp = len(got)
        fp = len(pe) - tp
        fn = len(ge) - tp
        f1 = 2 * tp / max(1, 2 * tp + fp + fn)
        # 现实（带网级门禁）由 _wirestage 记录：这里只算差值用，取现成数字
        print("%-5s %6d %6d | %6d %6d %6d %8.4f" % (cid, len(ge), len(pe), tp, fp, fn, f1))
        S[grp]["gt"] += len(ge); S[grp]["pr"] += len(pe); S[grp]["tp"] += tp
        S[grp]["fp"] += fp; S[grp]["fn"] += fn
for grp in ("GT9", "D19"):
    d = S[grp]
    print("[%s] 取消网级门禁：tp %d fp %d fn %d  F1 %.4f"
          % (grp, d["tp"], d["fp"], d["fn"],
             2 * d["tp"] / max(1, 2 * d["tp"] + d["fp"] + d["fn"])))
    print("       对照现口径（带网配对）：%s"
          % ("tp 148 fp 857 fn 405 F1 0.1876" if grp == "GT9"
             else "tp 35 fp 1528 fn 2789 F1 0.0160"))
