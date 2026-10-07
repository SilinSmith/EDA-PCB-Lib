"""量 netH（权重 0.35）的结构性杠杆：把预测的每张网**拆成单引脚网**会怎样。

背景（已实测，220 张 `work\\_submit_out220r27`）：
  GT  22691 个引脚，其中 **22199（97.8%）** 出现在 `nets[*].hyperGraph` 里；
  我们 20373 个引脚，只有 **8419（41.3%）** 出现在网里。
  但 `net.pred_pins 8419` 里真正能在 GT 网中找到的只有 **4818**，而匈牙利匹配给出的 `tp` 只有 **3236**。
  差额 1582 是「网的分组方式与 GT 不同」造成的 —— GT 的网很小（220 张 10745 条网 / 22225 条引用 ≈ 2.07 引用/网，
  case 0001 的大小分布是 `{1:58, 2:47, 3:16, 4:2, 5:1, 12:1}`），而 `net_hypergraph_f1` 的 tp 是
  `Σ_matched |p ∩ g|`：我们一张多引脚网只能认领**一个** GT 网，剩下的引脚全变 fp。

本脚本对比三种「网粒度」下的 netH，全部走官方 `evaluate_case` 的流程（先 `assign_components` 拿 keymap，
再 `rewrite_keys` 改写预测，然后才打分）：
  A `asis`      —— 原样
  B `pairs`     —— 每张网拆成若干 2 引脚小网（余 1 的单独成网）
  C `singleton` —— 每张网拆成单引脚网

用法：
  python work\\tools\\diag\\_netsplit.py --pred work\\_submit_out220r27\\runs\\stage2 [--limit N]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "eda-pcb"))

from schematic_parser import data as D                      # noqa: E402
from schematic_parser import metrics as M                   # noqa: E402
from schematic_parser.data import net_pin_set, build_hypergraph  # noqa: E402


def _rehash(nets):
    """给拆出来的网重新编号，保持 dict 形状（name -> net dict）。"""
    return {"net_%d" % (i + 1): v for i, (_, v) in enumerate(nets)}


def split_nets(nets, mode):
    out = []
    for name, net in nets.items():
        if not isinstance(net, dict):
            out.append((name, net))
            continue
        refs = list(net_pin_set(net))
        edges = net.get("edges")
        if len(refs) <= 1:
            out.append((name, net))
            continue
        if mode == "singleton":
            chunks = [[r] for r in refs]
        else:                                    # pairs
            chunks = [refs[i:i + 2] for i in range(0, len(refs), 2)]
        for ch in chunks:
            nn = dict(net)
            nn["hyperGraph"] = build_hypergraph(ch)
            if edges:
                nn["edges"] = dict(edges)
            out.append((name, nn))
    return _rehash(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="stage2 产物目录")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--split", default="", help="只跑某个 split（10GT / 200_train_cases / 10_GTcase）")
    args = ap.parse_args()

    cases = []
    for c in D.discover(D.DEFAULT_DATA_ROOT):
        if args.split and getattr(c, "split", "") != args.split:
            continue
        p = os.path.join(args.pred, "%s.json" % c.case_id)
        if os.path.exists(p):
            cases.append((c, p))
    cases.sort(key=lambda t: t[0].case_id)
    if args.limit:
        cases = cases[:args.limit]
    print("配上的 case 数：%d" % len(cases))

    cfg = M.EvalConfig()
    agg = {m: {"tp": 0, "fp": 0, "fn": 0, "nets": 0} for m in ("asis", "pairs", "singleton")}
    for c, p in cases:
        gt = D.load_json(c.json_path)
        pr = D.load_json(p)
        _cf1, km = M.assign_components(gt, pr, cfg)
        pred2 = M.rewrite_keys(pr, km)
        for mode in ("asis", "pairs", "singleton"):
            doc = dict(pred2)
            if mode != "asis":
                doc["nets"] = split_nets(pred2.get("nets") or {}, mode)
            det = {}
            res, _pairs = M.net_hypergraph_f1(gt, doc, cfg, det)
            agg[mode]["tp"] += res.tp
            agg[mode]["fp"] += res.fp
            agg[mode]["fn"] += res.fn
            agg[mode]["nets"] += det.get("net.pred_count", 0)

    base = None
    for mode in ("asis", "pairs", "singleton"):
        a = agg[mode]
        p = a["tp"] / max(1, a["tp"] + a["fp"])
        r = a["tp"] / max(1, a["tp"] + a["fn"])
        f1 = 2 * p * r / max(1e-9, p + r)
        if base is None:
            base = f1
        print("%-10s tp=%-6d fp=%-6d fn=%-6d nets=%-6d  P=%.4f R=%.4f netH=%.4f  (%+.4f)"
              % (mode, a["tp"], a["fp"], a["fn"], a["nets"], p, r, f1, f1 - base))


if __name__ == "__main__":
    main()
