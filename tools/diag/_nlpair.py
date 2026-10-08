"""_nlpair.py —— `net_line` 是被**边**卡住，还是被**网络配对**卡住？

`net_line_f1` 只对 `net_hypergraph_f1` **配对成功**的那些 net 比边；没配对上的
net 的全部边直接记 `fn` / `fp`。所以如果 GT 的大部分边本来就在「没配对上的
net」里，那再怎么修边的形状（`_nldup.py` 已经把能试的都试了，只 +0.02）也没用，
真正的瓶颈是**网络本身的配对**（= `ev_bridge` 的并网判据）。

  python work\\tools\\diag\\_nlpair.py --out work\\_fast_out_novlm9
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
EDA = ROOT / "eda-pcb"
sys.path.insert(0, str(EDA))
sys.path.insert(0, str(ROOT / "work" / "tools" / "eval"))

from schematic_parser.data import DEFAULT_DATA_ROOT, load_json          # noqa: E402
from schematic_parser.metrics import (EvalConfig, normalise_doc,        # noqa: E402
                                      net_hypergraph_f1, rewrite_keys,
                                      assign_components, _edge_points)
from stage2_eval import build_index, resolve                            # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
OFFICIAL = ["1", "2", "3", "4", "5", "7", "8", "9", "10"]


def _edges(net):
    if not isinstance(net, dict):
        return []
    return [v for v in (net.get("edges") or {}).values()
            if _edge_points(v) is not None]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="work\\_fast_out_novlm9")
    a = ap.parse_args(argv)
    outdir = Path(a.out)
    if not outdir.is_absolute():
        outdir = ROOT / outdir
    cfg = EvalConfig()
    ex, di = build_index(DEFAULT_DATA_ROOT)
    print("%-6s %5s %5s %6s %6s %6s %6s %7s %8s"
          % ("case", "配对", "gN", "pN", "gE", "gE配对", "pE", "pE配对", "GT边覆盖"))
    print("-" * 76)
    TG = TGP = TE = TEP = 0
    for cid in OFFICIAL:
        c, _ = resolve(cid, ex, di)
        if c is None:
            c, _ = resolve(cid.lstrip("0") or "0", ex, di)
        if c is None:
            continue
        gt = normalise_doc(load_json(Path(c.json_path)))
        pr = normalise_doc(load_json(outdir / "final_results" / (c.case_id + "_Pred.json")))
        _comp, km = assign_components(gt, pr, cfg, {})
        p2 = rewrite_keys(pr, km)
        _net, pairs = net_hypergraph_f1(gt, p2, cfg, {})
        gi = {j for _i, j in pairs}
        pi = {i for i, _j in pairs}
        g = list(gt["nets"].items())
        p = list(p2["nets"].items())
        gE = sum(len(_edges(v)) for _k, v in g)
        gEp = sum(len(_edges(v)) for j, (_k, v) in enumerate(g) if j in gi)
        pE = sum(len(_edges(v)) for _k, v in p)
        pEp = sum(len(_edges(v)) for i, (_k, v) in enumerate(p) if i in pi)
        print("%-6s %5d %5d %6d %6d %6d %6d %7d %7.1f%%"
              % (c.case_id, len(pairs), len(g), len(p), gE, gEp, pE, pEp,
                 100.0 * gEp / max(1, gE)))
        TG += gE; TGP += gEp; TE += pE; TEP += pEp
    print("-" * 76)
    print("合计 GT 边 %d（在配对 net 里的 %d = %.1f%%）；我方边 %d（%d = %.1f%%）"
          % (TG, TGP, 100.0 * TGP / max(1, TG), TE, TEP, 100.0 * TEP / max(1, TE)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
