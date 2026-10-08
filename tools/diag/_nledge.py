"""_nledge.py —— 把某一对配对 net 的 GT 边/我方边**逐条并排打出来**，肉眼看差在哪。

`_nlpair.py` 已经证明：GT 边 95.3% 都落在配对成功的 net 里 ⇒ `net_line` 不是被
「网络配对」卡住的，而是被**边的两个端点对不上**卡住的。那就要看端点差多少。

  python work\\tools\\diag\\_nledge.py --case 1 --top 3
  python work\\tools\\diag\\_nledge.py --case 2 --net 3
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
                                      assign_components, _edge_points,
                                      _seg_distance)
from stage2_eval import build_index, resolve                            # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")


def _pt(edge):
    pts = _edge_points(edge) or []
    out = []
    for p in pts:
        if isinstance(p, dict):
            out.append((round(float(p["x"]), 1), round(float(p["y"]), 1)))
        else:
            out.append((round(float(p[0]), 1), round(float(p[1]), 1)))
    return out


def _fmt(pts):
    if not pts:
        return "∅"
    a = pts[0]
    b = pts[-1] if len(pts) > 1 else pts[0]
    return "(%g,%g)-(%g,%g)" % (a[0], a[1], b[0], b[1])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="1")
    ap.add_argument("--out", default="work\\_fast_out_novlm9")
    ap.add_argument("--top", type=int, default=3, help="只打边数最多的前 N 对")
    ap.add_argument("--net", type=int, default=0, help="只打第 N 对（1 起）")
    a = ap.parse_args(argv)
    outdir = Path(a.out)
    if not outdir.is_absolute():
        outdir = ROOT / outdir
    cfg = EvalConfig()
    ex, di = build_index(DEFAULT_DATA_ROOT)
    c, _ = resolve(a.case, ex, di)
    if c is None:
        c, _ = resolve(a.case.lstrip("0") or "0", ex, di)
    gt = normalise_doc(load_json(Path(c.json_path)))
    pr = normalise_doc(load_json(outdir / "final_results" / (c.case_id + "_Pred.json")))
    _comp, km = assign_components(gt, pr, cfg, {})
    p2 = rewrite_keys(pr, km)
    _net, pairs = net_hypergraph_f1(gt, p2, cfg, {})
    g = list(gt["nets"].items())
    p = list(p2["nets"].items())
    info = []
    for i, j in pairs:
        ge = [e for e in (g[j][1].get("edges") or {}).values() if _edge_points(e)]
        pe = [e for e in (p[i][1].get("edges") or {}).values() if _edge_points(e)]
        info.append((len(ge) + len(pe), i, j, ge, pe))
    info.sort(key=lambda t: -t[0])
    pick = [info[a.net - 1]] if a.net else info[:a.top]
    print("case %s  共 %d 对配对 net" % (c.case_id, len(pairs)))
    for _n, i, j, ge, pe in pick:
        print("\n=== 配对#%d  GT %s(%d 边) ←→ 我方 %s(%d 边)"
              % (i + 1, g[j][0], len(ge), p[i][0], len(pe)))
        print("  --- GT ---")
        for e in ge:
            print("    %-46s %s" % (_fmt(_pt(e)), _pt(e)))
        print("  --- 我方（括号=到最近 GT 边的距离，>5 就配不上）---")
        for e in pe:
            best = min((_seg_distance(_pt(e), _pt(ge_k), cfg.line_unordered)
                        for ge_k in ge), default=None)
            print("    %-46s d=%-8s %s"
                  % (_fmt(_pt(e)), "None" if best is None else round(best, 1), _pt(e)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
