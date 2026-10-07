"""按 GT 网里 φ 符号的 type 拆分「最好单网命中」，估常识约束的收益上限。
用法: python work\\tools\\diag\\_gndpot.py --pred work\\_gt10_pb10\\runs\\stage2
"""
import argparse, sys
from pathlib import Path
from collections import defaultdict
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb")); sys.path.insert(0, str(ROOT / "work" / "tools" / "eval"))
from stage2_eval import build_index, resolve, load_json, DEFAULT_DATA_ROOT
from schematic_parser import metrics as M

ap = argparse.ArgumentParser(); ap.add_argument("--pred", required=True)
a = ap.parse_args()
exact, digits = build_index(DEFAULT_DATA_ROOT)
cfg = M.EvalConfig()
T = defaultdict(lambda: defaultdict(int))
for case in exact.values():
    gt = load_json(case.json_path)
    prp = Path(a.pred) / (case.case_id + ".json")
    if not prp.exists():
        continue
    pr = load_json(prp)
    _, keymap = M.assign_components(gt, pr, cfg)
    M.rewrite_keys(pr, keymap)
    gty = {k: (v.get("type") or "") for k, v in (gt.get("components") or {}).items()}
    gmem = {}
    for k, v in (gt.get("nets") or {}).items():
        hg = (v.get("hyperGraph") or "").strip("()")
        gmem[k] = set(x.strip() for x in hg.split(",") if x.strip())
    pmem = []
    for k, v in (pr.get("nets") or {}).items():
        hg = (v.get("hyperGraph") or "").strip("()")
        pmem.append(set(x.strip() for x in hg.split(",") if x.strip()))
    sc = M.evaluate_case(gt, pr, cfg).net_hypergraph
    T["_"]["real_tp"] += sc.tp
    T["_"]["real_fp"] += sc.fp
    T["_"]["real_fn"] += sc.fn
    for nm, gm in gmem.items():
        kinds = sorted({gty.get(m.rpartition(".")[0], "?") for m in gm if m.rpartition(".")[0].startswith("\u2205")})
        key = "+".join(kinds) if kinds else "(no-phi)"
        best = max((len(s & gm) for s in pmem), default=0)
        T[key]["n"] += 1; T[key]["mem"] += len(gm); T[key]["best"] += best
print("%-12s %6s %8s %10s" % ("GT网里φ的type", "网数", "GT成员", "单网命中上限"))
for k in sorted(T):
    if k == "_": continue
    v = T[k]
    print("%-12s %6d %8d %10d" % (k, v["n"], v["mem"], v["best"]))
print("real_tp=%d real_fp=%d real_fn=%d" % (T["_"]["real_tp"], T["_"]["real_fp"], T["_"]["real_fn"]))
