"""离线改写预测网络的 edges，按官方口径重打分（10GT）。"""
import copy, json, sys
from pathlib import Path
ROOT = Path(r"Y:\workspace\EDA")
sys.path.insert(0, str(ROOT / "eda-pcb")); sys.path.insert(0, str(ROOT / "work" / "tools" / "eval"))
from schematic_parser.metrics import EvalConfig, evaluate_case
from schematic_parser.data import DEFAULT_DATA_ROOT
from stage2_eval import build_index, resolve, load_json

PRED = ROOT / "work" / "_gt10_vlmrun2" / "final_results"
ex, di = build_index(DEFAULT_DATA_ROOT)
CASES = ["1","2","3","4","5","7","8","9","10"]

def members(net):
    hg = (net.get("hyperGraph") or "").strip("()")
    return [p for p in hg.split(",") if p.strip()]

def run(tag, fn):
    tot = 0.0; agg = {}
    for cid in CASES:
        c, _ = resolve(cid, ex, di)
        gt = load_json(Path(c.json_path))
        pr = copy.deepcopy(load_json(PRED / (cid + "_Pred.json")))
        fn(pr)
        sc = evaluate_case(gt, pr, case_id=cid, cfg=EvalConfig())
        tot += sc.score
        for m in ("component","pin","net_hypergraph","net_line"):
            a = agg.setdefault(m, [0,0.0,0,0,0])
            r = getattr(sc, m); a[0]+=1; a[1]+=r.f1; a[2]+=r.tp; a[3]+=r.fp; a[4]+=r.fn
    n = len(CASES)
    print("%-26s 总分 %7.4f | netH F1 %.4f tp%-5d fp%-5d fn%-5d | line F1 %.4f tp%-5d fp%-5d fn%-5d"
          % (tag, tot/n, agg["net_hypergraph"][1]/n, agg["net_hypergraph"][2], agg["net_hypergraph"][3], agg["net_hypergraph"][4],
             agg["net_line"][1]/n, agg["net_line"][2], agg["net_line"][3], agg["net_line"][4]))
    return tot/n

def clear_edges_by_members(doc, lo, hi):
    for k, v in (doc.get("nets") or {}).items():
        n = len(members(v))
        if lo <= n <= hi:
            v["edges"] = {}

def min_edge(doc, L):
    for k, v in (doc.get("nets") or {}).items():
        ed = v.get("edges") or {}
        keep = {}
        for ek, e in ed.items():
            if len(e) >= 2:
                d = ((e[1]["x"]-e[0]["x"])**2 + (e[1]["y"]-e[0]["y"])**2) ** .5
                if d >= L: keep[ek] = e
        v["edges"] = keep

run("A 基线", lambda d: None)
for lo, hi in ((1,1),(1,2),(1,3)):
    run("B 单成员网去边 %d-%d" % (lo,hi), lambda d, lo=lo, hi=hi: clear_edges_by_members(d, lo, hi))
for L in (20, 40, 60, 90):
    run("C MIN_EDGE_LEN=%d" % L, lambda d, L=L: min_edge(d, L))
for L in (40, 60):
    run("D 单成员去边 + 边长>=%d" % L, lambda d, L=L: (clear_edges_by_members(d,1,1), min_edge(d, L)))
