"""把某个 case 的「GT 网 <-> 我们网」的碎裂结构打印出来。
用法: python work\\tools\\diag\\_netfrag.py --pred work\\_gt10_ns0 --case 1
"""
import argparse, json, sys, collections
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))
sys.path.insert(0, str(ROOT / "work" / "tools" / "eval"))
from schematic_parser import metrics as M
from stage2_eval import build_index, resolve, load_json, DEFAULT_DATA_ROOT


def refs(net):
    hg = net.get("hyperGraph") or ""
    hg = hg.strip()
    if hg.startswith("(") and hg.endswith(")"):
        hg = hg[1:-1]
    out = set()
    for part in hg.split(","):
        part = part.strip()
        if not part or "." not in part:
            continue
        k, _, n = part.rpartition(".")
        out.add((k, n))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--case", default="1")
    a = ap.parse_args()
    exact, digits = build_index(DEFAULT_DATA_ROOT)
    case, how = resolve(a.case, exact, digits)
    if case is None:
        raise SystemExit("找不到 case %s (%s)" % (a.case, how))
    gt = load_json(case.json_path)
    pr = load_json(Path(a.pred) / (case.case_id + ".json"))
    cfg = M.EvalConfig()
    F1, keymap = M.assign_components(gt, pr, cfg)
    pr2 = M.rewrite_keys(pr, keymap)
    g_nets = [(k, refs(v)) for k, v in (gt.get("nets") or {}).items()]
    p_nets = [(k, refs(v)) for k, v in (pr2.get("nets") or {}).items()]
    print("GT 网 %d / 我们网 %d；GT 引用 %d / 我们引用 %d"
          % (len(g_nets), len(p_nets), sum(len(s) for _, s in g_nets),
             sum(len(s) for _, s in p_nets)))
    print("\n== 每条 GT 网被我们拆成几条 ==")
    hit = 0
    for gk, gs in sorted(g_nets, key=lambda kv: -len(kv[1])):
        owner = []
        for pk, ps in p_nets:
            inter = ps & gs
            if inter:
                owner.append((pk, len(inter), len(ps)))
        hit += len(gs)
        print("  %-9s n=%-3d 被拆成 %d 条: %s" %
              (gk, len(gs), len(owner),
               ", ".join("%s(+%d/%d)" % (p, m, t) for p, m, t in owner)))
    print("\n== 我们最大的十条网，成员散落在哪些 GT 网 ==")
    for pk, ps in sorted(p_nets, key=lambda kv: -len(kv[1]))[:10]:
        spread = collections.Counter()
        for gk, gs in g_nets:
            if ps & gs:
                spread[gk] = len(ps & gs)
        print("  %-9s n=%-3d -> %s" % (pk, len(ps), dict(spread)))


main()
