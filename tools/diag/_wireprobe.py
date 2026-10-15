# -*- coding: utf-8 -*-
"""_wireprobe.py —— 抽查预测边与 GT 边的坐标系/端点对齐情况。"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))
from schematic_parser.data import discover, SPLITS, DEFAULT_DATA_ROOT, load_json  # noqa: E402

idx = {c.case_id: c for c in discover(DEFAULT_DATA_ROOT, SPLITS) if c.json_path}

PAIRS = [("0048", ROOT / "work/_f220_A/0048.json"),
         ("0036", ROOT / "work/_f220_A/0036.json"),
         ("0092", ROOT / "work/_f220_A/0092.json"),
         ("0019", ROOT / "work/_f220_A/0019.json"),
         ("1", ROOT / "work/_fast_out_reg9/final_results/1_Pred.json"),
         ("5", ROOT / "work/_fast_out_reg9/final_results/5_Pred.json")]


def pts_of(d):
    out = []
    for k, n in (d.get("nets") or {}).items():
        for ek, e in (n.get("edges") or {}).items():
            for q in e:
                if isinstance(q, dict) and "x" in q:
                    out.append((float(q["x"]), float(q["y"])))
                elif isinstance(q, (list, tuple)) and len(q) == 2:
                    out.append((float(q[0]), float(q[1])))
    return out


def rng(P, i):
    v = [q[i] for q in P]
    return (round(min(v), 1), round(max(v), 1)) if v else None


for cid, p in PAIRS:
    c = idx[cid]
    gt = load_json(c.json_path)
    pr = json.loads(Path(p).read_text(encoding="utf-8"))
    gp, pp = pts_of(gt), pts_of(pr)
    print("=== %s  gt=%s" % (cid, c.json_path))
    print("   gt image=%s  pred top keys=%s" % (gt.get("image"), list(pr.keys())[:16]))
    print("   gt pts %4d  x %s  y %s" % (len(gp), rng(gp, 0), rng(gp, 1)))
    print("   pr pts %4d  x %s  y %s" % (len(pp), rng(pp, 0), rng(pp, 1)))
    # 抽前 3 条 GT 边 + 预测边原样打出来
    gE = [(k, ek, e) for k, n in (gt.get("nets") or {}).items()
          for ek, e in (n.get("edges") or {}).items()]
    pE = [(k, ek, e) for k, n in (pr.get("nets") or {}).items()
          for ek, e in (n.get("edges") or {}).items()]
    print("   GT 边 %d 条，前 3：" % len(gE))
    for k, ek, e in gE[:3]:
        print("      %s/%s %s" % (k, ek, json.dumps(e, ensure_ascii=False)[:160]))
    print("   预测边 %d 条，前 3：" % len(pE))
    for k, ek, e in pE[:3]:
        print("      %s/%s %s" % (k, ek, json.dumps(e, ensure_ascii=False)[:160]))
    # 预测每个 net 的 hyperGraph 前 2 条
    print("   预测 net 样例: %s" % json.dumps(
        {k: v.get("hyperGraph") for k, v in list((pr.get("nets") or {}).items())[:4]},
        ensure_ascii=False)[:400])
    print("   GT   net 样例: %s" % json.dumps(
        {k: v.get("hyperGraph") for k, v in list((gt.get("nets") or {}).items())[:4]},
        ensure_ascii=False)[:400])
