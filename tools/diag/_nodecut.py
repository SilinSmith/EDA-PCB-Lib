# -*- coding: utf-8 -*-
"""Node-based edge endpoint post-processing (offline trial; ASCII-only on purpose).

Idea from work/_pinorac.py: among the 211 endpoint targets that the GT oracle
(work/_endoracle.py --mode end, +1.28 total) wants to move, a large share sit on a
node that already exists in OUR edge graph, i.e. some *other* edge ends within a
couple of px of our edge's line.  Two ways to use that:
  node : snap our endpoint onto that node (inward trim, no new edge)
  cut  : split our edge at that node (T-junction knowledge)

Usage:
  python -X utf8 work\\_nodecut.py --run work\\_fast_out_finalrd8 --mode node --node-r 20 --out nd20
  python -X utf8 eda-pcb\\tools\\eval\\fast.py --score-only --cases 1,2,3,4,5,6,7,8,9,10 --out nd20
"""
import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]   # archived copy: work/ -> tools/diag/
RUNS = ROOT / "work"


def _pt(p):
    if isinstance(p, dict):
        return (float(p["x"]), float(p["y"]))
    return (float(p[0]), float(p[1]))


def _out(p, x, y):
    if isinstance(p, dict):
        return {"x": round(x, 2), "y": round(y, 2)}
    return [round(x, 2), round(y, 2)]


def _proj(p, a, b):
    """(projection point, perpendicular distance, t unclamped)."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    L2 = dx * dx + dy * dy
    if L2 <= 1e-9:
        return (a, ((p[0] - a[0]) ** 2 + (p[1] - a[1]) ** 2) ** 0.5, 0.0)
    t = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / L2
    tc = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    q = (a[0] + dx * tc, a[1] + dy * tc)
    return (q, ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) ** 0.5, t)


def _dist(p, q):
    return ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(RUNS / "_fast_out_finalrd8"))
    ap.add_argument("--mode", default="node", choices=["node", "cut"])
    ap.add_argument("--node-r", dest="node_r", type=float, default=20.0,
                    help="node mode: max along-axis distance of the node from the end")
    ap.add_argument("--perp", type=float, default=2.0, help="lateral tolerance")
    ap.add_argument("--margin", type=float, default=3.0,
                    help="node must project at least this far from both of our ends")
    ap.add_argument("--pin-guard", dest="pin_guard", type=float, default=6.0,
                    help="node mode: do not move an end that is this close to a pin")
    ap.add_argument("--trim", type=float, default=0.0,
                    help="node mode: residual inward trim for ends that did not snap")
    ap.add_argument("--out", default="nodecut")
    ap.add_argument("--cases", default="1,2,3,4,5,6,7,8,9,10")
    a = ap.parse_args()

    src = Path(a.run)
    dst = RUNS / ("_fast_out_" + a.out)
    if (dst / "final_results").exists():
        shutil.rmtree(dst / "final_results")
    (dst / "final_results").mkdir(parents=True, exist_ok=True)
    cases = [c.strip() for c in a.cases.split(",") if c.strip()]
    n_edges = n_end = 0
    for cid in cases:
        p = src / "final_results" / ("%s_Pred.json" % cid)
        if not p.exists():
            print("! missing %s" % p)
            continue
        doc = json.loads(p.read_text(encoding="utf-8"))
        pins = []
        for _ck, pp in (doc.get("pins") or {}).items():
            for _pk, rec in (pp or {}).items():
                q = (rec or {}).get("point") or {}
                if "x" in q and "y" in q:
                    pins.append((float(q["x"]), float(q["y"])))
        n = 0
        for net in (doc.get("nets") or {}).values():
            edges = net.get("edges") or {}
            segs = {}
            for k, seg in edges.items():
                if isinstance(seg, list) and len(seg) == 2:
                    segs[k] = (_pt(seg[0]), _pt(seg[1]))
            for k, seg in list(edges.items()):
                if k not in segs:
                    continue
                p0, p1 = segs[k]
                # nodes: endpoints of other edges that project onto our line
                nodes = []
                for k2, (q0, q1) in segs.items():
                    if k2 == k:
                        continue
                    for q in (q0, q1):
                        pr, dperp, t = _proj(q, p0, p1)
                        if dperp > a.perp or not (0.0 < t < 1.0):
                            continue
                        if _dist(pr, p0) < a.margin or _dist(pr, p1) < a.margin:
                            continue
                        if any(_dist(pr, w) <= 1.0 for w in nodes):
                            continue
                        nodes.append(pr)
                if a.mode == "cut":
                    if not nodes:
                        continue
                    nodes.sort(key=lambda w: _dist(w, p0))
                    chain = [p0] + nodes + [p1]
                    edges[k] = [_out(seg[0], chain[0][0], chain[0][1]),
                                _out(seg[1], chain[1][0], chain[1][1])]
                    for si in range(1, len(chain) - 1):
                        edges["%s~%d" % (k, si)] = [
                            _out(seg[0], chain[si][0], chain[si][1]),
                            _out(seg[1], chain[si + 1][0], chain[si + 1][1])]
                    n += 1
                    n_end += len(nodes)
                    continue
                # ---- node mode
                new = []
                for i, pt in enumerate((p0, p1)):
                    other = p1 if i == 0 else p0
                    if pins and min(_dist(pt, q) for q in pins) <= a.pin_guard:
                        new.append(pt)
                        continue
                    best = None
                    for w in nodes:
                        al = _dist(pt, w)
                        if al <= a.node_r and (best is None or al < best[0]):
                            best = (al, w)
                    if best is not None:
                        new.append(best[1])
                        n_end += 1
                    elif a.trim > 0:
                        dx, dy = other[0] - pt[0], other[1] - pt[1]
                        L = (dx * dx + dy * dy) ** 0.5
                        tt = min(a.trim, L * 0.4)
                        new.append((pt[0] + dx / L * tt, pt[1] + dy / L * tt))
                    else:
                        new.append(pt)
                if _dist(new[0], p0) > 1e-9 or _dist(new[1], p1) > 1e-9:
                    edges[k] = [_out(seg[0], new[0][0], new[0][1]),
                                _out(seg[1], new[1][0], new[1][1])]
                    n += 1
        n_edges += n
        (dst / "final_results" / ("%s_Pred.json" % cid)).write_text(
            json.dumps(doc, ensure_ascii=False), encoding="utf-8")
        print("case %-4s changed %d edges" % (cid, n))
    print("total changed edges %d, touched ends %d, mode=%s r=%.1f perp=%.1f margin=%.1f => %s"
          % (n_edges, n_end, a.mode, a.node_r, a.perp, a.margin, dst))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
