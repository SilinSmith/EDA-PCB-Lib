import sys, math, collections
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")
from schematic_parser.data import DEFAULT_DATA_ROOT, discover, load_json
from schematic_parser.metrics import IC_TYPES, canon_type, normalise_doc, _point
cs = [c for c in discover(DEFAULT_DATA_ROOT) if c.split == "200_train_cases"]
st = collections.Counter(); dists = []
def box(bb):
    if isinstance(bb, dict): v = [bb.get("x1"), bb.get("y1"), bb.get("x2"), bb.get("y2")]
    elif isinstance(bb, (list, tuple)) and len(bb) >= 4: v = list(bb[:4])
    else: return None
    if any(t is None for t in v): return None
    x1, y1, x2, y2 = [float(t) for t in v]
    return min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2)
for c in cs[:60]:
    gt = normalise_doc(load_json(c.json_path))
    for comp, v in (gt.get("components") or {}).items():
        if canon_type((v or {}).get("type")) not in IC_TYPES: continue
        b = box(v.get("bbox"))
        if not b: continue
        x1, y1, x2, y2 = b
        for pk, g in ((gt.get("pins") or {}).get(comp) or {}).items():
            if not isinstance(g, dict): continue
            p = _point(g.get("point"))
            if not p: continue
            x, y = p
            dx = max(x1 - x, x - x2, 0.0); dy = max(y1 - y, y - y2, 0.0)
            out = math.hypot(dx, dy)
            st["pins"] += 1; st["outside"] += (out > 1.0)
            dists.append(out)
dists.sort(); n = len(dists)
print("GT IC 引脚 %d 根；点落在元件 bbox 外（>1px）的 %d = %.1f%%" % (
    st["pins"], st["outside"], 100.0*st["outside"]/max(1,st["pins"])))
for q in (10, 25, 50, 75, 90, 99):
    print("  到 bbox 边界距离 p%-2d = %.1f px" % (q, dists[min(n-1,int(n*q/100))]))
