"""看 case `1`（10GT）的 `U1` 到底为什么没配上键：GT 的 IC 框 vs 我们预测里的同一个框。

用法：`python work\\tools\\diag\\_c1.py [case_id]`
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))

from schematic_parser.data import DEFAULT_DATA_ROOT, SPLITS, discover  # noqa: E402

CID = sys.argv[1] if len(sys.argv) > 1 else "1"

idx = {}
for sp in SPLITS:
    for c in discover(DEFAULT_DATA_ROOT, splits=(sp,)):
        idx.setdefault(c.case_id, c)

c = idx[CID]
gt = json.loads(Path(c.json_path).read_text(encoding="utf-8"))
print("=== GT", CID, c.split, c.json_path)
for k, v in gt["components"].items():
    if str(v.get("type", "")).lower() in ("box", "boxic", "ic"):
        print("   GT %-8s %-6s %-22s %s pins=%s" % (
            k, v.get("type"), v.get("Name"), v.get("bbox"),
            len(v.get("pins") or [])))

for cand in ("work/_submit_out220w/runs/stage3_vlm/%s.json" % CID,
             "work/_submit_out220w/runs/stage2/%s.json" % CID,
             "work/_submit_out220z/runs/stage2/%s.json" % CID,
             "work/_submit_out220z/final_results/%s_Pred.json" % CID):
    p = ROOT / cand
    if not p.exists():
        continue
    pr = json.loads(p.read_text(encoding="utf-8"))
    print("=== PRED", cand, "n_comp", len(pr.get("components", {})),
          "n_pins", len(pr.get("pins", {})))
    pins = pr.get("pins", {})
    for k, v in pr["components"].items():
        bb = v.get("bbox") or []
        mine = pins.get(k) or {}
        big = bool(bb) and (bb[2] - bb[0]) > 250 and (bb[3] - bb[1]) > 250
        if str(v.get("type", "")).lower() in ("box", "boxic", "ic") or big:
            print("   PR %-8s %-6s %-22s %s npins=%d src=%s" % (
                k, v.get("type"), v.get("designator") or v.get("Name"),
                v.get("bbox"), len(mine), v.get("source")))
    break

# --- 逐引脚对比（U1 / 第一个 GT IC 框）---
gt_ic = [k for k, v in gt["components"].items()
         if str(v.get("type", "")).lower() in ("box", "boxic", "ic")]
if gt_ic:
    gk = gt_ic[0]
    gp = (gt.get("pins") or {}).get(gk) or {}
    pp = (pr.get("pins") or {}).get(gk) or {}
    print("=== PIN COMPARE", gk, "gt", len(gp), "pred", len(pp))
    print("  GT  nums:", sorted(gp.keys()))
    print("  PR  nums:", sorted(pp.keys()))
    # 按 pinname 配
    gmap = {}
    for pk, pv in gp.items():
        gmap.setdefault(str(pv.get("pinname")), []).append(pv.get("point"))
    pmap = {}
    for pk, pv in pp.items():
        pmap.setdefault(str(pv.get("pinname")), []).append(pv.get("point"))
    hit = miss = 0
    for nm, pts in gmap.items():
        if nm not in pmap:
            miss += 1
            continue
        d = min(((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2) ** 0.5
                for a in pts for b in pmap[nm])
        if d <= 5.0:
            hit += 1
        else:
            miss += 1
            if miss <= 6:
                print("   ~ %-8s gt%s nearest pred%s d=%.1f" % (nm, pts, pmap[nm], d))
    print("  name-matched %d / %d (missing %d)" % (hit, len(gmap), miss))
