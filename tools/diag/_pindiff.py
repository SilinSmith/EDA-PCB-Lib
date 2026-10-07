"""鐪?VLM 閭ｄ竴瓒熷埌搴曟妸 IC 鐨勫紩鑴氭敼鎴愪簡浠€涔堬細`runs/stage2/<cid>.json` vs `runs/stage3_vlm/<cid>.json`銆?
鐢ㄦ硶锛歚python work\\tools\\diag\\_pindiff.py 9`
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))

from schematic_parser.data import DEFAULT_DATA_ROOT, SPLITS, discover  # noqa: E402

CID = sys.argv[1] if len(sys.argv) > 1 else "9"
RUN = ROOT / "work" / "_submit_out220w" / "runs"

idx = {}
for sp in SPLITS:
    for c in discover(DEFAULT_DATA_ROOT, splits=(sp,)):
        idx.setdefault(c.case_id, c)
gt = json.loads(Path(idx[CID].json_path).read_text(encoding="utf-8"))
a = json.loads((RUN / "stage2" / ("%s.json" % CID)).read_text(encoding="utf-8-sig"))
b = json.loads((RUN / "stage3_vlm" / ("%s.json" % CID)).read_text(encoding="utf-8-sig"))

gt_ic = [k for k, v in gt["components"].items()
         if str(v.get("type", "")).lower() in ("box", "boxic", "ic")]
print("GT IC keys:", gt_ic)
for k in gt_ic:
    gp = (gt.get("pins") or {}).get(k) or {}
    print("== GT %s pins=%d" % (k, len(gp)))
    for pk in sorted(gp)[:40]:
        pv = gp[pk]
        print("     %-10s name=%-14s pt=%s" % (pk, pv.get("pinname"),
                                               pv.get("point")))
for tag, doc in (("stage2", a), ("vlm", b)):
    print("== %s  comps=%d" % (tag, len(doc.get("components", {}))))
    for k in gt_ic:
        pp = (doc.get("pins") or {}).get(k) or {}
        print("   %s pins=%d :" % (k, len(pp)))
        for pk in list(sorted(pp))[:40]:
            pv = pp[pk]
            print("     %-12s name=%-14s pt=%s" % (pk, pv.get("pinname"),
                                                   pv.get("point")))

