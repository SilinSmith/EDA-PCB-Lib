# -*- coding: utf-8 -*-
"""Part 1 probe: the C151..C156 row in 10GT case 1, end to end."""
import json
import re
import sys
from pathlib import Path

ROOT = Path(r"Y:\workspace\EDA")
sys.path.insert(0, str(ROOT / "eda-pcb"))
from schematic_parser import data as D  # noqa: E402

RUN = ROOT / "work" / "_submit_out220r36" / "runs"
FUS = ROOT / "work" / "_fusion_out220no4"
VLM = ROOT / "work" / "_fusion_vlm"


def L(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


cases = [c for c in D.discover() if c.split == "10GT"]
case = [c for c in cases if c.case_id == "1"][0]
gt = L(case.json_path)
s1 = L(RUN / "stage1" / "1.stage1.json")
assigned = L(RUN / "stage2" / "_assigned" / "1.json")
st2 = L(RUN / "stage2" / "1.json")
fs = L(FUS / "1.json")
vlm = L(VLM / "1.json")

DPAT = re.compile(r"^[A-Za-z]{1,3}\d{3,4}$")

print("=" * 78)
print("A. GT components, case 1 (10GT)")
print("=" * 78)
pat = re.compile(r"^C1[0-9]?[0-9]?$")
print("-- keys matching ^[A-Za-z]{1,3}\\d{3,4}$ --")
for k in sorted(gt["components"]):
    if DPAT.match(k):
        c = gt["components"][k]
        print("  %-8s type=%-4s Name=%-8s value=%-8s bbox=%s" %
              (k, c.get("type"), c.get("Name"), c.get("value"), c.get("bbox")))
print("-- keys matching ^C1[0-9][0-9]$ --")
for k in sorted(gt["components"]):
    if re.match(r"^C1\d\d$", k):
        c = gt["components"][k]
        print("  %-8s type=%-4s Name=%-8s value=%-8s bbox=%s" %
              (k, c.get("type"), c.get("Name"), c.get("value"), c.get("bbox")))
print("-- all C-prefixed GT keys --")
cs = sorted(k for k in gt["components"] if k.startswith("C"))
print("  ", cs)

print()
print("=" * 78)
print("B. stage1 components near the row (y in [1000,1100])")
print("=" * 78)
comps = s1["components"]
row = []
for c in comps:
    b = c.get("bbox_json") or c.get("bbox_img")
    cy = (b[1] + b[3]) / 2.0
    if 1000 <= cy <= 1120:
        row.append(c)
row.sort(key=lambda c: c["bbox_json"][0])
for c in row:
    b = c["bbox_json"]
    print("  %-6s type=%-3s score=%.4f bbox_json=%s ctr=(%.1f,%.1f)" %
          (c["id"], c.get("type"), c.get("score") or 0, [round(v, 1) for v in b],
           (b[0] + b[2]) / 2, (b[1] + b[3]) / 2))


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    i = (x1 - x0) * (y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / ua


print()
print("-- GT row key -> nearest stage1 comp (by bbox_json IoU, else centre dist) --")
gtrow = ["C151", "C152", "C153", "C154", "C155", "C156"]
mapping = {}
for k in gtrow:
    gb = gt["components"][k]["bbox"]
    best = None
    for c in comps:
        b = c.get("bbox_json") or c.get("bbox_img")
        i = iou(gb, b)
        cd = ((gb[0] + gb[2]) / 2 - (b[0] + b[2]) / 2) ** 2 + \
             ((gb[1] + gb[3]) / 2 - (b[1] + b[3]) / 2) ** 2
        sc = (i, -cd)
        if best is None or sc > best[0]:
            best = (sc, c, i, cd ** 0.5)
    mapping[k] = best[1]["id"]
    print("  GT %-5s bbox=%-26s -> %s  IoU=%.3f ctrdist=%.2f  stage1 bbox_json=%s" %
          (k, gb, best[1]["id"], best[2], best[3],
           [round(v, 1) for v in (best[1].get("bbox_json") or [])]))

print()
print("=" * 78)
print("C. stage1 text boxes near the row (y in [130,220] img coords)")
print("=" * 78)
for t in s1["texts"]:
    b = t.get("bbox_img")
    if b and 130 <= (b[1] + b[3]) / 2 <= 220:
        print("  %-6s bbox_img=%s  w=%d h=%d  w/h=%.2f" %
              (t.get("id"), [round(v, 1) for v in b], b[2] - b[0], b[3] - b[1],
               (b[2] - b[0]) / max(1e-6, b[3] - b[1])))

print()
print("=" * 78)
print("D. _assigned: comps + desig_slots for the row")
print("=" * 78)
for k in gtrow:
    cid = mapping[k]
    rec = (assigned["comps"] or {}).get(cid) or {}
    print("GT %-5s cid=%s" % (k, cid))
    print("   comps[cid]: designator=%s value=%s partname=%s type=%s" %
          (json.dumps(rec.get("designator"), ensure_ascii=False),
           json.dumps(rec.get("value"), ensure_ascii=False),
           json.dumps(rec.get("partname"), ensure_ascii=False), rec.get("type")))
    for i, cd in enumerate((assigned.get("desig_slots") or {}).get(cid) or []):
        print("   slot[%d] text=%-8r canon=%-8r conf=%.4f score=%s box=%s d=%s box_i=%s" %
              (i, cd.get("text"), cd.get("canon"), cd.get("conf") or 0.0,
               cd.get("score"), [int(v) for v in (cd.get("box") or [])],
               round(cd.get("d"), 1) if cd.get("d") is not None else None,
               cd.get("box_i")))

print()
print("=" * 78)
print("E. fusion pool: comp.<cid>.key candidates (from_stage2 + priors + solve)")
print("=" * 78)
import os  # noqa: E402
os.environ["FUSION_LLM"] = "0"
os.environ["FUSION_CROPFIX"] = "0"
from fusion import claims as CL  # noqa: E402
from fusion import priors as PR  # noqa: E402
from fusion import solve as SV  # noqa: E402

vlm_arg = vlm if vlm else None
pool = CL.from_stage2("1", s1, assigned, bgr=None, vlm=vlm_arg)
pri = PR.Priors()
pst = pri.annotate(pool)
sol = SV.solve(pool)
for k in gtrow:
    cid = mapping[k]
    sid = "comp.%s.key" % cid
    print("GT %-5s %s" % (k, sid))
    for a in pool.alts(sid):
        print("   alt val=%-10r score=%8.4f src=%s ev=%s" %
              (a.value, a.score, a.source,
               json.dumps(a.evidence, ensure_ascii=False)[:160]))
    print("   -> solve chose: %s   (assign=%s)" %
          (pool.chosen(sid), sol["assign"].get(sid)))
print()
print("-- resolved-keys map from solve (all comp.*.key) --")
rk = {s: v for s, v in sol["assign"].items() if s.endswith(".key") and s.startswith("comp.")}
for cid in sorted(rk):
    print("   %-12s -> %s" % (cid, rk[cid]))

print()
print("=" * 78)
print("F. final designators")
print("=" * 78)
print("stage2/1.json components keys:", sorted(k for k in st2["components"] if k.startswith("C1")))
print("fusion_out220no4/1.json keys :", sorted(k for k in fs["components"] if k.startswith("C1")))
for doc, nm in ((st2, "stage2"), (fs, "fusion_out220no4")):
    for k in sorted(doc["components"]):
        if re.match(r"^C1[0-9]?[0-9]?$", k):
            c = doc["components"][k]
            print("  %-16s %-8s type=%-3s value=%-8s bbox=%s" %
                  (nm, k, c.get("type"), c.get("value"),
                   [round(v, 1) for v in (c.get("bbox") or [])]))
print()
print("-- what fusion doc has for the row bboxes --")
for k in gtrow:
    gb = gt["components"][k]["bbox"]
    best = None
    for dk, dc in fs["components"].items():
        b = dc.get("bbox")
        if not b:
            continue
        i = iou(gb, b)
        if best is None or i > best[0]:
            best = (i, dk, dc)
    print("  GT %-5s -> fusion key %-8s IoU=%.3f value=%s" %
          (k, best[1], best[0], best[2].get("value")))
