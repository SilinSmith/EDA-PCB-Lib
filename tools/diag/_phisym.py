import json, sys, collections
from pathlib import Path
sys.path.insert(0, "eda-pcb")
from schematic_parser.data import discover
CASES = [str(i) for i in range(1, 11)]
pred = Path("work/_gt10_pb10/runs/stage2")
tot = collections.Counter(); gtypes = collections.Counter(); ptypes = collections.Counter()
for c in discover():
    if c.case_id not in CASES: continue
    gt = json.loads(Path(c.json_path).read_text(encoding="utf-8"))
    pr = json.loads((pred / ("%s.json" % c.case_id)).read_text(encoding="utf-8"))
    for k, v in (gt.get("components") or {}).items():
        gtypes[str(v.get("type"))] += 1
        if str(k).startswith("\u2205"): tot["gt_phi"] += 1
    for k, v in (pr.get("components") or {}).items():
        ptypes[str(v.get("type"))] += 1
        if str(k).startswith("\u2205"): tot["pr_phi"] += 1
    for nk, n in (gt.get("nets") or {}).items():
        tot["gt_nets"] += 1
        hg = str(n.get("hyperGraph") or "")
        inner = hg.strip("()")
        mem = [x for x in inner.split(",") if x]
        tot["gt_mem"] += len(mem)
        nsym = sum(1 for x in mem if x.split(".")[0].startswith("\u2205"))
        if nsym: tot["gt_nets_with_phi"] += 1
        tot["gt_mem_phi"] += nsym
print(dict(tot))
print()
print("GT comp types:", dict(gtypes.most_common(20)))
print()
print("PR comp types:", dict(ptypes.most_common(20)))
