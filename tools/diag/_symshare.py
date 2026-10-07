import json, sys, collections
from pathlib import Path
sys.path.insert(0, "eda-pcb")
from schematic_parser.data import discover
import importlib
M = importlib.import_module("schematic_parser.metrics")
CASES = [str(i) for i in range(1, 11)]
tot = collections.Counter()
types = collections.Counter()
members = collections.Counter()
for c in discover():
    if c.case_id not in CASES: continue
    gt = json.loads(Path(c.json_path).read_text(encoding="utf-8"))
    comps = gt.get("components") or {}
    for nk, n in (gt.get("nets") or {}).items():
        tot["nets"] += 1
        refs = n.get("net_pin_set") or n.get("pins") or []
        for r in refs:
            tot["members"] += 1
            comp = str(r).split(".")[0]
            members["has_ref"] += 1
            if comp.startswith("\u2205"):
                tot["sym_members"] += 1
                cv = comps.get(comp) or {}
                types[str(cv.get("type"))] += 1
                tot["nets_with_sym"] += 0
            else:
                tot["real_members"] += 1
        if any(str(r).split(".")[0].startswith("\u2205") for r in refs):
            tot["nets_with_sym"] += 1
print(dict(tot))
print("sym types:", dict(types.most_common(10)))
