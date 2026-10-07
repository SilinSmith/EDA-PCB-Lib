import json, sys
from pathlib import Path
sys.path.insert(0, "eda-pcb")
from schematic_parser.data import discover
for c in discover():
    if c.case_id != "1": continue
    gt = json.loads(Path(c.json_path).read_text(encoding="utf-8"))
    print("top keys:", list(gt.keys()))
    nets = gt.get("nets") or {}
    print("n nets:", len(nets))
    for k, v in list(nets.items())[:4]:
        print(" ", k, json.dumps(v, ensure_ascii=False)[:300])
    comps = gt.get("components") or {}
    om = [k for k in comps if str(k).startswith("\u2205")]
    print("phi comps:", om[:6], "n=", len(om))
    if om:
        print("  sample:", json.dumps(comps[om[0]], ensure_ascii=False)[:300])
