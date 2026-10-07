import json, sys
from pathlib import Path
sys.path.insert(0, "eda-pcb")
from schematic_parser.data import discover
for c in discover():
    if c.case_id != "1": continue
    gt = json.loads(Path(c.json_path).read_text(encoding="utf-8"))
    comps = gt.get("components") or {}
    print("GT phi comps:")
    for k, v in comps.items():
        if str(k).startswith("\u2205"):
            print("   %-8s type=%-5s val=%-6s bbox=%s pins=%s"
                  % (k, v.get("type"), v.get("value"), v.get("bbox"),
                     json.dumps((gt.get("pins") or {}).get(k), ensure_ascii=False)))
    print()
    for nk, n in (gt.get("nets") or {}).items():
        hg = str(n.get("hyperGraph") or "")
        if "\u2205" in hg:
            print("   %-8s %s" % (nk, hg[:120]))
