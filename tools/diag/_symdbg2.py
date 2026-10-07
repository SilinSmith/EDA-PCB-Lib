import json, sys
from pathlib import Path
sys.path.insert(0, "eda-pcb")
from stage2 import assemble as A
s1 = json.loads(Path("work/_gt10_pb10/runs/stage1/1.stage1.json").read_text(encoding="utf-8"))
asg = json.loads(Path("work/_gt10_pb10/runs/stage2/_assigned/1.json").read_text(encoding="utf-8"))
orig = A.merge_by_symbol
def spy(nets, syms):
    print("  syms entries:", len(syms), "| nets:", len(nets))
    print("  sample syms:", list(syms.items())[:4])
    return orig(nets, syms)
A.merge_by_symbol = spy
doc = A.build_doc(s1, asg, "1")
print("out nets:", len(doc["nets"]))
for k, v in list(doc["nets"].items()):
    hg = str(v.get("hyperGraph"))
    if len(hg) > 60:
        print("   ", k, hg[:180])
