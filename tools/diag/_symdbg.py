import json, sys
from pathlib import Path
sys.path.insert(0, "eda-pcb")
sys.argv = ["x"]
from stage2 import assemble as A
print("SYM_MERGE =", A.SYM_MERGE, "POWER_TYPES =", sorted(A.POWER_TYPES)[:6])
s1 = json.loads(Path("work/_gt10_pb10/runs/stage1/1.stage1.json").read_text(encoding="utf-8"))
asg = json.loads(Path("work/_gt10_pb10/runs/stage2/_assigned/1.json").read_text(encoding="utf-8"))
# 手工重现 build_doc 的前半段：看 keymap / num_of / syms
doc = A.build_doc(s1, asg, "1")
print("out nets:", len(doc["nets"]))
print("p keys:", [k for k in doc["components"] if k.startswith("p")][:8])
