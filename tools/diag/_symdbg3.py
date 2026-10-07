import json, sys, collections
from pathlib import Path
sys.path.insert(0, "eda-pcb")
from stage2 import assemble as A
s1 = json.loads(Path("work/_gt10_pb10/runs/stage1/1.stage1.json").read_text(encoding="utf-8"))
asg = json.loads(Path("work/_gt10_pb10/runs/stage2/_assigned/1.json").read_text(encoding="utf-8"))
orig = A.merge_by_symbol
def spy(nets, syms):
    r = orig(nets, syms)
    ns = collections.Counter()
    for s in syms.values():
        for x in s: ns[x] += 1
    print("  syms entries=%d nets=%d -> after_symbol=%d" % (len(syms), len(nets), len(r)))
    print("  每个符号被多少条 net 引用:", dict(ns))
    return r
A.merge_by_symbol = spy
ol = A.merge_by_label
def spy2(nets):
    r = ol(nets)
    print("  after_label=%d" % len(r))
    return r
A.merge_by_label = spy2
doc = A.build_doc(s1, asg, "1")
print("final nets:", len(doc["nets"]))
