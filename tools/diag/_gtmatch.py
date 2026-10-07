import sys, json
from pathlib import Path
sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")
sys.path.insert(0, r"Y:\workspace\EDA\work\tools\eval")
from schematic_parser.data import DEFAULT_DATA_ROOT, discover
from stage2_eval import build_index, resolve
cases = discover(DEFAULT_DATA_ROOT)
print("DEFAULT_DATA_ROOT:", DEFAULT_DATA_ROOT)
print("discover() ->", len(cases), "个 case")
from collections import Counter
print("split 分布:", Counter(c.split for c in cases))
ids = set(c.case_id for c in cases)
pred = Path(r"Y:\workspace\EDA\work\_submit_out220r32\runs\stage3_vlm")
got = [p.stem for p in pred.glob("*.json")]
hit = [g for g in got if g in ids]
print("产物 %d 个，其中能配上 GT 的 %d 个" % (len(got), len(hit)))
miss = sorted(set(got) - ids)[:12]
print("配不上的例子:", miss)
