import json, pathlib, collections, sys
sys.path.insert(0, "eda-pcb")
from schematic_parser import data as D
cs = list(D.discover())
g = collections.defaultdict(list)
for c in cs: g[c.split].append(c.case_id)
for k in sorted(g): print(k, len(g[k]), sorted(g[k])[:14])
