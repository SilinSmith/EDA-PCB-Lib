import json, sys, collections
from pathlib import Path
sys.path.insert(0, "eda-pcb")
from schematic_parser.data import discover
nums = collections.Counter()
for c in discover():
    if c.case_id not in [str(i) for i in range(1, 11)]: continue
    gt = json.loads(Path(c.json_path).read_text(encoding="utf-8"))
    comps = gt.get("components") or {}
    for k, grp in (gt.get("pins") or {}).items():
        cv = comps.get(k) or {}
        if str(k).startswith("\u2205"):
            for pk in grp:
                nums[pk] += 1
        else:
            nums["_real_%s" % str(cv.get("type"))] += 1
print("phi pin keys:", dict(nums.most_common(20)))
