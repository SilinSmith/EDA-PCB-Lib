import json, pathlib
from PIL import Image
bad = 0
for f in sorted(pathlib.Path("work/viz/cases").glob("*.json")):
    d = json.loads(f.read_text(encoding="utf-8"))
    d0 = pathlib.Path("work/viz") / d["assets"]
    sizes = {}
    for ly in ("base","gt","ours","texts","pins","net"):
        p = d0 / (ly + ".png")
        if p.exists(): sizes[ly] = Image.open(p).size
    want = (d["ow"], d["oh"])
    ok = all(s == want for s in sizes.values())
    # 越界的框（相对 ow/oh）
    oob = 0
    for c in (d["ours"]["comps"] + d["gt"]["comps"]):
        b = c["box"]
        if b[2] > d["ow"] + 2 or b[3] > d["oh"] + 2 or b[0] < -2 or b[1] < -2: oob += 1
    print("%-20s ow/oh=%s layers=%d all_same=%s 超界框=%d" % (d["id"], want, len(sizes), ok, oob))
    if not ok: bad += 1
print("尺寸不一致的图:", bad)
