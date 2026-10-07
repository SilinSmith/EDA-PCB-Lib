import json, pathlib
from PIL import Image
bad=0
for f in sorted(pathlib.Path("work/viz/cases").glob("*.json")):
    d=json.loads(f.read_text(encoding="utf-8"))
    d0=pathlib.Path("work/viz")/d["assets"]
    want=(d["png_w"], d["png_h"]); sizes={}
    for ly in ("base","gt","ours","texts","pins","net"):
        p=d0/(ly+".png")
        if p.exists(): sizes[ly]=Image.open(p).size
    ok=all(s==want for s in sizes.values())
    if not ok: bad+=1; print("MISMATCH", d["id"], want, sizes)
print("图层尺寸与 png_w/png_h 不一致:", bad)
