"""临时探针：看 stage1 doc 的 image.path 能不能读出来。"""
import json
import sys
from pathlib import Path

RUN = Path(r"Y:\workspace\EDA\work\_submit_out220r36\runs")
case = sys.argv[1] if len(sys.argv) > 1 else "0001"
d = json.loads((RUN / "stage1" / ("%s.stage1.json" % case)).read_text("utf-8"))
img = d.get("image") or {}
print("image =", {k: img.get(k) for k in ("path", "w", "h")})
p = img.get("path")
print("exists =", Path(p).exists() if p else None)
sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")
try:
    from stage2.text import imread_unicode
    b = imread_unicode(p)
    print("shape =", None if b is None else b.shape)
except Exception as exc:
    print("ERR:", type(exc).__name__, exc)

v = Path(r"Y:\workspace\EDA\work\_fusion_vlm") / ("%s.json" % case)
print("vlm json exists =", v.exists())
