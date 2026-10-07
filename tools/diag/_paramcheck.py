"""核对 workflow.html 各节点 params 栏里的常量是否与当前代码一致。"""
import re
from pathlib import Path

R = Path(r"Y:\workspace\EDA\eda-pcb")
CASES = [
    ("ev_sym.conf", R / "stage1/detect.py", r"^.*CONF.*$"),
    ("ev_sym.权重", R / "stage1/detect.py", r"^.*(?:WEIGHT|CKPT|weights).*$"),
    ("ev_vlm.inner", R / "vlm/enrich.py", r"^.*\binner\b.*=.*$"),
    ("ev_vlm.margin", R / "vlm/enrich.py", r"^.*margin.*=.*$"),
    ("assemble.VALUE_LIKE", R / "stage2/assemble.py", r"^VALUE_LIKE.*$"),
    ("stage1.imgsz", R / "stage1/detect.py", r"^.*IMGSZ.*$"),
]
for name, p, pat in CASES:
    print("=== " + name + "  (" + p.name + ")")
    if not p.is_file():
        print("   (文件不存在)", p)
        continue
    rx = re.compile(pat, re.M)
    hits = [m.group(0).strip() for m in rx.finditer(p.read_text(encoding="utf-8"))]
    for h in hits[:8]:
        print("   ", h[:190])
    if not hits:
        print("    (无匹配)")

print()
print("=== metrics 口径 (schematic_parser) ===")
sp = Path(r"Y:\workspace\EDA\schematic_parser")
if sp.is_dir():
    for f in sorted(sp.rglob("*.py")):
        if "__pycache__" in str(f):
            continue
        s = f.read_text(encoding="utf-8", errors="replace")
        for m in re.finditer(r"^.*(bbox_tol|pin_tol|line_tol|unkeyed_match|net_mode).*=.*$", s, re.M):
            print("   %-28s %s" % (f.name, m.group(0).strip()[:150]))
else:
    print("   ", sp, "不存在")
