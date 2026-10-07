"""第二批参数核对：detect.py 的 conf/imgsz、metrics.py 的口径、enrich.py 的引脚名开关。"""
import re
from pathlib import Path

R = Path(r"Y:\workspace\EDA\eda-pcb")


def show(title, path, pat, n=10):
    print("=== " + title + "  (" + path.name + ")")
    if not path.is_file():
        print("   (不存在)", path)
        return
    rx = re.compile(pat, re.M)
    hits = [m.group(0).strip() for m in rx.finditer(path.read_text(encoding="utf-8"))]
    for h in hits[:n]:
        print("   ", h[:200])
    if not hits:
        print("    (无匹配)")


show("detect 初始化签名", R / "stage1/detect.py", r"^def __init__\(self.*$|^\s+def __init__\(self.*$", 3)
show("detect conf", R / "stage1/detect.py", r"^.*\bconf\b.*$", 10)
show("detect imgsz", R / "stage1/detect.py", r"^.*imgsz.*$", 6)
show("pipeline 里的 detect 调用", R / "stage1/pipeline.py", r"^.*(imgsz|conf)\s*=.*$", 8)
show("metrics 口径", R / "schematic_parser/metrics.py",
     r"^.*(bbox_tol|pin_tol|line_tol|unkeyed_match|net_mode|TOL)\s*[:=].*$", 20)
show("enrich 引脚名相关默认", R / "vlm/enrich.py", r"^.*(pin|inner)\b.*=.*$", 24)
