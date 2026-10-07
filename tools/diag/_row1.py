# -*- coding: utf-8 -*-
"""把 case 1 电容排（GT `C155`..`C164`）整排打出来：
真值键 + GT 框、我方每个候选文字框（x0/w/h/w-h/读法）+ 该 cid 的胜出位号。
用来给新 `axis.py` 当验收样例。
"""
import json
import sys
from pathlib import Path

ROOT = Path("Y:/workspace/EDA")
sys.path.insert(0, str(ROOT / "eda-pcb"))
from schematic_parser import data as D  # noqa: E402
from PIL import Image  # noqa: E402

R = ROOT / "work/_submit_out220r36/runs"
CASES = {c.case_id: c for c in D.discover()}


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def main():
    cid = sys.argv[1] if len(sys.argv) > 1 else "1"
    c = CASES[cid]
    gt = load(c.json_path)
    asg = load(R / "stage2/_assigned" / (cid + ".json"))
    H = float(Image.open(c.png_path).size[1])
    lo = float(sys.argv[2]) if len(sys.argv) > 2 else 600.0
    hi = float(sys.argv[3]) if len(sys.argv) > 3 else 700.0

    def flip(b):
        return [b[0], H - b[3], b[2], H - b[1]]

    print("== GT 元件（图像空间 y 中点在 [%g,%g)）==" % (lo, hi))
    rows = []
    for k, v in (gt.get("components") or {}).items():
        bb = v.get("bbox")
        if not bb:
            continue
        g = flip([float(x) for x in bb])
        yc = (g[1] + g[3]) / 2
        if lo <= yc < hi:
            rows.append((g[0], k, g, v.get("type") or v.get("Name") or ""))
    for x0, k, g, t in sorted(rows):
        print("   %-8s x0=%7.1f 框=[%.1f,%.1f,%.1f,%.1f] 高=%.1f  %s"
              % (k, x0, g[0], g[1], g[2], g[3], g[3] - g[1], t))
    print("   共 %d 个" % len(rows))

    print()
    print("== 我方位号候选（desig_slots）在同一 y 带 ==")
    out = []
    for cp, sl in (asg.get("desig_slots") or {}).items():
        for s in sl:
            bx = [float(x) for x in s["box"]]
            yc = (bx[1] + bx[3]) / 2
            if lo <= yc < hi:
                out.append((bx[0], cp, s, bx))
    for x0, cp, s, bx in sorted(out):
        w, h = bx[2] - bx[0], bx[3] - bx[1]
        print("   %-6s x0=%7.1f 读 %-8s conf=%.3f 分=%7.3f 框 w=%.1f h=%.1f w/h=%.2f d=%.1f"
              % (cp, x0, s.get("text"), s.get("conf", 0), s.get("score", 0), w, h, w / h, s.get("d", 0)))
    print("   共 %d 条候选" % len(out))

    print()
    print("== 我方 `comps[cid].designator` 胜出 ==")
    for cp in sorted(asg.get("comps") or {}):
        d = (asg["comps"][cp] or {}).get("designator")
        if not d:
            continue
        bx = [float(x) for x in d["box"]]
        yc = (bx[1] + bx[3]) / 2
        if lo <= yc < hi:
            w, h = bx[2] - bx[0], bx[3] - bx[1]
            print("   %-6s → %-8s w/h=%.2f d=%.1f" % (cp, d.get("canon") or d.get("text"), w / h, d.get("d", 0)))

    print()
    print("== 融合产物里的键（x 序）==")
    fus = json.loads((ROOT / "work/_fusion_out220no4" / (cid + ".json")).read_text(encoding="utf-8"))
    fs = []
    for k, v in (fus.get("components") or {}).items():
        bb = v.get("bbox")
        yc = (bb[1] + bb[3]) / 2 if bb else -1
        if lo <= yc < hi:
            fs.append((bb[0] if bb else 0, k, v.get("Name"), v.get("value")))
    for x0, k, nm, val in sorted(fs):
        print("   %-8s x0=%7.1f Name=%s value=%s" % (k, x0, nm, val))


if __name__ == "__main__":
    main()
