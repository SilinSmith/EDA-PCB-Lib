# -*- coding: utf-8 -*-
"""**前端召回**：在完全不看 LLM 的前提下，stage1 产物里到底「有没有」GT 要的东西。

对每张 case 量三件事：
  ① 元件：GT 的每个框，stage1 里有没有一个框的角点距 <= bbox_tol(20px)；
     还有 IoU>=0.5 的口径；以及「连 40px / 80px 内都没有」的**不可恢复**数。
  ② 引脚：GT 的每个引脚点，stage1 里有没有一个引脚点在 5px 内（官方 pin_tol）。
  ③ 供给量：stage1 一共吐了多少个元件框 / 引脚点（用来对照「漏」还是「没提」）。

坐标：GT 的 `bbox` 与 `point` 是 **JSON 空间（y 向上）**，stage1 的 `bbox_img` 是
图像空间 ⇒ 统一翻到图像空间再比。

    python work\\tools\\diag\\_frontrecall.py [case ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(r"Y:\workspace\EDA")
sys.path.insert(0, str(ROOT / "eda-pcb"))
from schematic_parser import data as D                       # noqa: E402
from stage2.text import imread_unicode                       # noqa: E402

S1 = ROOT / "work" / "_submit_out220r36" / "runs" / "stage1"
TOL = 20.0


def cd(a, b):
    return max(abs(a[i] - b[i]) for i in range(4))


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def s1_boxes(doc, H):
    out = []
    for c in doc.get("components") or []:
        b = c.get("bbox_img")
        if b and len(b) == 4:
            out.append((c.get("type") or "?", [float(v) for v in b]))
    return out


def _pt_of(v):
    """从各种可能的写法里抠出一个点：(x, y)。"""
    if isinstance(v, dict):
        p = v.get("point") or v.get("pt") or v.get("xy")
        if isinstance(p, dict):
            return (float(p["x"]), float(p["y"]))
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            return (float(p[0]), float(p[1]))
        return None
    if isinstance(v, (list, tuple)) and len(v) >= 2 and not isinstance(v[0], (list, tuple, dict)):
        return (float(v[0]), float(v[1]))
    return None


def s1_pins(doc, H):
    """stage1 的 `pins` 实测形状：`{cid: [[x, y], ...]}`（图像空间）。"""
    pts = []
    p = doc.get("pins")

    def eat(v):
        if isinstance(v, dict):
            for vv in v.values():
                eat(vv)
        elif isinstance(v, (list, tuple)):
            q = _pt_of(v)
            if q is not None:
                pts.append(q)
            else:
                for vv in v:
                    eat(vv)

    if p:
        eat(p)
    return pts


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    cases = sys.argv[1:] or ["1", "2", "3", "4", "5", "7", "8", "9", "10",
                             "0092", "0048", "0019", "0024", "0075", "0086",
                             "0002", "0001", "0042", "0103"]
    agg = {}
    tot = {"gt": 0, "hit20": 0, "hit40": 0, "hit80": 0, "miss": 0,
           "iou50": 0, "s1": 0, "gtpin": 0, "pin5": 0, "pin15": 0,
           "pinmiss": 0, "s1pin": 0}
    for cid in cases:
        case = D.find_case(cid)
        gt = json.loads(Path(case.json_path).read_text(encoding="utf-8"))
        sp = S1 / f"{case.case_id}.stage1.json"
        if not sp.exists():
            print(f"  [{cid}] 缺 stage1 产物，跳过"); continue
        doc = json.loads(sp.read_text(encoding="utf-8"))
        bgr = imread_unicode(case.png_path)
        H = bgr.shape[0]
        boxes = s1_boxes(doc, H)
        pts = s1_pins(doc, H)
        tot["s1"] += len(boxes)
        tot["s1pin"] += len(pts)

        row = {}
        for k, g in (gt.get("components") or {}).items():
            gb = g.get("bbox")
            if not gb or len(gb) != 4:
                continue
            gb = [float(v) for v in gb]
            gi = (gb[0], H - gb[3], gb[2], H - gb[1])
            t = g.get("type") or "?"
            near = [cd(gi, b) for _t, b in boxes]
            d = min(near) if near else 1e9
            best_iou = max((iou(gi, b) for _t, b in boxes), default=0.0)
            r = agg.setdefault(t, {"gt": 0, "hit20": 0, "miss": 0, "iou50": 0})
            tot["gt"] += 1; r["gt"] += 1
            if d <= TOL:
                tot["hit20"] += 1; r["hit20"] += 1
            if d <= 40:
                tot["hit40"] += 1
            if d <= 80:
                tot["hit80"] += 1
            else:
                tot["miss"] += 1; r["miss"] += 1
            if best_iou >= 0.5:
                tot["iou50"] += 1; r["iou50"] += 1
            row[t] = row.get(t, 0) + 1

        for k, gp in (gt.get("pins") or {}).items():
            for _pn, pv in gp.items():
                pt = pv.get("point")
                if not pt:
                    continue
                x = float(pt["x"]); y = H - float(pt["y"])
                tot["gtpin"] += 1
                if not pts:
                    tot["pinmiss"] += 1; continue
                d = min(((x - a) ** 2 + (y - b) ** 2) ** 0.5 for a, b in pts)
                if d <= 5.0:
                    tot["pin5"] += 1
                elif d <= 15.0:
                    tot["pin15"] += 1
                else:
                    tot["pinmiss"] += 1
        print(f"  [{cid}] GT 元件 {sum(row.values())} / stage1 框 {len(boxes)}"
              f" | GT 引脚 {tot['gtpin']} / stage1 点 {len(pts)}")

    print("\n================ 前端召回（stage1 产物，无 LLM） ================")
    g = tot["gt"] or 1
    print(f"元件：GT {tot['gt']} 个")
    print(f"  角点 <=20px（官方 bbox_tol）   {tot['hit20']:5d}  = {tot['hit20']/g:.3f}")
    print(f"  IoU >=0.5                     {tot['iou50']:5d}  = {tot['iou50']/g:.3f}")
    print(f"  角点 <=40px                   {tot['hit40']:5d}  = {tot['hit40']/g:.3f}")
    print(f"  角点 <=80px                   {tot['hit80']:5d}  = {tot['hit80']/g:.3f}")
    print(f"  **80px 内都没有（不可恢复）** {tot['miss']:5d}  = {tot['miss']/g:.3f}")
    print(f"  stage1 一共吐了 {tot['s1']} 个框（GT 是 {tot['gt']}）")
    p = tot["gtpin"] or 1
    print(f"引脚：GT {tot['gtpin']} 个")
    print(f"  5px 内（官方 pin_tol）        {tot['pin5']:5d}  = {tot['pin5']/p:.3f}")
    print(f"  5~15px                       {tot['pin15']:5d}  = {tot['pin15']/p:.3f}")
    print(f"  **>15px（不可恢复）**         {tot['pinmiss']:5d}  = {tot['pinmiss']/p:.3f}")
    print(f"  stage1 一共吐了 {tot['s1pin']} 个引脚点（GT 是 {tot['gtpin']}）")

    print("\n按类（角点 <=20px 命中 / GT 数 / 80px 内都没有）：")
    for t, r in sorted(agg.items(), key=lambda kv: -kv[1]["gt"]):
        print(f"  {t:18s} {r['hit20']:4d}/{r['gt']:<4d} = {r['hit20']/r['gt']:.2f}"
              f"   漏{ r['miss']:4d}   IoU>=.5 {r['iou50']:4d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
