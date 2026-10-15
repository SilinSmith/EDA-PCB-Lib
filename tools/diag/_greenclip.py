# -*- coding: utf-8 -*-
"""把网线端点裁到「绿线起点」（离线试算，不跑管线）。

由来（case 2 的 R3 证据，§5.95 的后续取证）：
    GT net_4 = (R3.1, ∅177.175)，GT 边 (894,626)-(894,582)；
    R3.1 的 GT 脚点是 (894,641)（= bbox 下边 651 外 10px），∅177 的脚点是 (896,580)。
    ⇒ GT 的网线**只画绿线的起止**，不含元件自己的引线（引线是暗红）；
      我们的导线把引线也吞进去（15px）。
GT 网线中点颜色统计（10GT 全量、1917 个采样）：green 1463 / dark 317 / red 116 / light 21
⇒ 做一条**只对绿线生效**的保守规则：
    端点自身不是绿像素时，沿 E→O 方向逐像素走 ≤NEAR px，
    第一个「连续 ≥RUN 个像素为绿」的位置视为绿线起点，把端点挪过去；找不到就不动。

用法：
    python -X utf8 work/_greenclip.py --run work/_fast_out_pw20on --out greenclip [--near 45]
然后在 eda-pcb 下打分：
    python -X utf8 tools\eval\fast.py --score-only --cases 1,2,3,4,5,6,7,8,9,10 --out greenclip
"""
import argparse
import json
import math
import shutil
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[3]          # Y:\workspace\EDA（tools/diag/ 的约定）


def _green(px):
    b, g, r = int(px[0]), int(px[1]), int(px[2])
    return g > r + 20 and g > b + 20


def _px(img, x, y):
    h, w = img.shape[:2]
    xi, yi = int(round(x)), int(round(y))
    if xi < 0 or yi < 0 or xi >= w or yi >= h:
        return None
    return img[yi, xi]


def clip_doc(doc, img, near=45.0, run=3, min_shift=2.0):
    if img is None:
        return []
    h = img.shape[0]
    moves = []
    for nid, net in (doc.get("nets") or {}).items():
        for eid, raw in (net.get("edges") or {}).items():
            pts = [((float(q["x"]), float(q["y"])) if isinstance(q, dict)
                    else (float(q[0]), float(q[1]))) for q in raw]
            if len(pts) < 2:
                continue
            for end in (0, len(pts) - 1):
                other = pts[-1] if end == 0 else pts[0]
                ex, ey = pts[end]
                # json(左下原点) -> 图像(左上原点)
                ei = (ex, h - ey)
                oi = (other[0], h - other[1])
                if _px(img, *ei) is None:
                    continue
                if _green(_px(img, *ei)):
                    continue                     # 端点本来就在绿线上，不动
                dx, dy = oi[0] - ei[0], oi[1] - ei[1]
                L = math.hypot(dx, dy)
                if L < 1e-6:
                    continue
                ux, uy = dx / L, dy / L
                found = None
                k = 1.0
                while k <= near:
                    ok = True
                    for j in range(run):
                        p = _px(img, ei[0] + ux * (k + j), ei[1] + uy * (k + j))
                        if p is None or not _green(p):
                            ok = False
                            break
                    if ok:
                        found = k
                        break
                    k += 1.0
                if found is None or found < min_shift:
                    continue
                nx, ny = ei[0] + ux * found, ei[1] + uy * found
                moves.append((nid, eid, end, round(ex, 1), round(ey, 1),
                              round(nx, 1), round(h - ny, 1), round(found, 1)))
                pts[end] = (nx, h - ny)
            for i, q in enumerate(raw):
                if isinstance(q, dict):
                    q["x"], q["y"] = pts[i]
                else:
                    raw[i] = [pts[i][0], pts[i][1]]
    return moves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=r"work\_fast_out_pw20on")
    ap.add_argument("--in", dest="indir", default=r"work\_fast_in")
    ap.add_argument("--out", default="greenclip")
    ap.add_argument("--cases", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--near", type=float, default=45.0)
    ap.add_argument("--runlen", type=int, default=3)
    a = ap.parse_args()
    src = ROOT / a.run / "final_results"
    dst = ROOT / "work" / ("_fast_out_" + a.out) / "final_results"
    dst.mkdir(parents=True, exist_ok=True)
    total = 0
    for cid in [c.strip() for c in a.cases.split(",") if c.strip()]:
        p = src / ("%s_Pred.json" % cid)
        if not p.exists():
            print("! 缺 %s" % p)
            continue
        doc = json.loads(p.read_text(encoding="utf-8"))
        img = cv2.imread(str(ROOT / a.indir / ("%s.png" % cid)))
        moves = clip_doc(doc, img, near=a.near, run=a.runlen)
        (dst / p.name).write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
        total += len(moves)
        print("case %-4s 裁 %2d 个端点%s" % (cid, len(moves),
              ("  例：" + str(moves[:2])) if moves else ""))
    for extra in ("runs",):
        s = ROOT / a.run / extra
        if s.is_dir():
            t = ROOT / "work" / ("_fast_out_" + a.out) / extra
            if t.exists():
                shutil.rmtree(t)
            shutil.copytree(s, t)
    print("合计裁 %d 个端点 -> %s" % (total, dst))


if __name__ == "__main__":
    main()
