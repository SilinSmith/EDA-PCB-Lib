# -*- coding: utf-8 -*-
"""把网线端点吸到「电源/地符号」的连接点上（离线试算，不跑管线）。

由来（work/_c2overlay.png / work/_pwrclip 的取证）：case 2 的 GT net_12 = (U2.2, ∅144.142)，
GT 边 (933,787)-(1021,787)，我们的是 (907,785.81)-(1020,786.52) —— **我们多画了 26px**：
我们的导线把电源符号自己那截引线也圈进来了，GT 的网线从符号的连接点起画。

连接点的规律（case 1/2 实测 7 个 ∅ 元件，全部吻合）：
    bbox 四边各向外 10px、居中的那个点就是它的脚点
    ∅202 bbox[1074,286,1084,296] pin(1079,276)  -> 上边外 10
    ∅109 bbox[450,425,470,444]  pin(460,454)    -> 下边外 10
    ∅104 bbox[1096,423,1115,443] pin(1125,433)  -> 右边外 10
    ∅144 bbox[905,781,915,791]  pin(925,786)    -> 右边外 10
所以对每个 gnd/v 元件生成 4 个候选点，谁落在边上、且在端点内侧，就把端点吸过去。

用法：
    python -X utf8 work/_pwrclip.py --run work/_fast_out_pnrn --out pwrclip [--near 45] [--perp 4]
然后在 eda-pcb 下打分：
    python -X utf8 tools\eval\fast.py --score-only --cases 1,2,3,4,5,6,7,8,9,10 --out pwrclip
"""
import argparse
import json
import math
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]          # Y:\workspace\EDA（tools/diag/ 的约定）
PWR = ("v", "gnd", "ground", "power", "earth")


def _canon(t):
    t = str(t or "").strip().lower()
    return {"ground": "gnd", "power": "v", "earth": "gnd"}.get(t, t)


def _cands(doc):
    out = []
    for key, c in (doc.get("components") or {}).items():
        if _canon(c.get("type")) not in PWR:
            continue
        try:
            x0, y0, x1, y1 = [float(v) for v in c["bbox"]]
        except Exception:
            continue
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        out.append((cx, y1 + 10.0, key, (x0, y0, x1, y1)))
        out.append((cx, y0 - 10.0, key, (x0, y0, x1, y1)))
        out.append((x0 - 10.0, cy, key, (x0, y0, x1, y1)))
        out.append((x1 + 10.0, cy, key, (x0, y0, x1, y1)))
    return out


def _box_dist(p, bb, tol):
    """点到 bbox（外扩 tol）的距离；在框内或外扩范围内返回 <=tol 的值。"""
    x0, y0, x1, y1 = bb
    dx = max(x0 - p[0], 0.0, p[0] - x1)
    dy = max(y0 - p[1], 0.0, p[1] - y1)
    return math.hypot(dx, dy)


def clip_doc(doc, near=45.0, perp=4.0, boxtol=None):
    cands = _cands(doc)
    moves = []
    for nid, net in (doc.get("nets") or {}).items():
        for eid, raw in (net.get("edges") or {}).items():
            pts = [((float(q["x"]), float(q["y"])) if isinstance(q, dict)
                    else (float(q[0]), float(q[1]))) for q in raw]
            if len(pts) < 2:
                continue
            for end in (0, len(pts) - 1):
                E, O = pts[end], pts[1 - end if end == 0 else -2]
                dx, dy = O[0] - E[0], O[1] - E[1]
                L = math.hypot(dx, dy)
                if L < 1e-6:
                    continue
                ux, uy = dx / L, dy / L
                best = None
                for px, py, key, bb in cands:
                    vx, vy = px - E[0], py - E[1]
                    t = vx * ux + vy * uy                 # 沿边方向的距离
                    if t <= 0.5 or t > near:
                        continue
                    off = abs(vx * (-uy) + vy * ux)       # 垂距
                    if off > perp:
                        continue
                    if boxtol is not None and _box_dist(E, bb, boxtol) > boxtol:
                        continue                          # 端点必须贴在这个电源符号上
                    if best is None or t < best[0]:
                        best = (t, (px, py), key, off)
                if best is not None:
                    moves.append((nid, eid, end, (round(E[0], 2), round(E[1], 2)),
                                  (round(best[1][0], 2), round(best[1][1], 2)),
                                  best[2], round(best[0], 1), round(best[3], 2)))
                    pts[end] = best[1]
            net["edges"][eid] = [{"x": round(x, 2), "y": round(y, 2)} for x, y in pts]
    return moves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=r"work\_fast_out_pnrn")
    ap.add_argument("--out", default="pwrclip")
    ap.add_argument("--cases", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--near", type=float, default=45.0)
    ap.add_argument("--perp", type=float, default=4.0)
    ap.add_argument("--boxtol", type=float, default=None,
                    help="只吸「端点贴在电源符号框上（外扩这么多 px 内）」的那些端点")
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
        moves = clip_doc(doc, near=a.near, perp=a.perp, boxtol=a.boxtol)
        (dst / p.name).write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
        total += len(moves)
        print("case %-4s 吸端点 %2d 个%s" % (cid, len(moves),
              ("  例：" + str(moves[:2])) if moves else ""))
    for extra in ("runs",):
        s = ROOT / a.run / extra
        if s.is_dir():
            t = ROOT / "work" / ("_fast_out_" + a.out) / extra
            if t.exists():
                shutil.rmtree(t)
            shutil.copytree(s, t)
    print("合计吸 %d 个端点 -> %s" % (total, dst))


if __name__ == "__main__":
    main()
