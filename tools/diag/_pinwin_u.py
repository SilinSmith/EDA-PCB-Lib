"""三臂几何对照：窗口里混进几条**别的引脚行**（不依赖 LLM）。

A 旧   ：u = 引脚→包围盒中心（对窄高 IC 是**斜的**）；垂直/沿线都用 `cross`（方框）
B 收窄 ：u 同 A；垂直半宽按引脚间距收到 `cp`（沿线余量不变）
C 定朝向：u = 引脚所在边的**内法向**（轴对齐）；垂直半宽 `cp`

用 GT 的引脚点当「行」的代理：窗口里落进几个**同一元件的别的引脚点**，
就是模型能看见几行可选项 —— 数字越大，报回别行名字的机会越大。

`--floor=` 改 `cp` 的下限（默认 12，与 `vlm/enrich.py` 的 `pin_cross_min` 一致）。

用法：python work\\tools\\diag\\_pinwin_u.py [--floor=12] [case ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(r"Y:\workspace\EDA\eda-pcb")))
from schematic_parser import data as D                          # noqa: E402
from vlm import enrich as E                                     # noqa: E402

CASES = "1 2 3 4 5 6 7 8 9 10".split()
FLOOR = 12.0
for a in sys.argv[1:]:
    if a.startswith("--floor="):
        FLOOR = float(a.split("=", 1)[1])


def box(pt, bbox, u, outer, inner, cross, cp):
    px, py = pt
    ux, uy = u
    ap = cross                       # 沿线余量保持旧值 ⇒ 改动只作用于垂直方向
    ax0, ay0 = px - ux * (outer + ap), py - uy * (outer + ap)
    ax1, ay1 = px + ux * (inner + ap), py + uy * (inner + ap)
    pxx, pyy = -uy * cp, ux * cp
    xs = [ax0 + pxx, ax0 - pxx, ax1 + pxx, ax1 - pxx]
    ys = [ay0 + pyy, ay0 - pyy, ay1 + pyy, ay1 - pyy]
    # 再与「旧框」取交，保证单调不超过旧框
    ox0 = min(px - ux * outer, px + ux * inner) - cross
    ox1 = max(px - ux * outer, px + ux * inner) + cross
    oy0 = min(py - uy * outer, py + uy * inner) - cross
    oy1 = max(py - uy * outer, py + uy * inner) + cross
    return (max(min(xs), ox0), max(min(ys), oy0), min(max(xs), ox1), min(max(ys), oy1))


def main() -> None:
    cases = [a for a in sys.argv[1:] if not a.startswith("--")] or CASES
    acc = {k: [0, 0, 0] for k in "ABC"}          # 混进总数 / 有混进的引脚 / 引脚数
    hts = {k: [] for k in "ABC"}
    print(f"cp 下限 = {FLOOR:.0f}px")
    print(f"{'case':>5} {'元件':>6} {'脚':>3} {'pitch':>6} {'cp':>5}  "
          f"{'A旧':>6} {'B收窄':>6} {'C定朝向':>7}")
    for cid in cases:
        try:
            case = D.find_case(cid)
            gt = json.loads(Path(case.json_path).read_text(encoding="utf-8"))
        except Exception as e:                                   # noqa: BLE001
            print(f"{cid:>5}  跳过 {type(e).__name__}")
            continue
        for gkey, gp in (gt.get("pins") or {}).items():
            pts = [(v["point"]["x"], v["point"]["y"]) for v in gp.values() if v.get("point")]
            comp = (gt.get("components") or {}).get(gkey) or {}
            bb = comp.get("bbox")
            if not bb or len(pts) < 4:
                continue
            ctr = ((bb[0] + bb[2]) / 2.0, (bb[1] + bb[3]) / 2.0)
            win = E.adaptive_window(bb, E.DEFAULTS["inner"], E.DEFAULTS["cross"],
                                    E.DEFAULTS["outer"], E.DEFAULTS["ref"])
            pitch = E._pin_pitch(pts)
            cp = min(win[2], max(FLOOR, 0.45 * pitch)) if pitch > 0 else win[2]
            cnt = {k: 0 for k in "ABC"}
            for i, q in enumerate(pts):
                dx, dy = ctr[0] - q[0], ctr[1] - q[1]
                n = (dx * dx + dy * dy) ** 0.5 or 1.0
                uA = (dx / n, dy / n)
                uC = E._edge_dir(bb, q)
                boxes = {"A": box(q, bb, uA, win[0], win[1], win[2], win[2]),
                         "B": box(q, bb, uA, win[0], win[1], win[2], cp),
                         "C": box(q, bb, uC, win[0], win[1], win[2], cp)}
                for k, b in boxes.items():
                    c = sum(1 for j, r in enumerate(pts)
                            if j != i and b[0] <= r[0] <= b[2] and b[1] <= r[1] <= b[3])
                    cnt[k] += c
                    acc[k][0] += c
                    acc[k][1] += (c > 0)
                    acc[k][2] += 1
                    hts[k].append(abs(b[3] - b[1]))
            n = len(pts)
            print(f"{cid:>5} {gkey:>6} {n:>3} {pitch:6.1f} {cp:5.1f}  "
                  f"{cnt['A']/n:6.2f} {cnt['B']/n:6.2f} {cnt['C']/n:7.2f}")
    print("-" * 66)
    for k, name in (("A", "A 旧(斜向 u, cross 方框)  "), ("B", "B 收窄(斜向 u)          "),
                    ("C", "C 定朝向(边法向) + 收窄 ")):
        t, ki, n = acc[k]
        med = sorted(hts[k])[len(hts[k]) // 2] if hts[k] else 0
        print(f"{name}: 平均混进别的引脚 {t/max(1,n):.2f} 个；有混进 {ki}/{n} = {ki/max(1,n):.1%}；"
              f"窗口中位高 {med:.0f}px")


if __name__ == "__main__":
    main()
