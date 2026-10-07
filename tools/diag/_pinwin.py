"""不依赖 LLM 的几何判据：引脚窗口里**混进了几条别的引脚行**。

动机：`vlm/enrich.py` 的引脚窗口旧实现把 `cross` 同时加在 x 和 y 上，垂直总高
`2×cross`（缩放后 40~48px），而 QFP 引脚间距常 19~30px ⇒ 窗口必然框住相邻引脚。
模型因此能看见 2~3 行编号/名称，报回**别的行**的名字（10GT 引脚错 55 处里 41 处是
name_mismatch：位置对、名字错）。

这里用 GT 的引脚点当「行」的代理：对每个引脚，算它的窗口（旧几何 / 新几何）覆盖了
**同一元件里另外几个引脚的点**。旧窗口覆盖多 = 模型有多行可选 = 名字会串。

用法：
    python work\\tools\\diag\\_pinwin.py [case ...]          默认 10GT 全部
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(r"Y:\workspace\EDA\eda-pcb")))
from schematic_parser import data as D                          # noqa: E402
from vlm import enrich as E                                     # noqa: E402

CASES = "1 2 3 4 5 6 7 8 9 10".split()


def box_of(bgr_shape, pt, ctr, outer, inner, cross, cp):
    """新几何：**沿线余量保持旧值 `cross`**，只把垂直半宽收到 `cp`，再与旧框取交
    （与 `vlm/enrich.py:crop_dir` 的实现一致）。"""
    h, w = bgr_shape
    px, py = float(pt[0]), float(pt[1])
    dx, dy = float(ctr[0]) - px, float(ctr[1]) - py
    n = (dx * dx + dy * dy) ** 0.5
    if n < 1e-3:
        dx, dy, n = 1.0, 0.0, 1.0
    ux, uy = dx / n, dy / n
    ap = float(cross)
    ax0, ay0 = px - ux * (outer + ap), py - uy * (outer + ap)
    ax1, ay1 = px + ux * (inner + ap), py + uy * (inner + ap)
    pxx, pyy = -uy * cp, ux * cp
    xs = [ax0 + pxx, ax0 - pxx, ax1 + pxx, ax1 - pxx]
    ys = [ay0 + pyy, ay0 - pyy, ay1 + pyy, ay1 - pyy]
    ox = box_of_old(bgr_shape, pt, ctr, outer, inner, cross)
    return (max(min(xs), ox[0]), max(min(ys), ox[1]),
            min(max(xs), ox[2]), min(max(ys), ox[3]))


def box_of_old(bgr_shape, pt, ctr, outer, inner, cross):
    h, w = bgr_shape
    px, py = float(pt[0]), float(pt[1])
    dx, dy = float(ctr[0]) - px, float(ctr[1]) - py
    n = (dx * dx + dy * dy) ** 0.5
    if n < 1e-3:
        dx, dy, n = 1.0, 0.0, 1.0
    ux, uy = dx / n, dy / n
    ax0, ay0 = px - ux * outer, py - uy * outer
    ax1, ay1 = px + ux * inner, py + uy * inner
    return (min(ax0, ax1) - cross, min(ay0, ay1) - cross,
            max(ax0, ax1) + cross, max(ay0, ay1) + cross)


def main() -> None:
    cases = [a for a in sys.argv[1:] if not a.startswith("--")] or CASES
    tot = {"old": [0, 0, 0], "new": [0, 0, 0]}     # 覆盖的别的引脚总数 / 有覆盖的引脚数 / 引脚数
    print(f"{'case':>5} {'元件':>6} {'脚':>3} {'pitch':>6} {'cp':>5} "
          f"{'旧:别人行/脚':>13} {'新:别人行/脚':>13}")
    for cid in cases:
        try:
            case = D.find_case(cid)
            gt = json.loads(Path(case.json_path).read_text(encoding="utf-8"))
        except Exception as e:                                   # noqa: BLE001
            print(f"{cid:>5}  跳过 {type(e).__name__}")
            continue
        H = None
        for gkey, gp in (gt.get("pins") or {}).items():
            pts = [(v["point"]["x"], v["point"]["y"]) for v in gp.values() if v.get("point")]
            if len(pts) < 4:
                continue
            comp = (gt.get("components") or {}).get(gkey) or {}
            bb = comp.get("bbox")
            if not bb:
                continue
            ctr = ((bb[0] + bb[2]) / 2.0, (bb[1] + bb[3]) / 2.0)
            win = E.adaptive_window(bb, E.DEFAULTS["inner"], E.DEFAULTS["cross"],
                                    E.DEFAULTS["outer"], E.DEFAULTS["ref"])
            pitch = E._pin_pitch(pts)
            cp = min(win[2], max(6.0, 0.45 * pitch)) if pitch > 0 else win[2]
            if H is None:
                from stage2.text import imread_unicode
                H = imread_unicode(case.png_path).shape[:2]
            o_other = n_other = p_n = 0
            for i, q in enumerate(pts):
                bo = box_of_old(H, q, ctr, win[0], win[1], win[2])
                bn = box_of(H, q, ctr, win[0], win[1], win[2], cp)
                co = sum(1 for j, r in enumerate(pts) if j != i and bo[0] <= r[0] <= bo[2] and bo[1] <= r[1] <= bo[3])
                cn = sum(1 for j, r in enumerate(pts) if j != i and bn[0] <= r[0] <= bn[2] and bn[1] <= r[1] <= bn[3])
                o_other += co; n_other += cn; p_n += 1
                tot["old"][0] += co; tot["old"][1] += (co > 0)
                tot["new"][0] += cn; tot["new"][1] += (cn > 0)
            tot["old"][2] += p_n; tot["new"][2] += p_n
            print(f"{cid:>5} {gkey:>6} {p_n:>3} {pitch:6.1f} {cp:5.1f} "
                  f"{o_other/p_n:13.2f} {n_other/p_n:13.2f}")
    for tag in ("old", "new"):
        t, k, n = tot[tag]
        print(f"[{ '旧' if tag=='old' else '新' }] 平均窗口里混进别的引脚 {t/max(1,n):.2f} 个；"
              f"有混进的引脚 {k}/{n} = {k/max(1,n):.1%}")


if __name__ == "__main__":
    main()
