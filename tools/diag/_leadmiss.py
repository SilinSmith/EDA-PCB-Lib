# -*- coding: utf-8 -*-
"""**两脚元件引脚候选审计**：为什么 r/c/l/d/led 的引脚总是挂不上网（`_netdiff.py` 的
`fn:r 41 / fn:c 27 / fn:l 12 / fn:d 2 / fn:led 5` 合计 87 个成员缺口）。

case 7 的 R1 是典型：GT `net_8 = (R1.1, U1.5)`、`net_5 = (C8.1, R1.2)`，我们却输出
`net_15=(R1.1)` 与 `net_16=(R1.2)` 两条**单成员网**。逐点看：
* GT 引脚（JSON 帧）`(1072,660)`/`(1112,660)` ⇒ 图像帧 `(1072,343)`/`(1112,343)`，
  框 `bbox_img=[1084.5,340,1104.5,348]`（20×8，**横放**，引脚在左右）；
* 真正接到引脚上的导线是 `seg 78 (964,343)-(1075,342)` 与 `seg 76 (1244.5,344)-(1113,345)`
  —— 端点离框边 **9.7 / 8.6 px**，而 `pins.py` 的 `TOUCH_TOL = 6` ⇒ **判不了"接触"**；
* 于是候选落到了框上下两侧的**噪声竖线段**（`seg 96/97/98/100`）端点上 ⇒ 候选取点
  在错的轴上（`(1086.8,330)`/`(1091.2,358)`）⇒ 引脚挂错网、单成员网遍地。

本探针逐 **GT 两脚元件引脚**量三件事（官方 10 张，只读）：
1. `d_cand`：我方 stage1 候选点里离 GT 引脚最近的点的距离（>6 就是「没接上」）；
2. `axis`：该最近候选相对框中心是**沿 GT 引脚轴**还是**垂直轴**；
3. `wire`：GT 引脚点附近（≤ `--wire-tol`）有没有导线端点，以及那个端点的**外伸距离**
   （用来验证 `TOUCH_TOL=6` 是否卡掉了真导线）。

    python -X utf8 eda-pcb/tools/diag/_leadmiss.py --run work/_fast_out_fin --split 10GT
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))

from schematic_parser.data import DEFAULT_DATA_ROOT, discover                     # noqa: E402

TWO_PIN = frozenset({"r", "c", "l", "d", "led", "fuse", "crystal_2pin", "crystal"})
IC_TYPES = frozenset({"box", "boxic", "ic"})


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def comps_of(doc):
    cs = doc.get("components") or {}
    if isinstance(cs, dict):
        return cs
    return {str(v.get("id")): v for v in cs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=r"Y:\workspace\EDA\work\_fast_out_fin")
    ap.add_argument("--split", default="10GT")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--case", default="")
    ap.add_argument("--wire-tol", type=float, default=20.0)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    run = Path(a.run)
    if not run.is_absolute():
        run = (ROOT / run).resolve()
    cases = [c for c in discover(DEFAULT_DATA_ROOT) if c.split == a.split]
    if a.case:
        cases = [c for c in cases if c.case_id == a.case]
    tot = collections.Counter()
    for c in cases[:a.n]:
        st_p = run / "runs" / "stage1" / ("%s.stage1.json" % c.case_id)
        if not st_p.exists():
            continue
        st = load(st_p)
        H = float((st.get("image") or {}).get("h") or 0.0)
        gt = load(Path(c.json_path))
        gtc = comps_of(gt)
        stc = {x["id"]: x for x in (st.get("components") or [])}
        stpins = st.get("pins") or {}
        segs = ((st.get("wire_detail") or {}).get("segments") or [])
        # GT 元件 → 我方 stage1 元件（框中心最近、<40px）
        match = {}
        for name, g in gtc.items():
            if str(g.get("type") or "").lower() not in TWO_PIN:
                continue
            bb = g.get("bbox") or []
            if len(bb) != 4:
                continue
            gx, gy = (bb[0] + bb[2]) / 2.0, H - (bb[1] + bb[3]) / 2.0
            best = None
            for cid, s in stc.items():
                sb = s.get("bbox_img") or []
                if len(sb) != 4:
                    continue
                d = math.hypot((sb[0] + sb[2]) / 2.0 - gx, (sb[1] + sb[3]) / 2.0 - gy)
                if d < 40.0 and (best is None or d < best[0]):
                    best = (d, cid)
            if best:
                match[name] = best[1]
        stat = collections.Counter()
        for name, g in gtc.items():
            if str(g.get("type") or "").lower() not in TWO_PIN:
                continue
            cid = match.get(name)
            if not cid:
                stat["no_match"] += 1
                continue
            s = stc[cid]
            cands = [tuple(p) for p in (stpins.get(cid) or [])]
            sb = s.get("bbox_img")
            cx, cy = (sb[0] + sb[2]) / 2.0, (sb[1] + sb[3]) / 2.0
            gp = (gt.get("pins") or {}).get(name) or {}
            gpts = []
            for pk, pv in gp.items():
                pt = (pv or {}).get("point") or {}
                if "x" in pt:
                    gpts.append(((pt["x"], H - pt["y"]), pk))
            if len(gpts) < 2:
                stat["gt_pins_lt2"] += 1
                continue
            # GT 引脚轴
            (x1, y1), (x2, y2) = gpts[0][0], gpts[1][0]
            axis_x = abs(x2 - x1) >= abs(y2 - y1)
            for (gpx, gpy), pk in gpts:
                stat["pins"] += 1
                row = gpy if axis_x else gpx
                perp = gpx if axis_x else gpy
                # GT 外伸
                if axis_x:
                    e1, e2 = sb[0], sb[2]
                else:
                    e1, e2 = sb[1], sb[3]
                out_gt = max(e1 - perp, perp - e2, 0.0)
                # 最近候选
                d_cand, ax_ok = 1e9, False
                for cp in cands:
                    d = math.hypot(cp[0] - gpx, cp[1] - gpy)
                    if d < d_cand:
                        d_cand = d
                        if axis_x:
                            ax_ok = abs(cp[0] - cx) >= abs(cp[1] - cy)
                        else:
                            ax_ok = abs(cp[1] - cy) >= abs(cp[0] - cx)
                # 引脚点附近的导线端点
                w_best = None
                for sg in segs:
                    for ex, ey in ((sg[0][0], sg[0][1]), (sg[1][0], sg[1][1])):
                        d = math.hypot(ex - gpx, ey - gpy)
                        if d <= a.wire_tol and (w_best is None or d < w_best[0]):
                            wo = max(e1 - ex, ex - e2) if axis_x else max(e1 - ey, ey - e2)
                            w_best = (d, wo)
                if d_cand <= 6.0:
                    stat["cand_attached"] += 1
                    if ax_ok:
                        stat["cand_on_axis"] += 1
                    else:
                        stat["cand_off_axis_hit"] += 1
                else:
                    stat["cand_far"] += 1
                    if w_best:
                        stat["cand_far_but_wire_near"] += 1
                        if w_best[1] > 6.0:
                            stat["cand_far_wire_out_gt6"] += 1
                    else:
                        stat["cand_far_no_wire"] += 1
                if a.verbose and d_cand > 6.0 and w_best:
                    print("     %-5s %-4s out_gt %5.1f 最近候选距 %5.1f ｜ 导线端点 %5.1f px、"
                          "外伸 %5.1f" % (name, pk, out_gt, d_cand, w_best[0], w_best[1]))
        print("  case %-6s 两脚元件 %2d ｜ 引脚 %3d ｜ 候选贴上 %3d（其中在轴 %3d / 错轴 %3d）"
              " ｜ 候选太远 %3d（**附近有导线 %3d，其中外伸>6px %3d**）｜ 无导线 %3d"
              % (c.case_id, sum(1 for n, g in gtc.items()
                                if str(g.get("type") or "").lower() in TWO_PIN and n in match),
                 stat["pins"], stat["cand_attached"], stat["cand_on_axis"],
                 stat["cand_off_axis_hit"], stat["cand_far"],
                 stat["cand_far_but_wire_near"], stat["cand_far_wire_out_gt6"],
                 stat["cand_far_no_wire"]))
        tot.update(stat)
    print("\n== 汇总（官方 %s）==" % a.split)
    print("  引脚 %d ｜ 候选贴上 %d（在轴 %d / 错轴 %d）｜ 候选太远 %d（附近有导线 %d，"
          "其中**导线外伸 > TOUCH_TOL=6** %d）｜ 附近无导线 %d"
          % (tot["pins"], tot["cand_attached"], tot["cand_on_axis"], tot["cand_off_axis_hit"],
             tot["cand_far"], tot["cand_far_but_wire_near"], tot["cand_far_wire_out_gt6"],
             tot["cand_far_no_wire"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
