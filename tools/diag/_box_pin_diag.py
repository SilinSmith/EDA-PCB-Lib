"""`box`（IC）引脚为什么只 recall 0.47 —— 一次性诊断。

用 **GT 的框**（所以元件定位误差被排除在外）问四件事：
  1. GT 引脚到**最近提取段采样点**的距离分布 —— 段到底有没有画到引脚上；
  2. GT 引脚到**自己那个框**的距离 —— 复核「框外 10px」先验；
  3. 每框的 GT 引脚数 vs 预测引脚数 —— 少了是聚类并掉了，多了是毛刺；
  4. 同一条边上相邻 GT 引脚的**间距** —— 小于 CLUSTER_R 就会被聚成一个。
"""
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")
from schematic_parser.data import DEFAULT_DATA_ROOT, discover  # noqa: E402
from stage1.pins import find_pins, sample_segments  # noqa: E402


def main():
    stage1 = sys.argv[1] if len(sys.argv) > 1 else r"Y:\workspace\EDA\work\stage1_all13"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    cases = sorted(discover(DEFAULT_DATA_ROOT), key=lambda c: c.case_id)[:limit]
    d_seg, d_box, ratio, pitch = [], [], [], []
    n_comp = 0
    for case in cases:
        sp = os.path.join(stage1, case.case_id + ".stage1.json")
        if not os.path.exists(sp):
            continue
        sd = json.load(open(sp, encoding="utf-8"))
        H = float(sd["image"]["h"])
        g = json.load(open(case.json_path, encoding="utf-8"))
        gcomp = {k: v for k, v in (g.get("components") or {}).items()
                 if (v.get("type") or "") == "box"}
        if not gcomp:
            continue
        segs = (sd.get("wire_detail") or {}).get("segments") or []
        pts = sample_segments(segs, 3.0)
        comps, gpts = [], {}
        for cid, c in gcomp.items():
            bb = c.get("bbox")
            if not bb or len(bb) != 4:
                continue
            x1, y1, x2, y2 = [float(v) for v in bb]
            comps.append({"id": cid, "type": "box", "bbox_img": [x1, H - y2, x2, H - y1]})
            pl = (g.get("pins") or {}).get(cid) or {}
            gpts[cid] = [(float(p["point"]["x"]), H - float(p["point"]["y"]))
                         for p in pl.values() if (p.get("point") or {}).get("x") is not None]
        pred = find_pins(comps, segs)
        for cid, gp in gpts.items():
            if not gp:
                continue
            n_comp += 1
            x1, y1, x2, y2 = [float(v) for v in gcomp[cid]["bbox"]]
            box = (x1, H - y2, x2, H - y1)
            G = np.asarray(gp, np.float64)
            if len(pts):
                dd = np.hypot(G[:, 0][:, None] - pts[:, 0][None, :],
                              G[:, 1][:, None] - pts[:, 1][None, :]).min(axis=1)
                d_seg.extend(dd.tolist())
            for (gx, gy) in gp:
                d_box.append(min(abs(gx - box[0]), abs(gx - box[2]),
                                 abs(gy - box[1]), abs(gy - box[3])))
            ratio.append((len(gp), len(pred.get(cid, []))))
            by = collections.defaultdict(list)
            for (gx, gy) in gp:
                dl, dr = abs(gx - x1), abs(gx - x2)
                dt, db2 = abs(gy - box[1]), abs(gy - box[3])
                m = min(dl, dr, dt, db2)
                by[3 if m == dl else 1 if m == dr else 0 if m == dt else 2].append(
                    gy if m in (dl, dr) else gx)
            for v in by.values():
                v = sorted(v)
                pitch.extend(v[i] - v[i - 1] for i in range(1, len(v)))

    def q(name, arr):
        if not arr:
            print("%-32s (no data)" % name)
            return
        a = np.asarray(arr, np.float64)
        print("%-32s n=%d p10 %.1f p25 %.1f p50 %.1f p75 %.1f p90 %.1f mean %.1f"
              % (name, len(a), *np.percentile(a, [10, 25, 50, 75, 90]), a.mean()))

    print("box 元件 %d 个" % n_comp)
    q("GT引脚 -> 最近提取段采样点", d_seg)
    if d_seg:
        a = np.asarray(d_seg)
        print("%-32s <=3px %.1f%%  <=6px %.1f%%  <=10px %.1f%%  <=20px %.1f%%"
              % ("", 100 * (a <= 3).mean(), 100 * (a <= 6).mean(),
                 100 * (a <= 10).mean(), 100 * (a <= 20).mean()))
    q("GT引脚 -> 自己那个框的边", d_box)
    if ratio:
        r = np.asarray(ratio, np.float64)
        print("每框 GT引脚 mean %.2f / 预测引脚 mean %.2f；GT>预测 的框占 %.1f%%"
              % (r[:, 0].mean(), r[:, 1].mean(), 100 * (r[:, 0] > r[:, 1]).mean()))
    if pitch:
        a = np.asarray(pitch, np.float64)
        print("同边相邻GT引脚间距 p10 %.1f p25 %.1f p50 %.1f p90 %.1f；<12px %.1f%%  <7px %.1f%%"
              % (*np.percentile(a, [10, 25, 50, 90]),
                 100 * (a < 12).mean(), 100 * (a < 7).mean()))


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()
