# -*- coding: utf-8 -*-
"""量「碎片网到别的线段」的缺口，用来判断并网容差该怎么定。

对 `runs/stage1/<case>.stage1.json` 里每个成员数较少的网，取它自己的所有端点，
到**别的段**上算「端点到线段内部」的距离（只认投影参数落在 (0.02, 0.98) 的 T 型/十字），
报出最小的一对。若大多数缺口都 <= 20px，说明把全局 `WIRE_TOL` 抬到 20 就能救，
但那边实测会崩（55.89 -> 51.01）=> 需要「按方向/交叉」的局部规则而不是全局容差。

用法：
    python work\\tools\\diag\\_segap.py --dir work\\_fast_out_wtol_def --maxpins 3
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(ROOT, "work", "_fast_out_wtol_def"))
    ap.add_argument("--cases", default="")
    ap.add_argument("--maxpins", type=int, default=3, help="只看成员数 <= 这个值的网")
    ap.add_argument("--show", type=int, default=60)
    a = ap.parse_args()

    sdir = os.path.join(a.dir, "runs", "stage1")
    if not os.path.isdir(sdir):
        print("没有目录", sdir)
        return 1
    cases = [c.strip() for c in a.cases.split(",") if c.strip()] or \
        sorted(fn[:-len(".stage1.json")] for fn in os.listdir(sdir) if fn.endswith(".stage1.json"))

    buckets = {}
    rows = []
    for case in cases:
        p = os.path.join(sdir, case + ".stage1.json")
        if not os.path.exists(p):
            continue
        s1 = json.load(open(p, encoding="utf-8"))
        raw = (s1.get("wire_detail") or {}).get("segments") or []
        segs = []
        for s in raw:
            arr = np.asarray(s, np.float64).reshape(-1, 2)
            if len(arr) < 2:
                continue
            segs.append(np.concatenate([arr[0], arr[-1]]))
        for n in (s1.get("nets") or []):
            pins = n.get("pins") or []
            if not pins or len(pins) > a.maxpins:
                continue
            idx = [int(i) for i in (n.get("seg_index") or []) if 0 <= int(i) < len(segs)]
            if not idx:
                continue
            ends = []
            for i in idx:
                ends.append(segs[i][:2])
                ends.append(segs[i][2:])
            best = (1e9, None, None)
            for j, sg in enumerate(segs):
                if j in idx:
                    continue
                aa, bb = sg[:2], sg[2:]
                ab = bb - aa
                L2 = max(1e-9, float(ab @ ab))
                for pt in ends:
                    t = float((pt - aa) @ ab / L2)
                    if t < 0.02 or t > 0.98:
                        continue
                    d = float(np.hypot(*(pt - (aa + t * ab))))
                    if d < best[0]:
                        best = (d, j, round(t, 2))
            if best[0] > 1e8:
                continue
            b = int(best[0] // 5.0) * 5
            buckets[b] = buckets.get(b, 0) + 1
            rows.append((best[0], case, n.get("id"), len(pins), len(idx), best[1], best[2]))

    rows.sort(reverse=True)
    print("缺口直方图（端点到别的线段内部的最小距离，5px 一桶）：")
    for b in sorted(buckets):
        print("  %3d~%3dpx : %d" % (b, b + 4, buckets[b]))
    tot = len(rows)
    if tot:
        for lim in (2, 5, 8, 10, 12, 15, 20, 30):
            k = sum(1 for r in rows if r[0] <= lim)
            print("  <= %-3dpx 可救 %3d / %d (%.1f%%)" % (lim, k, tot, 100.0 * k / tot))
    print("\n最大的 %d 个缺口：" % a.show)
    for r in rows[:a.show]:
        print("  %6.1fpx  case %-4s net %-7s npins=%d nseg=%-3d -> seg%s t=%s" % r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
