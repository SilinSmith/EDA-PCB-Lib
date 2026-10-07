"""判定 GT 框的 y 方向：对每个 case 分别按「y 向上（翻转）」与「y 向下（不翻转）」
两种假设量框内墨迹占比，看哪种高。顺手统计「零墨迹」的框。

用法:
    python work\tools\diag\_axistest.py --splits 10GT --by-type
    python work\tools\diag\_axistest.py --case 10 --render R1
"""
import argparse
import collections
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")
sys.path.insert(0, os.path.join(r"Y:\workspace\EDA\eda-pcb", "schematic_parser"))
from schematic_parser.data import discover, DEFAULT_DATA_ROOT        # noqa: E402


def ink(gray, ib, H, W):
    x0 = max(0, int(ib[0])); x1 = min(W, int(ib[2]))
    y0 = max(0, int(ib[1])); y1 = min(H, int(ib[3]))
    if x1 <= x0 or y1 <= y0:
        return -1.0
    return float((gray[y0:y1, x0:x1] < 200).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="")
    ap.add_argument("--case", default="")
    ap.add_argument("--by-type", action="store_true")
    ap.add_argument("--render", default="")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cases = list(discover(DEFAULT_DATA_ROOT))
    if a.splits:
        keep = set(x.strip() for x in a.splits.split(","))
        cases = [c for c in cases if c.split in keep]
    if a.case:
        cases = [c for c in cases if c.case_id == a.case]

    tot = 0
    agg = {"up": 0.0, "down": 0.0}
    zero_up = collections.Counter(); zero_down = collections.Counter()
    zero_by_case = collections.Counter()
    for c in cases:
        gt = json.load(open(c.json_path, encoding="utf-8"))
        gray = cv2.imdecode(np.fromfile(c.png_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        H, W = gray.shape[:2]
        for k, v in (gt.get("components") or {}).items():
            b = v.get("bbox") or v.get("bbox_json")
            if not b:
                continue
            b = [float(x) for x in b]
            up = (b[0], H - b[3], b[2], H - b[1])
            down = (b[0], b[1], b[2], b[3])
            iu = ink(gray, up, H, W); idn = ink(gray, down, H, W)
            tot += 1
            agg["up"] += max(0.0, iu); agg["down"] += max(0.0, idn)
            if iu < 0.002:
                zero_up[(v.get("type") or "?").lower()] += 1
                zero_by_case[c.case_id] += 1
            if idn < 0.002:
                zero_down[(v.get("type") or "?").lower()] += 1
    print("case 数 %d，GT 元件 %d" % (len(cases), tot))
    print("平均框内墨迹占比： y 向上(翻转) %.4f   y 向下(不翻转) %.4f"
          % (agg["up"] / max(1, tot), agg["down"] / max(1, tot)))
    print("零墨迹框数：     y 向上 %d (%.3f)   y 向下 %d (%.3f)"
          % (sum(zero_up.values()), sum(zero_up.values()) / max(1, tot),
             sum(zero_down.values()), sum(zero_down.values()) / max(1, tot)))
    if a.by_type:
        print("  零墨迹（y 向上）按类型：", zero_up.most_common(15))
        print("  零墨迹（y 向上）按 case：", zero_by_case.most_common(15))

    if a.render:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval"))
        import stage2_eval
        exact, digits = stage2_eval.build_index(DEFAULT_DATA_ROOT)
        c, how = stage2_eval.resolve(a.case or cases[0].case_id, exact, digits)
        gt = json.load(open(c.json_path, encoding="utf-8"))
        bgr = cv2.imdecode(np.fromfile(c.png_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        H, W = bgr.shape[:2]
        v = gt["components"][a.render]
        b = [float(x) for x in (v.get("bbox") or v.get("bbox_json"))]
        cx = (b[0] + b[2]) / 2.0; cy = (b[1] + b[3]) / 2.0
        T = 120
        for tag, iy in (("up", (H - b[3], H - b[1])), ("down", (b[1], b[3]))):
            y0 = int(max(0, H - cy - T)); y1 = int(min(H, H - cy + T))
            x0 = int(max(0, cx - 2 * T)); x1 = int(min(W, cx + 2 * T))
            im = bgr[y0:y1, x0:x1].copy()
            cv2.rectangle(im, (int(b[0] - x0), int(iy[0] - y0)),
                          (int(b[2] - x0), int(iy[1] - y0)), (0, 0, 255), 2)
            im = cv2.resize(im, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_NEAREST)
            head = np.full((26, im.shape[1], 3), 255, np.uint8)
            cv2.putText(head, "%s %s hypothesis" % (a.render, tag), (4, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 0, 0), 1, cv2.LINE_AA)
            cv2.imwrite(a.out or (r"Y:\workspace\EDA\work\preview\_axis_%s_%s_%s.png"
                                  % (c.case_id, a.render, tag)),
                        np.vstack([head, im]))
        print("rendered", a.render, "in", c.case_id)


if __name__ == "__main__":
    main()
