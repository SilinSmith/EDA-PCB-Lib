"""看 GT 里 r / l 到底长什么样：在原始分辨率上裁 GT 框，按类型拼接触表。

用法:
    python work\tools\diag\_rlook.py --case 0050 --types r,l --n 24
    python work\tools\diag\_rlook.py --stats          # 全库类型 × 来源 计数
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
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval"))
from schematic_parser.data import discover, DEFAULT_DATA_ROOT        # noqa: E402
import stage2_eval                                                   # noqa: E402


def load_case(stem):
    exact, digits = stage2_eval.build_index(DEFAULT_DATA_ROOT)
    c, how = stage2_eval.resolve(stem, exact, digits)
    if c is None:
        raise SystemExit("找不到 case " + stem)
    with open(c.json_path, encoding="utf-8") as f:
        return c, json.load(f)


def read_img(path):
    buf = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def stats():
    """全库：type -> 出现次数，按 split 分开。"""
    per = collections.defaultdict(collections.Counter)
    tot = collections.Counter()
    for c in discover(DEFAULT_DATA_ROOT):
        with open(c.json_path, encoding="utf-8") as f:
            gt = json.load(f)
        for k, v in (gt.get("components") or {}).items():
            t = (v.get("type") or "?").lower()
            per[c.split][t] += 1
            tot[t] += 1
    print("总元件 %d" % sum(tot.values()))
    print("%-14s %8s   %s" % ("type", "总", "按 split"))
    for t, n in tot.most_common():
        s = " ".join("%s=%d" % (k, per[k][t]) for k in sorted(per) if per[k][t])
        print("%-14s %8d   %s" % (t, n, s))


def look(stem, types, n, out):
    c, gt = load_case(stem)
    bgr = read_img(c.png_path)
    H = bgr.shape[0]
    cells = []
    for k, v in (gt.get("components") or {}).items():
        t = (v.get("type") or "?").lower()
        if t not in types:
            continue
        b = v.get("bbox") or v.get("bbox_json")
        if not b:
            continue
        bb = [float(x) for x in b]
        ib = (int(bb[0]), int(H - bb[3]), int(bb[2]), int(H - bb[1]))
        x0 = max(0, ib[0] - 12); y0 = max(0, ib[1] - 12)
        x1 = min(bgr.shape[1], ib[2] + 12); y1 = min(bgr.shape[0], ib[3] + 12)
        im = bgr[y0:y1, x0:x1]
        if im.size == 0:
            continue
        s = max(1.0, 96.0 / max(1, im.shape[0]))
        im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
        cells.append((k, t, bb, im))
        if len(cells) >= n:
            break
    if not cells:
        raise SystemExit("没有 %s 类型的元件" % ",".join(types))
    print("%s: %s  图像 %s" % (c.case_id, c.split, bgr.shape))
    h = 140
    tiles = []
    for k, t, bb, im in cells:
        sc = h / im.shape[0]
        im = cv2.resize(im, (max(1, int(im.shape[1] * sc)), h), interpolation=cv2.INTER_NEAREST)
        pad = np.full((h + 26, im.shape[1] + 8, 3), 255, np.uint8)
        pad[26:26 + h, 4:4 + im.shape[1]] = im
        cv2.putText(pad, "%s %s %.0fx%.0f" % (k, t, bb[2] - bb[0], bb[3] - bb[1]),
                    (2, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 200), 1, cv2.LINE_AA)
        tiles.append(pad)
        print("  %-8s type=%-3s bbox=%s  %.0fx%.0f" % (k, t, [round(x) for x in bb],
                                                       bb[2] - bb[0], bb[3] - bb[1]))
    cols = min(6, len(tiles))
    rows = (len(tiles) + cols - 1) // cols
    tw = max(t.shape[1] for t in tiles)
    th = tiles[0].shape[0]
    out_im = np.full((rows * th, cols * tw, 3), 245, np.uint8)
    for i, t in enumerate(tiles):
        r, cc = divmod(i, cols)
        out_im[r * th:r * th + t.shape[0], cc * tw:cc * tw + t.shape[1]] = t
    cv2.imwrite(out, out_im)
    print("saved", out, out_im.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--case", default="0050")
    ap.add_argument("--types", default="r,l")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.stats:
        stats()
        return
    out = a.out or (r"Y:\workspace\EDA\work\preview\_rlook_%s.png" % a.case)
    look(a.case, set(x.strip() for x in a.types.split(",")), a.n, out)


if __name__ == "__main__":
    main()
