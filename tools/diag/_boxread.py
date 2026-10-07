# -*- coding: utf-8 -*-
"""GT box(IC) 元件的位号到底有没有印在图上、印在环带哪个位置。

做法：不看我们的归属判定，直接从 stage1 的文字框出发，取每颗 GT box
元件外圈「环带」（bbox 外扩 `--pad`，再减掉 bbox 本身）里的所有文字框，
按邻近关系并成几组，逐组 OCR，看能不能读出形如 `X1` / `X?` 的位号。

用法：
    python work\tools\diag\_boxread.py --limit 0 --pad 70
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "eda-pcb"))

from PIL import Image  # noqa: E402
from stage2.text import TextEngine, imread_unicode, pad_box  # noqa: E402

ROOT = REPO / "eda-pcb" / "yolo_part" / "Dataset(waitforpreprocess)"
S1DIR = REPO / "work" / "stage1_v6"
DES_RE = re.compile(r"^[A-Za-z]{1,3}\d{1,3}$")


def find_png(case):
    for base in ("200_train_cases", "10GT"):
        for d in glob.glob(str(ROOT / base / (case + "*"))):
            p = Path(d)
            if p.is_dir() and p.name.split("-")[0] == case:
                g = next(p.glob("*.png"), None)
                if g:
                    return g
    return None


def find_gt(case):
    for base in ("200_train_cases", "10GT"):
        for d in glob.glob(str(ROOT / base / (case + "*"))):
            p = Path(d)
            if p.is_dir() and p.name.split("-")[0] == case:
                g = next((x for x in p.glob("*.json")), None)
                if g:
                    return g
    return None


def group(boxes, gap):
    """把文字框按「外接矩形间隙 <= gap」并成连通块（简单 O(n^2)）。"""
    parent = list(range(len(boxes)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def near(a, b):
        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = b
        dx = max(0, max(ax0, bx0) - min(ax1, bx1))
        dy = max(0, max(ay0, by0) - min(ay1, by1))
        return dx <= gap and dy <= gap

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if near(boxes[i], boxes[j]):
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
    gs = {}
    for i in range(len(boxes)):
        gs.setdefault(find(i), []).append(boxes[i])
    out = []
    for v in gs.values():
        out.append([min(b[0] for b in v), min(b[1] for b in v),
                    max(b[2] for b in v), max(b[3] for b in v)])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pad", type=float, default=70.0)
    ap.add_argument("--gap", type=float, default=14.0)
    ap.add_argument("--samples", type=int, default=30)
    a = ap.parse_args()

    eng = TextEngine()
    files = sorted(S1DIR.glob("*.stage1.json"))
    if a.limit:
        files = files[:a.limit]
    stat = Counter()
    samples = []
    for f in files:
        case = f.name.split(".")[0]
        gtj = find_gt(case)
        s1 = json.loads(f.read_text(encoding="utf-8"))
        H = float(s1["image"]["h"])
        if not gtj:
            continue
        gt = json.loads(Path(gtj).read_text(encoding="utf-8"))
        boxes = [(k, v) for k, v in (gt.get("components") or {}).items()
                 if (v.get("type") or "") == "box"]
        if not boxes:
            continue
        texts = s1.get("texts") or []
        for k, v in boxes:
            b = v["bbox"]
            iy0, iy1 = H - b[3], H - b[1]
            ring = (b[0] - a.pad, iy0 - a.pad, b[2] + a.pad, iy1 + a.pad)
            stat["gt_box"] += 1
            cand = []
            for t in texts:
                tb = t.get("bbox_img")
                if not tb:
                    continue
                cx, cy = (tb[0] + tb[2]) / 2.0, (tb[1] + tb[3]) / 2.0
                if not (ring[0] <= cx <= ring[2] and ring[1] <= cy <= ring[3]):
                    continue
                # 环带 = 外扩框 减 bbox 本身
                if b[0] <= cx <= b[2] and iy0 <= cy <= iy1:
                    continue
                cand.append(tb)
            if not cand:
                stat["no_text"] += 1
                continue
            gs = group(cand, a.gap)
            stat["ring_groups"] += len(gs)
            try:
                im = imread_unicode(find_png(case))
            except Exception:
                stat["no_png"] += 1
                continue
            res = eng.read(im, [pad_box(g) for g in gs], ["designator"])
            got = set()
            for rr in res:
                t = (rr["reads"].get("designator") or ("", 0.0))[0]
                if t:
                    got.add(t)
            if got:
                stat["any_read"] += 1
                if k in got:
                    stat["exact_hit"] += 1
                else:
                    if len(samples) < a.samples:
                        samples.append((case, k, sorted(got)[:4]))
                stat["mismatch"] += 1
            else:
                stat["read_fail"] += 1
    print("=== GT box 位号「环带直读」实验（pad=%.0f gap=%.0f）===" % (a.pad, a.gap))
    print(json.dumps(stat, ensure_ascii=False, indent=2))
    n = max(1, stat["gt_box"])
    print("环带里有字 %.3f | 读出至少一个候选 %.3f | 其中正好等于 GT 键 %.3f | 读错 %.3f"
          % (1 - stat["no_text"] / n, stat["any_read"] / n,
             stat["exact_hit"] / n, stat["mismatch"] / n))
    print("\n读错/读不出的样例（case, GT键, 读到的候选）：")
    for r in samples:
        print("   ", r)


if __name__ == "__main__":
    main()
