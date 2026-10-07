# -*- coding: utf-8 -*-
"""兜底键元件「名下明明有文字框，却没读出位号」——把那些框的真实 CRNN 读法打出来。

由来：`_pwhy.py` 量出 2601 个兜底键元件里 **92.2% 在 10px 内就有一个 stage1 文字框**，
但 `_assigned.comps[cid].designator` 是空的。要么是 CRNN 没读出位号，
要么是读出来了但 `canon_designator` 掐掉了（合并框 / 前缀不符）。

    python work\\tools\\diag\\_pnull.py --pred work\\_crnntrunc\\final_results --limit 60
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(r"Y:\workspace\EDA")
sys.path.insert(0, str(ROOT / "eda-pcb"))
sys.path.insert(0, str(ROOT / "work" / "tools" / "eval"))

import schematic_parser.metrics as M                      # noqa: E402
import stage2_eval as SE                                  # noqa: E402
from stage2.text import TextEngine, imread_unicode        # noqa: E402
from stage2.assign import canon_designator                # noqa: E402


def rdist(a, b):
    dx = max(b[0] - a[2], a[0] - b[2], 0.0)
    dy = max(b[1] - a[3], a[1] - b[3], 0.0)
    return math.hypot(dx, dy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--limit", type=int, default=60, help="看多少张")
    ap.add_argument("--near", type=float, default=12.0)
    ap.add_argument("--show", type=int, default=30)
    a = ap.parse_args()

    root = Path(a.pred)
    s2 = root.parent / "runs" / "stage2"
    engine = TextEngine()
    ex, di = SE.build_index(SE.DEFAULT_DATA_ROOT)

    bucket = Counter()
    samples = []
    n = 0
    for f in sorted(root.glob("*_Pred.json"))[: a.limit]:
        try:
            c, _ = SE.resolve(f.stem.replace("_Pred", ""), ex, di)
        except Exception:
            continue
        cid = f.stem.replace("_Pred", "")
        s1f = s2.parent / "stage1" / (cid + ".stage1.json")
        if not s1f.is_file():
            continue
        p = json.loads(f.read_text(encoding="utf-8"))
        g = json.loads(Path(c.json_path).read_text(encoding="utf-8"))
        gc, pc = g.get("components") or {}, p.get("components") or {}
        if not gc or not pc:
            continue
        _f1, km = M.assign_components(g, p, M.EvalConfig(), {})
        up = [k for k in pc if k not in set(km)]
        pnnn = [k for k in up if k.startswith("p") and k[1:].isdigit()]
        if not pnnn:
            continue
        s1 = json.loads(s1f.read_text(encoding="utf-8"))
        texts = s1.get("texts") or []
        comps = s1.get("components") or []
        if not texts:
            continue
        # 挑出「10px 内有文字框」的那些元件，连同最近的框
        want, boxes = [], []
        for pk in pnnn:
            i = int(pk[1:])
            if i >= len(comps):
                continue
            cb = comps[i].get("bbox_img")
            if not cb:
                continue
            ds = [(rdist(cb, t["bbox_img"]), t) for t in texts]
            d, t = min(ds, key=lambda x: x[0])
            if d <= a.near:
                want.append((pk, comps[i].get("type"), d, t))
                boxes.append([float(v) for v in t["bbox_img"]])
        if not want:
            continue
        bgr = imread_unicode(c.png_path)
        H = float(bgr.shape[0])
        # 注意：TextEngine.read 要图像坐标（y 向下）；stage1 的 bbox_img 本来就是
        ib = [b if ii == 1 or ii == 3 else b for ii, b in enumerate(boxes)]  # 原样
        reads = engine.read(bgr, [tuple(b) for b in ib], ["designator"])
        for (pk, ctype, d, t), rr in zip(want, reads):
            n += 1
            txt = (rr.get("reads") or {}).get("designator", ("", 0.0))[0]
            can, fid = canon_designator(txt, ctype or "other")
            if not txt:
                b = "A_CRNN 什么都没读出来"
            elif can:
                b = "B_读出来且能 canon（说明是归属/竞争问题）"
            else:
                b = "C_读出来了但 canon 掐掉"
            bucket[b] += 1
            if len(samples) < a.show:
                samples.append((cid, pk, ctype, round(d, 1), t["bbox_img"], txt, can))

    print("看了 %d 张，落在 10~12px 内的兜底键元件 %d 个" % (a.limit, n))
    for k, v in bucket.most_common():
        print("  %-40s %6d  %5.1f%%" % (k, v, 100.0 * v / max(1, n)))
    print()
    for cid, pk, ctype, d, bb, txt, can in samples:
        print("  %-18s %-6s %-8s d=%-5s box=%-26s read=%-16r canon=%r"
              % (cid, pk, ctype, d, [round(x) for x in bb], txt, can))


if __name__ == "__main__":
    main()
