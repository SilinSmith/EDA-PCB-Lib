# -*- coding: utf-8 -*-
"""量「GT 的 IC 元件，位号在图上到底读不读得出来」。

这决定了 pin 分的真实天花板：`metrics.assign_components` 在
`unkeyed_match='gt'`（官方默认）下，只允许 **GT 键本身是 unkeyed（`∅NNN`）**
的元件走坐标匹配；GT 键 `U1` 只能靠「我们输出正好叫 `U1`」来配上。
所以读不出位号的 IC = 它全部引脚直接算 missing。

用法：
    python work\\tools\\diag\\_icread.py                 # 快速回归集 19 张
    python work\\tools\\diag\\_icread.py --cases 0092
    python work\\tools\\diag\\_icread.py --pad 60 --limit 40
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]           # Y:\workspace\EDA
EDA = ROOT / "eda-pcb"
sys.path.insert(0, str(EDA))
sys.path.insert(0, str(ROOT / "work" / "tools" / "eval"))

import cv2                                                     # noqa: E402
from schematic_parser.data import DEFAULT_DATA_ROOT, load_json  # noqa: E402
from schematic_parser.metrics import (IC_TYPES, EvalConfig, assign_components,
                                      canon_type, normalise_doc)  # noqa: E402
from stage2_eval import build_index, resolve as _resolve        # noqa: E402
from stage2.text import TextEngine, imread_unicode              # noqa: E402

FAST = ["1", "2", "3", "4", "5", "7", "8", "9", "10",
        "0092", "0048", "0019", "0024", "0075", "0086",
        "0002", "0001", "0042", "0103"]


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    it = (ix2 - ix1) * (iy2 - iy1)
    return it / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - it + 1e-9)


def _gap_dist(a, b):
    dx = max(a[0] - b[2], 0, b[0] - a[2])
    dy = max(a[1] - b[3], 0, b[1] - a[3])
    return (dx * dx + dy * dy) ** 0.5


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="")
    ap.add_argument("--pred", default="work/_fast_out/final_results")
    ap.add_argument("--stage1", default="work/_fast_out/runs/stage1")
    ap.add_argument("--pad", type=float, default=60.0, help="只看离 IC 框这么近的文字框")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--show", type=int, default=14)
    a = ap.parse_args(argv)

    exact, digits = build_index(DEFAULT_DATA_ROOT)
    ids = [s.strip() for s in a.cases.split(",") if s.strip()] or FAST
    engine = TextEngine(device="auto")

    tot = hit_key = read_ok = read_other = read_none = 0
    rows = []
    for cid in ids:
        c, _ = _resolve(cid, exact, digits)
        if c is None:
            c, _ = _resolve(cid.lstrip("0"), exact, digits)
        if c is None or not c.json_path:
            continue
        cid = c.case_id
        pf = Path(a.pred) / ("%s_Pred.json" % cid)
        sf = Path(a.stage1) / ("%s.stage1.json" % cid)
        if not pf.exists() or not sf.exists():
            continue
        gt = load_json(Path(c.json_path))
        pr = load_json(pf)
        g, p = normalise_doc(gt), normalise_doc(pr)
        f1, km = assign_components(g, p, EvalConfig(), {})
        matched = set(km.values())
        st = json.loads(sf.read_text(encoding="utf-8"))
        boxes = [(t["bbox_img"][0], t["bbox_img"][1], t["bbox_img"][2], t["bbox_img"][3])
                 for t in (st.get("texts") or []) if t.get("bbox_img")]
        bgr = imread_unicode(c.png_path)
        H = bgr.shape[0]

        todo = []
        for k, v in (gt.get("components") or {}).items():
            if canon_type(v.get("type")) not in IC_TYPES or not v.get("bbox"):
                continue
            if k in matched:
                continue
            b = v["bbox"]
            img = (b[0], H - b[3], b[2], H - b[1])
            near = [bb for bb in boxes if _gap_dist(img, bb) <= a.pad]
            tot += 1
            todo.append((k, img, near, v.get("Name")))

        # 一次把这张图所有需要的框都过 CRNN
        allb = []
        for _k, _im, near, _n in todo:
            allb.extend(near)
        uniq = sorted(set(allb))
        reads = engine.read(bgr, uniq, ["designator"]) if uniq else []
        rmap = {tuple(r["bbox"]): (r.get("reads") or {}).get("designator") for r in reads}

        for k, img, near, name in todo:
            got = []
            for bb in near:
                r = rmap.get(tuple(bb))
                if r and r[0]:
                    got.append(str(r[0]))
            up = k.upper()
            if any(s.strip().upper() == up for s in got):
                hit_key += 1
                verdict = "A_读得出这个键"
            elif got:
                read_other += 1
                verdict = "B_读了别的"
            else:
                read_none += 1
                verdict = "D_附近啥也没读出来"
            rows.append((cid, k, name, verdict, got[:4]))

    print("=" * 78)
    print("GT IC 元件（键没配上的）%d 个" % tot)
    print("  A_附近读得出这个键   %4d  %.3f" % (hit_key, hit_key / max(1, tot)))
    print("  B_读了别的位号       %4d  %.3f" % (read_other, read_other / max(1, tot)))
    print("  D_附近啥也没读出来   %4d  %.3f" % (read_none, read_none / max(1, tot)))
    print("=" * 78)
    from collections import Counter
    print("按判决：", dict(Counter(r[3] for r in rows)))
    print("样例：")
    for r in rows[:a.show]:
        print("   %-8s GT=%-6s Name=%-16s  %-20s %s"
              % (r[0], r[1], str(r[2])[:16], r[3], r[4]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
