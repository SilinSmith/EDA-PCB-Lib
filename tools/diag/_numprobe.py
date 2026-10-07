# -*- coding: utf-8 -*-
"""引脚**号**读不出来的原因：是「字没检出」还是「检出没认领」？

用法：
    python work\\tools\\diag\\_numprobe.py --dir work\\_fast_out_off9b [--cases 1,2,3]

看的是 `<dir>\\runs\\stage1\\<case>.stage1.json`：
  * `components` 里 `type in IC_TYPES` 的框 ⇒ GT 里算 pin_f1 的那些；
  * `pins[cid]` 是 stage1 交出的引脚点（含 pincol 按格点补出来的）；
  * `texts[i] = {"bbox_img": [...], "text": ...}`（图像空间，y 向下）。

对每个引脚点，按 `pincol._num_candidates` **完全同一套判据**找「贴着框边的
小文字框」（`w<=NUM_MAX_W(42)`、`h<=NUM_MAX_H(26)`、中心在边外侧 `reach(26)` 内），
再对每个点算：
  * `cand`  贴边小文字框的个数（>0 表示图上那个号**被检出了**）；
  * `hit`   我们最终交出的引脚键里，这个点上有没有号；
  * `txt`   候选框的文字（如果 `texts` 里带 `text` 字段）。

**判决口径**：`cand>0 & hit=0` ⇒ 「检出了但没认领」（可修：认领/拟合逻辑）；
`cand=0 & hit=0` ⇒ 「字根本没检出」（要修 textbox 检出或放宽尺寸闸）。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EDA = ROOT / "eda-pcb"
sys.path.insert(0, str(EDA))
sys.path.insert(0, str(EDA / "stage1"))
sys.stdout.reconfigure(encoding="utf-8")

import pincol  # noqa: E402

IC_TYPES = ("box", "ic", "boxic", "bigic")


def canon_type(t):
    return str(t or "").strip().lower()


def side_of(p, box):
    """引脚点在框的哪条边外（离最近边）。"""
    x, y = float(p[0]), float(p[1])
    x0, y0, x1, y1 = [float(v) for v in box]
    d = {"L": abs(x - x0), "R": abs(x - x1), "T": abs(y - y0), "B": abs(y - y1)}
    return min(d, key=d.get), min(d.values())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="work/_fast_out")
    ap.add_argument("--cases", default="")
    ap.add_argument("--show", type=int, default=6)
    a = ap.parse_args(argv)

    d = Path(a.dir)
    if not d.is_absolute():
        d = ROOT / d
    s1 = d / "runs" / "stage1"
    want = [c.strip() for c in a.cases.split(",") if c.strip()]
    files = sorted(s1.glob("*.stage1.json"),
                   key=lambda p: (len(p.stem), p.stem))
    if want:
        files = [p for p in files if p.name.split(".")[0] in want]

    tst = Counter()
    for fp in files:
        case = fp.name.split(".")[0]
        doc = json.loads(fp.read_text(encoding="utf-8"))
        texts = doc.get("texts") or []
        comps = doc.get("components") or []
        pins = doc.get("pins") or {}
        im = doc.get("image") or {}
        H = float(im.get("h") or im.get("height") or 0) or None
        pred = d / "final_results" / (case + "_Pred.json")
        pdoc = json.loads(pred.read_text(encoding="utf-8")) if pred.exists() else {}
        ppins = pdoc.get("pins") or {}
        pcomps = pdoc.get("components") or {}

        for c in comps:
            ct = canon_type(c.get("type"))
            if ct not in IC_TYPES:
                continue
            cid = c.get("id")
            box = c.get("bbox_img")
            pts = pins.get(cid) or []
            if not box or not pts:
                continue
            desig = match_desig(c, pcomps)
            if H is None:
                H = max([float(v[3]) for v in
                         (x.get("bbox_img") or [0, 0, 0, 0]) for x in comps] or [0])
            ppts = []
            for k, v in (ppins.get(desig) or {}).items():
                try:
                    ppts.append((float(v["point"]["x"]), H - float(v["point"]["y"]),
                                 str(k).replace("pin_", "")))
                except (KeyError, TypeError, ValueError):
                    continue
            got = {n for _x, _y, n in ppts}
            n_cand = n_hit = n_both = n_none = 0
            rows = []
            for p in pts:
                side, _dist = side_of(p, box)
                cand = pincol._num_candidates(texts, box, side)
                near = []
                for along, i in cand:
                    ref = float(p[1]) if side in ("L", "R") else float(p[0])
                    if abs(along - ref) <= pincol.BAND:
                        near.append((along, i))
                # 这个点上「我们交出的号」= 离它最近的预测引脚点
                hit, hd = None, 1e9
                for x, y, n in ppts:
                    dd = abs(x - float(p[0])) + abs(y - float(p[1]))
                    if dd < hd:
                        hd, hit = dd, n
                if hd > 12.0:
                    hit = None
                if near and hit is not None:
                    n_both += 1
                elif near:
                    n_cand += 1
                elif hit is not None:
                    n_hit += 1
                else:
                    n_none += 1
                rows.append((p, side, len(near), text_of(texts, near), hit))
            tot = len(pts)
            tst["stub"] += tot
            tst["cand_no_hit"] += n_cand
            tst["hit_no_cand"] += n_hit
            tst["both"] += n_both
            tst["neither"] += n_none
            print("[%s %s %s→%s] 引脚点 %d ｜ 贴边有小字 %d ｜ 有小字且交出号 %d ｜ "
                  "**有小字但没交出号 %d** ｜ **交出号但没小字 %d** ｜ 都没有 %d"
                  % (case, cid, ct, desig, tot, n_cand + n_both, n_both, n_cand,
                     n_hit, n_none))
            if a.show and (n_cand or n_hit or n_none):
                print("      交出 %d 个号：%s" % (len(got), sorted(got)[:26]))
                for p, side, k, t, hit in rows[:a.show]:
                    print("      %s side=%s 贴边小字 %d %s 交出=%s"
                          % ([round(float(v), 1) for v in p[:2]], side, k, t, hit))

    print()
    print("=== 合计 ===")
    print("IC 引脚点 %d ｜ 有小字且交出号 %d ｜ **有小字却没交出号 %d** ｜ "
          "**交出号却没小字 %d** ｜ 两头都没有 %d"
          % (tst["stub"], tst["both"], tst["cand_no_hit"],
             tst["hit_no_cand"], tst["neither"]))
    print("⇒ 「有小字却没交出号」= 认得出来但没认领（可修）；"
          "「两头都没有」= 图上那个号根本没被检出（要修 textbox）")
    return 0


def text_of(texts, near):
    out = []
    for _along, i in near[:3]:
        t = texts[i] if 0 <= i < len(texts) else {}
        s = (t.get("text") or t.get("txt") or t.get("label") or "?")
        bb = t.get("bbox_img")
        sz = "%.0fx%.0f" % (float(bb[2]) - float(bb[0]),
                            float(bb[3]) - float(bb[1])) if bb else "?"
        out.append("%s(%s)" % (s, sz))
    return " ".join(out)


def iou(a, b):
    ax0, ay0, ax1, ay1 = [float(v) for v in a]
    bx0, by0, bx1, by1 = [float(v) for v in b]
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / ua if ua > 0 else 0.0


def match_desig(c, pcomps):
    """stage1 元件 → 预测元件键（按 JSON 空间 bbox 的 IoU 取最大）。"""
    bj = c.get("bbox_json") or c.get("bbox_img")
    best, bi = None, 0.0
    for k, v in (pcomps or {}).items():
        b = v.get("bbox")
        if not b:
            continue
        s = iou(bj, b)
        if s > bi:
            best, bi = k, s
    return best


if __name__ == "__main__":
    raise SystemExit(main())
