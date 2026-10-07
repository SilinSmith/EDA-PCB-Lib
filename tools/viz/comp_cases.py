# -*- coding: utf-8 -*-
"""第一阶段**元件级**错例的典型示例集（对应 `viz/wire_cases.py` 的导线级那一份）。

四类错例，各挑 N 个代表渲染成一张拼图：

    fn              GT 有、我们没检出来（漏检）
    fp              我们检出来了、GT 没有（误检）
    type_mismatch   框对上了，但 type 不同
    text_mismatch   框对上了、type 也对，但 Name / value 不同

每格画在原图窗口上：
    蓝框 = GT 元件框       绿框 = 我们检出的元件框       红框 = 本格的讨论主体
格底部一行小字写：类别 · case · GT/预测的 type 与文字。

用法（CWD 必须在 `eda-pcb`）：
    python Y:\\workspace\\EDA\\work\\tools\\viz\\comp_cases.py [stage1产出目录]
        [--per-cat 4] [--win 260] [--limit 220]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[3] / "eda-pcb"))

from schematic_parser.data import DEFAULT_DATA_ROOT, SPLITS, discover, load_json  # noqa: E402


def imread_unicode(path):
    """`cv2.imread` 读不了非 ASCII 路径（本项目目录名有中文），必须走 imdecode。"""
    buf = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)

OUT_DIR = Path(r"Y:\workspace\EDA\work\preview\compcases")
CATS = ("fn", "fp", "no_desig", "type_mismatch", "text_mismatch")
CAT_TITLE = {
    "fn": "漏检 FN      GT 有、我们没检出来",
    "fp": "误检 FP      我们检出来了、GT 没有",
    "no_desig": "没给位号      框在、type 也对，但我们一个位号都没挂上（Name 回退成内部 id）",
    "type_mismatch": "类型错       框对上了，type 不同",
    "text_mismatch": "文字错       框和 type 都对，位号也挂上了，但读错",
}
CELL_W, CELL_H = 340, 300
BAR = 34


def gt_index(data_root):
    idx = {}
    for c in discover(data_root, SPLITS):
        if not c.json_path:
            continue
        idx.setdefault(c.case_id, c)
        idx.setdefault(re.sub(r"\D", "", c.case_id) or c.case_id, c)
    return idx


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def gt_components(doc, H):
    out = []
    for key, v in (doc.get("components") or {}).items():
        b = v.get("bbox")
        if not b or len(b) != 4:
            continue
        box = (float(b[0]), H - float(b[3]), float(b[2]), H - float(b[1]))
        out.append({"key": str(key), "type": v.get("type"),
                    "name": v.get("Name"), "value": v.get("value"),
                    "box": box, "hit": None})
    return out


def pred_components(doc, H):
    """stage1 的 `components` 是 list（每项 `bbox_img`，图像坐标）；stage2/stage3
    是 dict（键=位号，每项 `bbox`，**题目的 y 向上坐标**）。两种都要吃。"""
    comps = doc.get("components") or []
    items = comps.items() if isinstance(comps, dict) else \
        [(c.get("id"), c) for c in comps]
    out = []
    for cid, c in items:
        b = c.get("bbox_img")
        if b and len(b) == 4:
            box = tuple(float(v) for v in b)
        else:
            b = c.get("bbox")
            if not b or len(b) != 4:
                continue
            box = (float(b[0]), H - float(b[3]), float(b[2]), H - float(b[1]))
        out.append({"id": c.get("id") or cid, "type": c.get("type"),
                    "name": c.get("Name"), "value": c.get("value"),
                    "box": box, "hit": None})
    return out


def text_of(d):
    if d is None:
        return None
    return d.get("name") if d.get("name") not in (None, "") else d.get("value")


_FALLBACK_KEY = re.compile(r"^p\d{3,4}$")


def _is_fallback_key(p):
    """`assemble.py:137` 的兜底 key：位号缺失或重名时写成 `"p%03d" % i`（小写 p）。
    真位号是大写字母开头，所以小写 `pNNN` 是安全的兜底标记。"""
    return bool(_FALLBACK_KEY.match(str(p.get("id") or "")))


def pair(gt, pr, tol=0.1):
    """贪心一对一配对：按 IoU 从高到低吃。返回 (配对表, 未配对的 gt, 未配对的 pr)。"""
    pairs = []
    cand = []
    for i, g in enumerate(gt):
        for j, p in enumerate(pr):
            v = iou(g["box"], p["box"])
            if v >= tol:
                cand.append((v, i, j))
    cand.sort(reverse=True)
    ug, up = set(range(len(gt))), set(range(len(pr)))
    for v, i, j in cand:
        if i in ug and j in up:
            pairs.append((i, j, v))
            ug.discard(i)
            up.discard(j)
    return pairs, [gt[i] for i in sorted(ug)], [pr[j] for j in sorted(up)]


def draw(img, window, gt, pr, subject):
    x1, y1, x2, y2 = [int(v) for v in window]
    x1, y1 = max(0, x1), max(0, y1)
    crop = img[y1:y2, x1:x2].copy()
    if crop.size == 0:
        return None
    for d, col in ((gt, (255, 140, 0)), (pr, (0, 200, 0))):
        for e in d:
            b = e["box"]
            cv2.rectangle(crop, (int(b[0]) - x1, int(b[1]) - y1),
                          (int(b[2]) - x1, int(b[3]) - y1), col, 1)
    if subject is not None:
        b = subject["box"]
        cv2.rectangle(crop, (int(b[0]) - x1, int(b[1]) - y1),
                      (int(b[2]) - x1, int(b[3]) - y1), (0, 0, 255), 2)
    return crop


def sheet(cards, path, title, legend):
    if not cards:
        return
    cols = min(4, len(cards))
    rows = (len(cards) + cols - 1) // cols
    head = 26
    canvas = np.full((head + BAR + rows * CELL_H, cols * CELL_W, 3), 255, np.uint8)
    cv2.putText(canvas, title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, legend, (8, head + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (90, 90, 90), 1, cv2.LINE_AA)
    for k, (crop, cap) in enumerate(cards):
        r, c = divmod(k, cols)
        y0 = head + BAR + r * CELL_H
        if crop is None:
            continue
        h, w = crop.shape[:2]
        s = min((CELL_W - 6) / w, (CELL_H - 26) / h)
        small = cv2.resize(crop, (max(1, int(w * s)), max(1, int(h * s))),
                           interpolation=cv2.INTER_NEAREST)
        canvas[y0 + 2:y0 + 2 + small.shape[0], c * CELL_W + 3:c * CELL_W + 3 + small.shape[1]] = small
        cv2.putText(canvas, cap, (c * CELL_W + 4, y0 + CELL_H - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 0, 0), 1, cv2.LINE_AA)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas)
    print(f"  -> {path}  ({canvas.shape[1]}x{canvas.shape[0]}, {len(cards)} 格)")


def main(argv=None):
    ap = argparse.ArgumentParser(description="元件级错例示例集")
    ap.add_argument("stage1", nargs="?", default=r"Y:\workspace\EDA\work\stage1_all14")
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--per-cat", type=int, default=4)
    ap.add_argument("--win", type=int, default=260)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    idx = gt_index(args.data_root)
    files = sorted(Path(args.stage1).glob("*.stage1.json"))
    if not files:                      # stage2/stage3 的产物是 <case>.json
        files = [f for f in sorted(Path(args.stage1).glob("*.json"))
                 if not f.name.startswith("_")]
    if args.limit:
        files = files[:args.limit]
    buckets = {c: [] for c in CATS}
    tally = {c: 0 for c in CATS}
    dbg = {"配对": 0, "GT元件": 0, "预测元件": 0}
    ncase = 0
    for f in files:
        case = f.name.replace(".stage1.json", "")
        rec = idx.get(case) or idx.get(re.sub(r"\D", "", case) or case)
        if rec is None:
            continue
        pdoc = json.loads(f.read_text(encoding="utf-8"))
        case = pdoc.get("case") or case
        img = imread_unicode(rec.png_path)
        H = img.shape[0]
        gdoc = load_json(rec.json_path)
        gt = gt_components(gdoc, H)
        pr = pred_components(pdoc, H)
        pairs, ug, up = pair(gt, pr)
        ncase += 1
        dbg["配对"] += len(pairs)
        dbg["GT元件"] += len(gt)
        dbg["预测元件"] += len(pr)
        for i, j, _v in pairs:
            g, p = gt[i], pr[j]
            if str(g["type"]).lower() != str(p["type"]).lower():
                tally["type_mismatch"] += 1
                buckets["type_mismatch"].append(
                    (case, img, g, p, g, f"${case} type GT={g['type']} vs 预测={p['type']}"))
            elif text_of(g) is not None or text_of(p) is not None:
                # 注意：**位号读成功时 `assemble.py` 会拿位号当 key**，于是
                # `Name == key` 对「C5」和「p004」都成立（实测 1574 条是好例）。
                # 真正的「没给位号」标记只有那个兜底 key `p%03d`（`assemble.py:137`）。
                # `Name == 键` 且键形如 `pNNN` 才是兜底。
                if _is_fallback_key(p) and text_of(g) not in (None, ""):
                    tally["no_desig"] += 1
                    buckets["no_desig"].append(
                        (case, img, g, p, g,
                         f"${case} {g['type']} GT {g['key']}={text_of(g)!r} 我们没给位号"))
                elif text_of(g) != text_of(p):
                    tally["text_mismatch"] += 1
                    buckets["text_mismatch"].append(
                        (case, img, g, p, g,
                         f"${case} {g['type']} 文字 GT={text_of(g)!r} vs 预测={text_of(p)!r}"))
        for g in ug:
            tally["fn"] += 1
            buckets["fn"].append((case, img, g, None, g,
                                  f"${case} GT {g['type']} key={g['key']} 没检出"))
        for p in up:
            tally["fp"] += 1
            buckets["fp"].append((case, img, None, p, p,
                                  f"${case} 多检出 {p['type']} id={p['id']}"))

    print(f"\n扫了 {ncase} 张图。错例条数：")
    print(f"  [dbg] {dbg}")
    for c in CATS:
        print(f"  {c:16s} {tally[c]:6d}")
    print()
    legend = ("蓝框=GT 元件   绿框=我们检出的元件   红框=本格讨论主体   "
              "（FN 格只有红+蓝，FP 格只有红+绿）")
    for c in CATS:
        print(f"  {c:14s} {tally[c]}")
    for c in CATS:
        cards = []
        for case, img, g, p, subj, cap in buckets[c][:args.per_cat]:
            b = subj["box"]
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            w = args.win
            win = (cx - w / 2, cy - w / 2, cx + w / 2, cy + w / 2)
            crop = draw(img, win, [g] if g else [], [p] if p else [], subj)
            if crop is not None:
                cards.append((crop, cap))
        sheet(cards, OUT_DIR / f"{c}.png", CAT_TITLE[c], legend)
    print(f"\n产物目录 {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
