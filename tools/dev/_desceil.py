# -*- coding: utf-8 -*-
"""**位号重解的天花板探针**：对每个「没挂上位号 / 挂了错位号」的 GT 元件，看它附近
   **还没被任何元件占用的** stage1 文字框里，有没有一个用 CRNN 能读出正确位号的 —— 有 = 可救。

动机：`_desacc.py` 显示派生档 component 的主要失分是位号挂载（伪键 47.6% / 挂错 20.9%），
而 `stage2/assign.py` 的「同一个框只能给一个人」全局去重会把框判给符号或邻居。
这个探针量的是「把这些框重新分配，最多能捞回多少」，不实现分配本身。

坐标系：GT 的 bbox 是 **JSON 约定（y 从图底往上）**；stage1 `texts[].bbox_img` 与
`_assigned` 槽位里的 `box` 都是**图像坐标（左上原点）**。翻转用 `y_img = H - y_json`。

用法：
    python -X utf8 work\tools\dev\_desceil.py --pred work\\_ro220b --limit 60
    python -X utf8 work\tools\dev\_desceil.py --pred work\\_ro220b --limit 60 --ckpt work\\ocr_official_v1\\best.pt
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

WORK = Path(r"Y:\workspace\EDA\work")
DR = Path(r"Y:\workspace\EDA\eda-pcb\yolo_part\Dataset(waitforpreprocess)")
SPLITS = ("200_train_cases", "10_GTcase", "10GT")
PSEUDO = re.compile(r"^(?:∅\d+|p\d{3,4})$")
DESIG = re.compile(r"^[A-Za-z]{1,4}\d{1,4}[A-Za-z]?$")
sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")


def iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def find_gt(case: str):
    for sp in SPLITS:
        d = DR / sp / case
        if not d.is_dir():
            continue
        js = sorted(d.glob("%s*.json" % case))
        js = [p for p in js if "target_named" in p.name] or js
        for p in js:
            try:
                g = json.loads(p.read_text(encoding="utf-8"))
            except Exception:                                        # noqa: BLE001
                continue
            if g.get("components"):
                return p, g
    return None, None


def norm(t: str) -> tuple[str, str]:
    a = (t or "").strip().upper()
    b = re.sub(r"[^0-9A-Za-z]", "", a)
    return a, b


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=str(WORK / "_ro220b"), help="最终产物目录（平铺 <case>.json）")
    ap.add_argument("--assigned", default=None, help="默认 <pred>\\_assigned")
    ap.add_argument("--stage1", default=str(WORK / "_full220" / "runs" / "stage1"))
    ap.add_argument("--cases-file", default=str(WORK / "_cases220.txt"))
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--min-iou", type=float, default=0.3)
    ap.add_argument("--radius", type=float, default=40.0, help="元件框外扩多少像素内找候选框")
    ap.add_argument("--max-box-h", type=float, default=24.0)
    ap.add_argument("--max-box-w", type=float, default=90.0)
    ap.add_argument("--include-occupied", action="store_true",
                    help="候选框不排除「已被别的元件当位号用」的（量全局重分配的头上空间）")
    ap.add_argument("--ckpt", default=str(WORK / "ocr_official_v1" / "best.pt"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-print", type=int, default=12)
    a = ap.parse_args()

    raw = Path(a.cases_file).read_text(encoding="utf-8")
    cases, seen = [], set()
    for tok in re.split(r"[,\s]+", raw):
        if not re.match(r"^[0-9A-Za-z_]+$", tok or "") or tok in seen:
            continue
        seen.add(tok)
        cases.append(tok)
    cases = cases[: a.limit]

    from PIL import Image
    from textrec.infer import Recognizer
    rec = Recognizer.load(a.ckpt, device=a.device)
    print("[ckpt] %s  cases=%d  radius=%.0f" % (a.ckpt, len(cases), a.radius))

    st = collections.Counter()
    examples = []
    for c in cases:
        p_gt, gt = find_gt(c)
        fp = Path(a.pred) / ("%s.json" % c)
        s1p = Path(a.stage1) / ("%s.stage1.json" % c)
        adp = Path(a.assigned or (Path(a.pred) / "_assigned")) / ("%s.json" % c)
        if p_gt is None or not fp.is_file() or not s1p.is_file():
            st["缺文件"] += 1
            continue
        pr = json.loads(fp.read_text(encoding="utf-8")).get("components") or {}
        s1 = json.loads(s1p.read_text(encoding="utf-8"))
        H = float((s1.get("image") or {}).get("h") or 0) or None
        if H is None:
            st["缺图高"] += 1
            continue
        texts = s1.get("texts") or []
        ad = json.loads(adp.read_text(encoding="utf-8")) if adp.is_file() else {}
        used_i = set()
        for _cid, cv in (ad.get("comps") or {}).items():
            ds = cv.get("designator")
            if isinstance(ds, dict) and ds.get("box_i") is not None:
                used_i.add(int(ds["box_i"]))

        # GT ↔ pred 配对（JSON 坐标）
        used = set()
        jobs = []                                     # (gk, bucket, comp_img_box)
        for gk, gv in (gt.get("components") or {}).items():
            gb = gv.get("bbox")
            if not gb or PSEUDO.match(str(gk)):
                continue
            best, bi = 0.0, None
            for pk, pv in pr.items():
                if pk in used:
                    continue
                v = iou(gb, pv.get("bbox") or [0, 0, 0, 0])
                if v > best:
                    best, bi = v, pk
            if bi is not None and best >= a.min_iou:
                used.add(bi)
            if bi is not None and best >= a.min_iou and gk.upper() == str(bi).upper():
                continue
            bucket = "伪键" if (bi is None or best < a.min_iou or PSEUDO.match(str(bi))) else "挂错"
            cb_img = [gb[0], H - gb[3], gb[2], H - gb[1]]
            jobs.append((gk, bucket, cb_img, None if bi is None else bi))

        # 候选框：中心落在元件框外扩 radius 内、未被用作任何元件位号、尺寸像文字
        items, owner = [], []
        for gk, bucket, cb_img, ours in jobs:
            ex = [cb_img[0] - a.radius, cb_img[1] - a.radius, cb_img[2] + a.radius, cb_img[3] + a.radius]
            cand = []
            for t in texts:
                ti = int(str(t.get("id") or "t-1").lstrip("t") or -1)
                bb = t.get("bbox_img") or [0, 0, 0, 0]
                w, h = bb[2] - bb[0], bb[3] - bb[1]
                if h > a.max_box_h or w > a.max_box_w or w <= 1 or h <= 1:
                    continue
                cx, cy = (bb[0] + bb[2]) / 2.0, (bb[1] + bb[3]) / 2.0
                if not (ex[0] <= cx <= ex[2] and ex[1] <= cy <= ex[3]):
                    continue
                if ti in used_i and not a.include_occupied:
                    st["候选全是占用框"] += 1
                    continue
                cand.append((ti, bb))
            for ti, bb in cand:
                items.append((bb, "designator"))
                owner.append((gk, bucket, ours, ti))
            if not cand:
                st["%s|附近没有空闲框" % bucket] += 1

        got = rec.read_many(Image.open((s1.get("image") or {}).get("path")).convert("L"), items) if items else []
        best: dict = {}
        for (gk, bucket, ours, ti), (txt, _cf) in zip(owner, got):
            a_, b_ = norm(txt)
            key = (gk, bucket)
            cur = best.get(key)
            hit = int(a_ == gk.upper() or b_ == gk.upper())
            shape = int(bool(DESIG.match(a_)))
            score = (hit, shape)
            if cur is None or score > cur[0]:
                best[key] = (score, txt, ours)
        for gk, bucket, cb_img, ours in jobs:
            v = best.get((gk, bucket))
            if v is None:
                continue                              # 已在「附近没有空闲框」里计过
            (hit, shape), txt, _ = v
            if hit:
                st["%s|可救（空闲框读出正确位号）" % bucket] += 1
                if len(examples) < a.max_print:
                    examples.append((c, gk, ours, "可救", txt))
            elif shape:
                st["%s|框里有别的位号" % bucket] += 1
            else:
                st["%s|框里不是位号" % bucket] += 1

    print("== 位号重解天花板（%s，%d 张）==" % (a.pred, len(cases)))
    for k in sorted(st):
        print("  %-34s %5d" % (k, st[k]))
    for e in examples:
        print("   例: case=%s GT=%-7s ours=%-7s %s -> 读作 %r" % e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
